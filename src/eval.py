import json
import os
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.amp import autocast


ROOT = Path(__file__).resolve().parent
EOMT = ROOT / "MaskArchitectureAnomaly_CourseProject-main" / "MaskArchitectureAnomaly_CourseProject-main" / "eomt"
OUT = ROOT / "step4_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CITY_CKPT = ROOT / "eomt_cityscapes.bin"
COCO_CKPT = ROOT / "eomt_coco.bin"
IMG_ZIP = ROOT / "leftImg8bit_trainvaltest.zip"
GT_ZIP = ROOT / "gtFine_trainvaltest.zip"

COMMON_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "vegetation", "sky",
    "person", "car", "bus", "truck", "bicycle", "motorcycle", "train",
    "traffic light",
]
COMMON_ID = {c: i for i, c in enumerate(COMMON_CLASSES)}

CITY_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
]
CITY_TO_COMMON = {i: COMMON_ID[c] for i, c in enumerate(CITY_NAMES) if c in COMMON_ID}

COCO_TO_COMMON_NAME = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    6: "train", 7: "truck", 9: "traffic light", 100: "road",
    109: "wall", 110: "wall", 111: "wall", 112: "wall",
    116: "vegetation", 117: "fence", 119: "sky", 125: "vegetation",
    129: "building", 131: "wall",
}
COCO_TO_COMMON = {k: COMMON_ID[v] for k, v in COCO_TO_COMMON_NAME.items()}

ID_TO_TRAIN_ID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
    31: 16, 32: 17, 33: 18,
}


def setup_repo_imports():
    os.chdir(EOMT)
    sys.path.insert(0, str(EOMT))
    from models.eomt import EoMT
    from models.vit import ViT
    from training.mask_classification_semantic import MaskClassificationSemantic
    from training.mask_classification_panoptic import MaskClassificationPanoptic
    return EoMT, ViT, MaskClassificationSemantic, MaskClassificationPanoptic


def build_model(kind):
    EoMTModel, ViT, Semantic, Panoptic = setup_repo_imports()
    if kind == "cityscapes":
        cfg_path = EOMT / "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
        ckpt, cls, n_cls, img_size, num_q = CITY_CKPT, Semantic, 19, (1024, 1024), 100
        extra = {}
    else:
        cfg_path = EOMT / "configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml"
        ckpt, cls, n_cls, img_size, num_q = COCO_CKPT, Panoptic, 133, (640, 640), 200
        cfg = yaml.safe_load(cfg_path.read_text())
        extra = {"stuff_classes": cfg["data"]["init_args"]["stuff_classes"]}

    cfg = yaml.safe_load(cfg_path.read_text())
    model_args = {k: v for k, v in cfg["model"]["init_args"].items() if k != "network"}
    encoder = ViT(img_size=img_size, backbone_name="vit_base_patch14_reg4_dinov2")
    net = EoMTModel(encoder=encoder, num_classes=n_cls, num_q=num_q, num_blocks=3, masked_attn_enabled=False)
    model = cls(img_size=img_size, num_classes=n_cls, network=net, **model_args, **extra)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return model.eval().to(DEVICE)


def resize(img, size):
    return F.interpolate(img[None], size=size, mode="bilinear", align_corners=False)[0]


def semantic_prediction(model, img):
    h, w = img.shape[-2:]
    scale = max(model.img_size[0] / h, model.img_size[1] / w)
    new_h, new_w = round(h * scale), round(w * scale)
    img = resize(img, (new_h, new_w))

    crops, origins = [], []
    n_crops = int(np.ceil(max(new_h, new_w) / min(model.img_size)))
    overlap = n_crops * min(model.img_size) - max(new_h, new_w)
    step = min(model.img_size) - (overlap / (n_crops - 1) if overlap > 0 else 0)

    for i in range(n_crops):
        start = int(i * step)
        end = start + min(model.img_size)
        crop = img[:, start:end, :] if new_h > new_w else img[:, :, start:end]
        crops.append(crop)
        origins.append((start, end))

    with torch.no_grad(), autocast("cuda", enabled=DEVICE.type == "cuda", dtype=torch.float16):
        masks, classes = model(torch.stack(crops))
        masks = F.interpolate(masks[-1], model.img_size, mode="bilinear")
        logits = model.to_per_pixel_logits_semantic(masks, classes[-1])

    full = torch.zeros((logits.shape[1], new_h, new_w), device=DEVICE)
    count = torch.zeros_like(full)
    for logit, (start, end) in zip(logits, origins):
        if new_h > new_w:
            full[:, start:end, :] += logit
            count[:, start:end, :] += 1
        else:
            full[:, :, start:end] += logit
            count[:, :, start:end] += 1

    full = F.interpolate((full / count.clamp_min(1))[None], (h, w), mode="bilinear")[0]
    return full.argmax(0).cpu().numpy().astype(np.uint8)


def panoptic_prediction(model, img):
    h, w = img.shape[-2:]
    scale = min(model.img_size[0] / h, model.img_size[1] / w)
    new_h, new_w = round(h * scale), round(w * scale)
    x = resize(img, (new_h, new_w))
    x = F.pad(x, [0, model.img_size[1] - new_w, 0, model.img_size[0] - new_h])

    with torch.no_grad(), autocast("cuda", enabled=DEVICE.type == "cuda", dtype=torch.float16):
        masks, classes = model(x[None])
        masks = F.interpolate(masks[-1], model.img_size, mode="bilinear")
        masks = F.interpolate(masks[:, :, :new_h, :new_w], (h, w), mode="bilinear")
        pred = model.to_per_pixel_preds_panoptic(
            [masks[0]], classes[-1], model.stuff_classes, model.mask_thresh, model.overlap_thresh
        )[0]

    pred = pred.cpu().numpy()
    return pred[:, :, 0].astype(np.int16), pred[:, :, 1].astype(np.int16)


def read_cityscapes_pairs():
    with zipfile.ZipFile(IMG_ZIP) as z:
        names = sorted(n for n in z.namelist() if n.startswith("leftImg8bit/val/") and n.endswith("_leftImg8bit.png"))
    pairs = []
    for img_name in names:
        city = Path(img_name).parent.name
        stem = Path(img_name).name.replace("_leftImg8bit.png", "")
        gt_name = f"gtFine/val/{city}/{stem}_gtFine_labelIds.png"
        pairs.append((img_name, gt_name))
    return pairs


def image_tensor(pil):
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).to(DEVICE)


def gt_to_train_id(gt):
    out = np.full(gt.shape, 255, dtype=np.uint8)
    for src, dst in ID_TO_TRAIN_ID.items():
        out[gt == src] = dst
    return out


def map_to_common(mask, mapping):
    out = np.full(mask.shape, 255, dtype=np.uint8)
    for src, dst in mapping.items():
        out[mask == src] = dst
    return out


class IoU:
    def __init__(self, n):
        self.n = n
        self.mat = np.zeros((n, n), dtype=np.int64)

    def update(self, pred, gt):
        keep = gt != 255
        pred, gt = pred[keep], gt[keep]
        keep = (pred >= 0) & (pred < self.n)
        pred, gt = pred[keep], gt[keep]
        self.mat += np.bincount(self.n * gt + pred, minlength=self.n**2).reshape(self.n, self.n)

    def result(self):
        tp = np.diag(self.mat)
        denom = self.mat.sum(0) + self.mat.sum(1) - tp
        iou = np.where(denom > 0, tp / denom, np.nan)
        return float(np.nanmean(iou)), iou


def colors(mask, n):
    rng = np.random.default_rng(0)
    palette = rng.integers(30, 240, size=(n, 3), dtype=np.uint8)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for x in np.unique(mask):
        if 0 <= x < n:
            out[mask == x] = palette[int(x)]
    return out


def panoptic_colors(sem, inst):
    out = colors(sem, 133)
    edge = np.zeros(sem.shape, dtype=bool)
    both = sem.astype(np.int64) * 100000 + inst.astype(np.int64)
    edge[1:] |= both[1:] != both[:-1]
    edge[:-1] |= both[1:] != both[:-1]
    edge[:, 1:] |= both[:, 1:] != both[:, :-1]
    edge[:, :-1] |= both[:, 1:] != both[:, :-1]
    out[edge] = 0
    return out


def save_figure(path, img, city_pred, coco_sem, coco_inst, gt):
    fig, ax = plt.subplots(1, 5, figsize=(22, 5))
    panels = [
        (np.asarray(img), "Input"),
        (colors(city_pred, 19), "Cityscapes semantic"),
        (panoptic_colors(coco_sem, coco_inst), "COCO panoptic"),
        (colors(map_to_common(coco_sem, COCO_TO_COMMON), len(COMMON_CLASSES)), "COCO mapped"),
        (colors(gt, 19), "Ground truth"),
    ]
    for a, (im, title) in zip(ax, panels):
        a.imshow(im)
        a.set_title(title)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(max_vis=5):
    OUT.mkdir(exist_ok=True)
    (OUT / "visualizations").mkdir(exist_ok=True)

    city_model = build_model("cityscapes")
    coco_model = build_model("coco")

    city_iou = IoU(len(COMMON_CLASSES))
    coco_iou = IoU(len(COMMON_CLASSES))

    pairs = read_cityscapes_pairs()
    with zipfile.ZipFile(IMG_ZIP) as img_zip, zipfile.ZipFile(GT_ZIP) as gt_zip:
        for i, (img_name, gt_name) in enumerate(pairs):
            print(f"{i + 1}/{len(pairs)} {Path(img_name).name}")

            with img_zip.open(img_name) as f:
                img = Image.open(f).convert("RGB")
                img.load()
            with gt_zip.open(gt_name) as f:
                gt = gt_to_train_id(np.asarray(Image.open(f)))

            x = image_tensor(img)
            city_pred = semantic_prediction(city_model, x)
            coco_sem, coco_inst = panoptic_prediction(coco_model, x)

            gt_common = map_to_common(gt, CITY_TO_COMMON)
            city_iou.update(map_to_common(city_pred, CITY_TO_COMMON), gt_common)
            coco_iou.update(map_to_common(coco_sem, COCO_TO_COMMON), gt_common)

            if i < max_vis:
                name = Path(img_name).name.replace("_leftImg8bit.png", "_comparison.png")
                save_figure(OUT / "visualizations" / name, img, city_pred, coco_sem, coco_inst, gt)

    city_miou, city_per_class = city_iou.result()
    coco_miou, coco_per_class = coco_iou.result()
    result = {
        "num_images": len(pairs),
        "classes": COMMON_CLASSES,
        "cityscapes_eomt_mIoU": city_miou,
        "coco_eomt_mIoU": coco_miou,
        "cityscapes_eomt_IoU": dict(zip(COMMON_CLASSES, city_per_class.tolist())),
        "coco_eomt_IoU": dict(zip(COMMON_CLASSES, coco_per_class.tolist())),
    }
    (OUT / "step4_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
