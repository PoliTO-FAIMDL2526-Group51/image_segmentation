import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, yaml
from PIL import Image
from torch.amp import autocast
from sklearn.metrics import average_precision_score, roc_curve
import scores

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METHODS = ["msp", "max_logit", "max_entropy", "rba"]

def gt_road_anomaly(g): valid=(g==0)|(g==2); return (g==2).astype(np.uint8), valid
def gt_standard(g):     valid=(g==0)|(g==1); return (g==1).astype(np.uint8), valid

DATASETS = {
    "RoadAnomaly":       dict(img_ext="jpg",  gt=gt_road_anomaly),
    "RoadAnomaly21":     dict(img_ext="png",  gt=gt_standard),
    "RoadObsticle21":    dict(img_ext="webp", gt=gt_standard),
    "fs_static":         dict(img_ext="jpg",  gt=gt_standard),
    "FS_LostFound_full": dict(img_ext="png",  gt=gt_standard),
}

def aggregate_T(mask_logits, class_logits, T):
    """S = Σ_q σ(mask_q)·softmax(class_q / T)[:-1]"""
    p = (class_logits / T).softmax(dim=-1)[..., :-1]
    m = mask_logits.sigmoid()
    return torch.einsum("bqc,bqhw->bchw", p, m)

def build_model(kind, ckpt, eomt_root):
    os.chdir(eomt_root); sys.path.insert(0, eomt_root)
    from models.eomt import EoMT
    from models.vit import ViT
    from training.mask_classification_semantic import MaskClassificationSemantic
    from training.mask_classification_panoptic import MaskClassificationPanoptic

    if kind == "cityscapes":
        cfg="configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
        Cls,n,size,q,extra = MaskClassificationSemantic,19,(1024,1024),100,{}
    elif kind in ("ft_head", "ft_last_block"):
        cfg="configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
        Cls,n,size,q,extra = MaskClassificationSemantic,19,(640,640),200,{}
    else:
        cfg="configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml"
        c=yaml.safe_load(Path(eomt_root,cfg).read_text())
        Cls,n,size,q = MaskClassificationPanoptic,133,(640,640),200
        extra={"stuff_classes": c["data"]["init_args"]["stuff_classes"]}

    c=yaml.safe_load(Path(eomt_root,cfg).read_text())
    margs={k:v for k,v in c["model"]["init_args"].items() if k!="network"}
    enc=ViT(img_size=size, backbone_name="vit_base_patch14_reg4_dinov2")
    net=EoMT(encoder=enc, num_classes=n, num_q=q, num_blocks=3, masked_attn_enabled=False)
    model=Cls(img_size=size, num_classes=n, network=net, **margs, **extra)

    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not any(k.startswith("network.") for k in sd):
        sd = {f"network.{k}": v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if not k.startswith("criterion.")]
    assert not bad and not res.unexpected_keys, f"DOSE NOT MATCH! missing(non-criterion)={bad}, unexpected={res.unexpected_keys}"
    return model.eval().to(DEVICE)

def image_tensor(pil):
    arr=np.asarray(pil.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(arr).permute(2,0,1).to(DEVICE)

def resize(img, size):
    return F.interpolate(img[None], size=size, mode="bilinear", align_corners=False)[0]

def infer_S(model, img, temp=1.0):
    """cityscapes/semantic: scale = max + cut window -> S [C,H,W]。"""
    h,w=img.shape[-2:]
    sc=max(model.img_size[0]/h, model.img_size[1]/w)
    nh,nw=round(h*sc),round(w*sc); img=resize(img,(nh,nw))

    crops,origins=[],[]
    n=int(np.ceil(max(nh,nw)/min(model.img_size)))
    ov=n*min(model.img_size)-max(nh,nw); step=min(model.img_size)-(ov/(n-1) if ov>0 else 0)

    for i in range(n):
        s=int(i*step); e=s+min(model.img_size)
        crops.append(img[:,s:e,:] if nh>nw else img[:,:,s:e]); origins.append((s,e))
    with torch.no_grad(), autocast("cuda", enabled=DEVICE.type=="cuda", dtype=torch.float16):
        masks,classes=model(torch.stack(crops))
        masks=F.interpolate(masks[-1], model.img_size, mode="bilinear")
        logits=aggregate_T(masks, classes[-1], temp)
    full=torch.zeros((logits.shape[1],nh,nw),device=DEVICE); cnt=torch.zeros_like(full)

    for lg,(s,e) in zip(logits,origins):
        if nh>nw: full[:,s:e,:]+=lg; cnt[:,s:e,:]+=1
        else:     full[:,:,s:e]+=lg; cnt[:,:,s:e]+=1
    full=F.interpolate((full/cnt.clamp_min(1))[None],(h,w),mode="bilinear")[0]
    return full.float()

def infer_S_pad(model, img, temp=1.0):
    """coco/panoptic: scale = min + pad, single forward -> S [C,H,W]。"""
    h,w=img.shape[-2:]
    sc=min(model.img_size[0]/h, model.img_size[1]/w)
    nh,nw=round(h*sc),round(w*sc)
    x=resize(img,(nh,nw))
    x=F.pad(x,[0, model.img_size[1]-nw, 0, model.img_size[0]-nh])

    with torch.no_grad(), autocast("cuda", enabled=DEVICE.type=="cuda", dtype=torch.float16):
        masks,classes=model(x[None])
        masks=F.interpolate(masks[-1], model.img_size, mode="bilinear")
        masks=F.interpolate(masks[:,:,:nh,:nw], (h,w), mode="bilinear")
        logits=aggregate_T(masks, classes[-1], temp)
    return logits[0].float()

def fpr_at_95_tpr(s,y):
    fpr,tpr,_=roc_curve(y,s); v=fpr[tpr>=0.95]; return float(np.min(v)) if len(v) else 1.0

def score_map(S,m):
    return scores.rba_score(S) if m=="rba" else scores.compute_anomaly_score(S,m)

def evaluate(model, data_root, name, infer_fn, temp=1.0):
    cfg=DATASETS[name]; d=Path(data_root)/name
    paths=sorted(glob.glob(str(d/"images"/f"*.{cfg['img_ext']}")))
    if not paths:
        print(f"[{name}] ZERO image: {d}/images/*.{cfg['img_ext']}"); return None
    
    acc={m:[] for m in METHODS}; labels=[]
    for i,p in enumerate(paths):
        gt_path=str(Path(p.replace("images","labels_masks")).with_suffix(".png"))
        S=infer_fn(model, image_tensor(Image.open(p)), temp)
        y,valid=cfg["gt"](np.array(Image.open(gt_path)))
        labels.append(y[valid])
        for m in METHODS: acc[m].append(score_map(S,m).cpu().numpy()[valid])
        print(f"  [{name} T={temp}] {i+1}/{len(paths)}", end="\r")
    y=np.concatenate(labels)
    
    out={"dataset":name,"temp":temp,"num_images":len(paths),"anomaly_frac":float(y.mean())}
    print(f"\n[{name} T={temp}] {len(paths)} imgs, anomaly {y.mean()*100:.2f}%")
    for m in METHODS:
        s=np.concatenate(acc[m])
        out[m]={"AuPRC":average_precision_score(y,s)*100,"FPR95":fpr_at_95_tpr(s,y)*100}
        print(f"  {m:12s} AuPRC={out[m]['AuPRC']:.2f}  FPR95={out[m]['FPR95']:.2f}")
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--eomt-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", choices=["cityscapes","coco","ft_head", "ft_last_block"], required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--datasets", nargs="+", default=["RoadAnomaly"])
    ap.add_argument("--temps", nargs="+", type=float, default=[1.0])
    ap.add_argument("--out", default="results")
    a=ap.parse_args()
    out_dir=Path(a.out).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    model=build_model(a.kind, a.ckpt, a.eomt_root)
    infer_fn = infer_S_pad if a.kind=="coco" else infer_S
    for T in a.temps:
        for name in a.datasets:
            fn=out_dir/f"{a.kind}_{name}_T{T}.json"
            if fn.exists():
                print(f"  skip (exists) {fn.name}"); continue
            res=evaluate(model, a.data_root, name, infer_fn, T)
            if res is None: continue
            fn.write_text(json.dumps(res, indent=2))
            print(f"  saved -> {fn}")

if __name__=="__main__":
    main()
