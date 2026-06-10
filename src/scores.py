import torch

def compute_anomaly_score(logits, method):

    if method == "max_logit":
        return -torch.max(logits, dim=0).values

    probs = torch.softmax(logits, dim=0)
    if method == "msp":
        return 1.0 - torch.max(probs, dim=0).values

    if method == "max_entropy":
        entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=0)
        return entropy / torch.log(
            torch.tensor(probs.shape[0], device=probs.device, dtype=probs.dtype)
        )

    raise ValueError(f"Unknown method: {method}")


def rba_score(S):
    return -S.tanh().sum(dim=0)
