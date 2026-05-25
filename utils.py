import torch




def eval_metrics(preds, gts):
    valid_mask = (gts > 0).detach()
    if valid_mask.sum() == 0:       # Invalid batch, zero out invalid metrics
        return 0.0, 0.0

    preds = torch.clamp(preds[valid_mask], min=1e-3)    # Avoid zero div
    gts = gts[valid_mask]

    rmse = torch.sqrt(torch.mean((preds - gts)**2))
    max_ratio = torch.max(preds/gts, gts/preds)
    delta1 = float((max_ratio < 1.25).float().mean())

    return rmse.item(), delta1.item()