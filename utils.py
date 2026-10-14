import torch




def eval_metrics(preds, gts):
    valid_mask = (gts > 0).detach()
    if valid_mask.sum() == 0:       # Invalid batch, zero out invalid metrics
        return 0.0, 0.0

    preds = torch.clamp(preds[valid_mask], min=1e-3)    # Avoid zero div
    gts = gts[valid_mask]

    rmse = torch.sqrt(torch.mean((gts - preds)**2))
    max_ratio = torch.max(preds/gts, gts/preds)
    delta1 = float((max_ratio < 1.25).float().mean())

    return rmse.item(), delta1


# a = torch.randn(1, 3, 224, 224)
# b = torch.randn(1, 3, 224, 224)
# print(eval_metrics(a, b))