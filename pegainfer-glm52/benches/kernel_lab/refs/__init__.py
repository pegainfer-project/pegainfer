"""Torch reference implementations (lazy torch; import-safe on CPU boxes)."""


def compute_metrics(got, want) -> dict:
    """Standard gate metrics between a kernel output and the f32 reference.

    got may be any float dtype (bf16 kernel stores); want is the f32 reference.
    """
    from kernel_lab.loader import require_torch

    torch = require_torch()
    g = got.to(torch.float32).flatten()
    w = want.to(torch.float32).flatten()
    diff = (g - w).abs()
    rel_l2 = (g - w).norm() / w.norm().clamp_min(1e-30)
    cosine = torch.nn.functional.cosine_similarity(g, w, dim=0)
    return {
        "rel_l2": float(rel_l2),
        "cosine": float(cosine),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
    }
