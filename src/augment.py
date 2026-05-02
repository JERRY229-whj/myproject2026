"""SSL augmentation utilities for IQ tensors.

Default policy:
- AWGN (50%): Weak noise reflecting RF channel reality.
- Amplitude scaling (30%): Moderate probability for realistic variations.
- Phase rotation (30%): Moderate probability.
- Guarantee: At least one augmentation is always applied.
"""

import math
import torch


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    # 채널0=I, 채널1=Q → I+jQ
    return x[:, 0, :] + 1j * x[:, 1, :]


def _from_complex(z: torch.Tensor) -> torch.Tensor:
    # 다시 [B,2,L], 같은 디바이스에 맞춰서 빈 텐서
    out = torch.zeros((z.shape[0], 2, z.shape[1]), dtype=torch.float32, device=z.device)
    out[:, 0, :] = z.real.float()
    out[:, 1, :] = z.imag.float()
    return out


def augment_batch(
    x: torch.Tensor,
    awgn_sigma: float = 0.03,
    amp_low: float = 0.95,
    amp_high: float = 1.05,
    phase_max: float = math.pi / 16,
    p_awgn: float = 0.5,  # 낮춤: 0.95 → 0.5
    p_scale: float = 0.3,  # 낮춤: 0.70 → 0.3
    p_phase: float = 0.3,  # 낮춤: 0.60 → 0.3
) -> torch.Tensor:
    z = _to_complex(x)
    b, l = z.shape
    
    # Track which augmentations are applied to ensure at least one.
    aug_applied = False

    # Apply weak complex noise with probability p_awgn.
    if torch.rand(1, device=x.device).item() < p_awgn:
        noise = awgn_sigma * (
            torch.randn((b, l), device=x.device) + 1j * torch.randn((b, l), device=x.device)
        )
        z = z + noise
        aug_applied = True

    # Apply global amplitude scaling per sample with probability p_scale.
    if torch.rand(1, device=x.device).item() < p_scale:
        scales = torch.empty((b, 1), device=x.device).uniform_(amp_low, amp_high)
        z = z * scales
        aug_applied = True

    # Apply small global phase rotation with probability p_phase.
    if torch.rand(1, device=x.device).item() < p_phase:
        theta = torch.empty((b, 1), device=x.device).uniform_(-phase_max, phase_max)
        rot = torch.cos(theta) + 1j * torch.sin(theta)
        z = z * rot
        aug_applied = True
    
    # Fallback: if no augmentation was applied, apply AWGN as default.
    if not aug_applied:
        noise = awgn_sigma * (
            torch.randn((b, l), device=x.device) + 1j * torch.randn((b, l), device=x.device)
        )
        z = z + noise

    return _from_complex(z)


def two_views(
    x: torch.Tensor,
    awgn_sigma: float = 0.03,
    amp_low: float = 0.95,
    amp_high: float = 1.05,
    phase_max: float = math.pi / 16,
    p_awgn: float = 0.5,
    p_scale: float = 0.3,
    p_phase: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Independent random augmentations for positive pairs.
    kwargs = {
        "awgn_sigma": awgn_sigma,
        "amp_low": amp_low,
        "amp_high": amp_high,
        "phase_max": phase_max,
        "p_awgn": p_awgn,
        "p_scale": p_scale,
        "p_phase": p_phase,
    }
    return augment_batch(x, **kwargs), augment_batch(x, **kwargs)
