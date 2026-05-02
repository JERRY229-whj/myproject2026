"""NT-Xent. 같은 배치 인덱스끼리만 positive."""

import torch
import torch.nn.functional as F  # cosine_similarity, cross_entropy


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    batch_size = z1.size(0)
    # [B,D] 두 개를 세로로 concat → 2B
    z = torch.cat([z1, z2], dim=0)
    # (2B,2B) 유사도 행렬, τ로 스케일
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature

    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    # 자기 자신 끊기
    sim = sim.masked_fill(mask, -9e15)

    positive_indices = torch.arange(batch_size, device=z.device)
    # positive 열 인덱스 (위반 + 아래반 각각)
    positives = torch.cat([positive_indices + batch_size, positive_indices], dim=0)

    # 행마다 softmax CE 한 방에 처리
    return F.cross_entropy(sim, positives)
