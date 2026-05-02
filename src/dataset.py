"""npz 읽어서 torch Dataset. DataLoader에 넣는 용도."""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class IQDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        labeled: bool = True,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        # npz 안의 배열들 로드
        data = np.load(npz_path, allow_pickle=True)
        x = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64) if "y" in data else None
        snr = data["snr_db"].astype(np.float32) if "snr_db" in data else None

        if indices is not None:
            x = x[indices]  # 부분 데이터만
            y = y[indices] if y is not None else None
            snr = snr[indices] if snr is not None else None

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y) if y is not None else None
        self.snr = torch.from_numpy(snr) if snr is not None else None
        self.labeled = labeled  # SSL이면 False

    def __len__(self) -> int:
        return self.x.shape[0]  # N

    def __getitem__(self, idx: int):
        if self.labeled and self.y is not None:
            return self.x[idx], self.y[idx]
        return self.x[idx]  # 라벨 없이 특성만
