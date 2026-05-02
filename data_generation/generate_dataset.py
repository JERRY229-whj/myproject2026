"""합성 IQ 만들어서 train/val/test npz로 떨구기. 루트에서 실행 (src path)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np

from src.channel_awgn import add_awgn
from src.modulations import MOD_TO_IDX, MODULATION_NAMES, random_symbols


def build_split(
    mods: list[str],
    snrs: list[float],
    samples_per_condition: int,
    length: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    # 샘플을 리스트에 쌓다가 마지막에 np.stack
    x_list, y_list, snr_list, mod_idx_list = [], [], [], []

    for mod in mods:
        for snr_db in snrs:
            for _ in range(samples_per_condition):
                s = random_symbols(mod, length, rng)  # TX 심볼
                r = add_awgn(s, snr_db=snr_db, rng=rng)  # RX (노이즈 섞임)
                # 수신 쪽 I/Q 텐서 [2, L]
                iq = np.stack([r.real, r.imag], axis=0).astype(np.float32)
                x_list.append(iq)
                y_list.append(MOD_TO_IDX[mod])
                snr_list.append(snr_db)
                mod_idx_list.append(MOD_TO_IDX[mod])

    return {
        "X": np.stack(x_list, axis=0),  # [N,2,L]
        "y": np.asarray(y_list, dtype=np.int64),
        "snr_db": np.asarray(snr_list, dtype=np.float32),
        "mod_idx": np.asarray(mod_idx_list, dtype=np.int64),
    }


def split_train_val(data: dict[str, np.ndarray], val_ratio: float, seed: int) -> tuple[dict, dict]:
    n = data["X"].shape[0]  # 샘플 수
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)  # 섞인 인덱스
    n_val = int(n * val_ratio)  # val 개수
    val_idx, train_idx = indices[:n_val], indices[n_val:]  # 앞=val 뒤=train
    # 같은 행 인덱스를 X,y,snr… 전부에 적용
    train = {k: v[train_idx] for k, v in data.items()}
    val = {k: v[val_idx] for k, v in data.items()}
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/generated")  # 저장 폴더
    parser.add_argument("--length", type=int, default=128)  # L
    parser.add_argument("--samples_per_condition", type=int, default=1000)  # (mod,snr)당
    parser.add_argument("--train_snrs", type=str, default="8,12,16")
    parser.add_argument("--test_snrs", type=str, default="-4,0,4")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)  # 없으면 생성
    # modulations랑 목록 맞출 것
    mods = list(MODULATION_NAMES)
    train_snrs = [float(x) for x in args.train_snrs.split(",")]  # 문자열 파싱
    test_snrs = [float(x) for x in args.test_snrs.split(",")]

    rng = np.random.default_rng(args.seed)  # 데이터 난수 (train블록→test까지 이어짐)
    train_val = build_split(mods, train_snrs, args.samples_per_condition, args.length, rng)
    # train/val 나누는 셔플은 seed+1
    train, val = split_train_val(train_val, val_ratio=args.val_ratio, seed=args.seed + 1)
    test = build_split(mods, test_snrs, args.samples_per_condition, args.length, rng)

    np.savez(os.path.join(args.out_dir, "train.npz"), **train)  # 키가 npz 배열 이름
    np.savez(os.path.join(args.out_dir, "val.npz"), **val)
    np.savez(os.path.join(args.out_dir, "test.npz"), **test)

    print("Saved:")  # shape만 확인용
    print(f"- {os.path.join(args.out_dir, 'train.npz')} : {train['X'].shape}")
    print(f"- {os.path.join(args.out_dir, 'val.npz')} : {val['X'].shape}")
    print(f"- {os.path.join(args.out_dir, 'test.npz')} : {test['X'].shape}")


if __name__ == "__main__":
    main()  
