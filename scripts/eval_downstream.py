"""인코더 얼리고 선형층만 학습. 라벨 얼마나 쓰는지 ratio로 스윕."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import IQDataset
from src.metrics import accuracy, confusion_matrix, snr_accuracy
from src.models import CNNEncoder
from src.modulations import NUM_MODULATION_CLASSES


def format_seconds(total_seconds: float) -> str:
    total_seconds = max(0, int(total_seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes:02d}m {seconds:02d}s"


class FrozenEncoderClassifier(nn.Module):
    """Linear probe: frozen encoder + trainable linear layer."""
    def __init__(
        self,
        encoder,
        feat_dim: int = 128,
        num_classes: int = NUM_MODULATION_CLASSES,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():  # Encoder is frozen
            feats = self.encoder(x)
        return self.classifier(feats)  # Only linear layer is trainable


def stratified_subset_indices(y: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for c in np.unique(y):  # 클래스마다 따로 샘플
        idx = np.where(y == c)[0]
        n_c = max(1, int(len(idx) * ratio))  # 너무 작아지면 1개는 남김
        selected.append(rng.choice(idx, size=n_c, replace=False))
    out = np.concatenate(selected)
    rng.shuffle(out)  # 순서 섞기
    return out


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_true.append(y.numpy())
            y_pred.append(pred)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return accuracy(y_true, y_pred), y_true, y_pred


def run_single_ratio(
    ratio: float,
    data_dir: str,
    encoder_path: str,
    out_dir: str,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    num_workers: int,
    device: torch.device,
    tb_root_dir: str,
) -> dict:
    train_all = np.load(os.path.join(data_dir, "train.npz"))
    train_y = train_all["y"].astype(np.int64)

    indices: Optional[np.ndarray]
    if ratio >= 0.999:  # 전부 쓰는 경우 (float 비교 여유)
        indices = None
    else:
        indices = stratified_subset_indices(train_y, ratio=ratio, seed=seed)

    train_ds = IQDataset(os.path.join(data_dir, "train.npz"), labeled=True, indices=indices)
    val_ds = IQDataset(os.path.join(data_dir, "val.npz"), labeled=True)
    test_ds = IQDataset(os.path.join(data_dir, "test.npz"), labeled=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    encoder = CNNEncoder(in_ch=2, feat_dim=128)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False  # 인코더 고정

    model = FrozenEncoderClassifier(
        encoder=encoder, feat_dim=128, num_classes=NUM_MODULATION_CLASSES
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)  # linear만 업데이트

    best_val, best_state = -1.0, None
    history = []
    epoch_times = []
    writer = None
    ratio_tag = f"{int(ratio * 100)}pct"
    writer = SummaryWriter(log_dir=os.path.join(tb_root_dir, ratio_tag))
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss, total = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            total += x.size(0)

        train_loss = running_loss / max(total, 1)
        val_acc, _, _ = evaluate(model, val_loader, device)
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = float(np.mean(epoch_times))
        eta_seconds = avg_epoch_time * (epochs - epoch)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        print(
            f"[ratio={ratio:.2f}] [{epoch:03d}/{epochs:03d}] "
            f"train_loss={train_loss:.4f} val_acc={val_acc:.4f} "
            f"epoch_time={epoch_time:.1f}s avg_epoch={avg_epoch_time:.1f}s "
            f"eta={format_seconds(eta_seconds)}"
        )
        writer.add_scalar("downstream/train_loss", train_loss, epoch)
        writer.add_scalar("downstream/val_acc", val_acc, epoch)
        writer.add_scalar("downstream/epoch_time_sec", epoch_time, epoch)
        writer.add_scalar("downstream/avg_epoch_time_sec", avg_epoch_time, epoch)
        writer.add_scalar("downstream/eta_sec", eta_seconds, epoch)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}  # GPU에 안 묶이게

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    test_npz = np.load(os.path.join(data_dir, "test.npz"))
    snr_db = test_npz["snr_db"].astype(np.float32)
    snr_acc = snr_accuracy(y_true, y_pred, snr_db)
    cm = confusion_matrix(y_true, y_pred, num_classes=NUM_MODULATION_CLASSES)

    ckpt_path = os.path.join(out_dir, f"downstream_{ratio_tag}.pt")
    torch.save(model.state_dict(), ckpt_path)
    writer.add_scalar("downstream/test_acc", test_acc, 0)
    for snr_val, acc_val in snr_acc.items():
        writer.add_scalar(f"downstream/test_snr_acc/snr_{snr_val}", acc_val, 0)
    writer.close()

    return {
        "ratio": ratio,
        "n_train_samples": len(train_ds),
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "snr_acc": snr_acc,
        "confusion_matrix": cm.tolist(),
        "checkpoint": ckpt_path,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/generated")
    parser.add_argument("--encoder_path", type=str, default="outputs/ssl/ssl_encoder.pt")
    parser.add_argument("--out_dir", type=str, default="outputs/downstream")
    parser.add_argument("--label_ratios", type=str, default="1.0,0.2,0.1")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tb_dir", type=str, default="", help="TensorBoard root directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ratios = [float(x) for x in args.label_ratios.split(",")]
    tb_root_dir = args.tb_dir if args.tb_dir else os.path.join(args.out_dir, "tb")
    results = []
    for ratio in ratios:
        # 100% 라벨 (ratio=1.0) 평가는 의미 없음: Supervised Learning이 더 직접적
        # SSL의 가치는 라벨 부족 시나리오(10%, 20%)에서만 측정됨
        if ratio >= 0.999:
            print(f"[ratio={ratio:.2f}] Skipped (full label not meaningful for SSL evaluation)")
            continue
        
        result = run_single_ratio(
            ratio=ratio,
            data_dir=args.data_dir,
            encoder_path=args.encoder_path,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed + int(ratio * 1000),  # ratio마다 다른 서브셋
            num_workers=args.num_workers,
            device=device,
            tb_root_dir=tb_root_dir,
        )
        results.append(result)
        print(f"[ratio={ratio:.2f}] test_acc={result['test_acc']:.4f}")

    summary = {"encoder_path": args.encoder_path, "results": results}
    summary_path = os.path.join(args.out_dir, "downstream_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved downstream summary: {summary_path}")


if __name__ == "__main__":
    main()
