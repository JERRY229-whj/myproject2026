"""지도학습 CNN. val acc 최고일 때 가중치 저장."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import IQDataset
from src.metrics import accuracy, confusion_matrix, snr_accuracy
from src.modulations import NUM_MODULATION_CLASSES
from src.models import SupervisedCNN


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()  # BN,Dropout 있으면 eval 모드
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()  # 최대 로짓 = 예측 클래스
            y_true.append(y.numpy())
            y_pred.append(pred)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return accuracy(y_true, y_pred), y_true, y_pred


def format_seconds(total_seconds: float) -> str:
    total_seconds = max(0, int(total_seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes:02d}m {seconds:02d}s"


def stratified_subset_indices(y: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """Stratified sampling: maintain class distribution."""
    rng = np.random.default_rng(seed)
    selected = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n_c = max(1, int(len(idx) * ratio))
        selected.append(rng.choice(idx, size=n_c, replace=False))
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/generated")  # npz 폴더
    parser.add_argument("--out_dir", type=str, default="outputs/sl")  # 체크포인트·json
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label_ratio", type=float, default=1.0, help="Fraction of training labels to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tb_dir", type=str, default="", help="TensorBoard log directory")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load training data and stratified sampling if needed
    train_all = np.load(os.path.join(args.data_dir, "train.npz"))
    train_y = train_all["y"].astype(np.int64)
    
    indices = None
    if args.label_ratio < 0.999:  # float comparison tolerance
        indices = stratified_subset_indices(train_y, ratio=args.label_ratio, seed=args.seed)
        ratio_tag = f"_{int(args.label_ratio * 100)}pct"
        args.out_dir = args.out_dir + ratio_tag
        os.makedirs(args.out_dir, exist_ok=True)
    
    train_ds = IQDataset(os.path.join(args.data_dir, "train.npz"), labeled=True, indices=indices)
    val_ds = IQDataset(os.path.join(args.data_dir, "val.npz"), labeled=True)
    test_ds = IQDataset(os.path.join(args.data_dir, "test.npz"), labeled=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = SupervisedCNN(num_classes=NUM_MODULATION_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    history = []
    epoch_times = []
    best_path = os.path.join(args.out_dir, "sl_best.pt")
    tb_dir = args.tb_dir if args.tb_dir else os.path.join(args.out_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()  # 학습 모드
        running_loss, total = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()  # 지난 step grad 비우기
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)  # 배치 손실 합
            total += x.size(0)

        train_loss = running_loss / max(total, 1)
        val_acc, _, _ = evaluate(model, val_loader, device)
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = float(np.mean(epoch_times))
        eta_seconds = avg_epoch_time * (args.epochs - epoch)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        print(
            f"[{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.4f} val_acc={val_acc:.4f} "
            f"epoch_time={epoch_time:.1f}s avg_epoch={avg_epoch_time:.1f}s "
            f"eta={format_seconds(eta_seconds)}"
        )
        writer.add_scalar("sl/train_loss", train_loss, epoch)
        writer.add_scalar("sl/val_acc", val_acc, epoch)
        writer.add_scalar("sl/epoch_time_sec", epoch_time, epoch)
        writer.add_scalar("sl/avg_epoch_time_sec", avg_epoch_time, epoch)
        writer.add_scalar("sl/eta_sec", eta_seconds, epoch)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)  # 가중치만

    model.load_state_dict(torch.load(best_path, map_location=device))  # best로 복구
    test_acc, y_true, y_pred = evaluate(model, test_loader, device)

    test_npz = np.load(os.path.join(args.data_dir, "test.npz"))
    snr_db = test_npz["snr_db"]
    snr_acc = snr_accuracy(y_true, y_pred, snr_db)
    cm = confusion_matrix(y_true, y_pred, num_classes=NUM_MODULATION_CLASSES)

    summary = {
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "snr_acc": snr_acc,
        "confusion_matrix": cm.tolist(),
        "history": history,
    }
    with open(os.path.join(args.out_dir, "sl_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Metrics saved to: {os.path.join(args.out_dir, 'sl_metrics.json')}")
    writer.add_scalar("sl/test_acc", test_acc, 0)
    for snr_val, acc_val in snr_acc.items():
        writer.add_scalar(f"sl/test_snr_acc/snr_{snr_val}", acc_val, 0)
    writer.close()


if __name__ == "__main__":
    main()
