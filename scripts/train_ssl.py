"""SSL: 라벨 안 쓰고 NT-Xent만. 끝나면 encoder만 따로 저장."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.augment import two_views
from src.dataset import IQDataset
from src.losses import nt_xent_loss
from src.models import SSLModel


def format_seconds(total_seconds: float) -> str:
    total_seconds = max(0, int(total_seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes:02d}m {seconds:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/generated")
    parser.add_argument("--out_dir", type=str, default="outputs/ssl")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.5)  # NT-Xent τ
    parser.add_argument("--awgn_sigma", type=float, default=0.03)
    parser.add_argument("--amp_low", type=float, default=0.95)
    parser.add_argument("--amp_high", type=float, default=1.05)
    parser.add_argument("--phase_max", type=float, default=float(np.pi / 16.0))
    parser.add_argument("--p_awgn", type=float, default=0.5)
    parser.add_argument("--p_scale", type=float, default=0.3)
    parser.add_argument("--p_phase", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tb_dir", type=str, default="", help="TensorBoard log directory")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = IQDataset(os.path.join(args.data_dir, "train.npz"), labeled=False)  # y 안 씀
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    model = SSLModel(feat_dim=128, proj_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 전부 학습 (나중에 encoder만 추출)

    history = []
    best_loss = float("inf")
    epoch_times = []
    best_path = os.path.join(args.out_dir, "ssl_model_best.pt")  # 전체 state
    encoder_path = os.path.join(args.out_dir, "ssl_encoder.pt")  # backbone만
    tb_dir = args.tb_dir if args.tb_dir else os.path.join(args.out_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss, n_samples = 0.0, 0
        for x in train_loader:
            x = x.to(device)
            x1, x2 = two_views(
                x,
                awgn_sigma=args.awgn_sigma,
                amp_low=args.amp_low,
                amp_high=args.amp_high,
                phase_max=args.phase_max,
                p_awgn=args.p_awgn,
                p_scale=args.p_scale,
                p_phase=args.p_phase,
            )  # positive 쌍

            z1, z2 = model(x1), model(x2)  # L2 정규된 임베딩
            loss = nt_xent_loss(z1, z2, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

        epoch_loss = running_loss / max(n_samples, 1)
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = float(np.mean(epoch_times))
        eta_seconds = avg_epoch_time * (args.epochs - epoch)
        history.append({"epoch": epoch, "ssl_loss": epoch_loss})
        print(
            f"[{epoch:03d}/{args.epochs:03d}] "
            f"ssl_loss={epoch_loss:.4f} "
            f"epoch_time={epoch_time:.1f}s avg_epoch={avg_epoch_time:.1f}s "
            f"eta={format_seconds(eta_seconds)}"
        )
        writer.add_scalar("ssl/train_nt_xent_loss", epoch_loss, epoch)
        writer.add_scalar("ssl/epoch_time_sec", epoch_time, epoch)
        writer.add_scalar("ssl/avg_epoch_time_sec", avg_epoch_time, epoch)
        writer.add_scalar("ssl/eta_sec", eta_seconds, epoch)

        if epoch_loss < best_loss:  # SL이랑 반대로 loss 낮을수록 저장
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    torch.save(model.encoder.state_dict(), encoder_path)  # downstream은 이거 로드

    summary = {
        "best_ssl_loss": best_loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "augment": {
            "awgn_sigma": args.awgn_sigma,
            "amp_low": args.amp_low,
            "amp_high": args.amp_high,
            "phase_max": args.phase_max,
            "p_awgn": args.p_awgn,
            "p_scale": args.p_scale,
            "p_phase": args.p_phase,
        },
        "history": history,
    }
    with open(os.path.join(args.out_dir, "ssl_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved SSL model: {best_path}")
    print(f"Saved encoder: {encoder_path}")
    writer.add_scalar("ssl/best_loss", best_loss, 0)
    writer.close()


if __name__ == "__main__":
    main()
