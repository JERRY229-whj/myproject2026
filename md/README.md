# RF Modulation Classification Pipeline (V1)

This repository contains a minimal symbol-level RF modulation pipeline:

- Synthetic data generation (BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM)
- AWGN channel (Es/N0)
- Supervised baseline training
- Contrastive SSL pretraining (SimCLR-style)
- Downstream frozen-encoder classifier evaluation

## Project structure

- `data_generation/generate_dataset.py`: create `train/val/test.npz`
- `scripts/train_supervised.py`: supervised baseline
- `scripts/train_ssl.py`: SSL pretraining
- `scripts/eval_downstream.py`: downstream classifier with frozen encoder
- `src/`: reusable modules (models, dataset, augment, losses, metrics)

## 1) Generate dataset

```bash
py -3 data_generation/generate_dataset.py --out_dir data/generated --length 128 --samples_per_condition 1000 --train_snrs 8,12,16 --test_snrs -4,0,4 --val_ratio 0.2
```

## 2) Train supervised baseline

```bash
py -3 scripts/train_supervised.py --data_dir data/generated --out_dir outputs/sl --epochs 30 --batch_size 256 --lr 1e-3
```

Outputs:

- `outputs/sl/sl_best.pt`
- `outputs/sl/sl_metrics.json`

## 3) SSL pretraining

```bash
py -3 scripts/train_ssl.py --data_dir data/generated --out_dir outputs/ssl --epochs 50 --batch_size 256 --lr 1e-3 --temperature 0.5
```

Outputs:

- `outputs/ssl/ssl_model_best.pt`
- `outputs/ssl/ssl_encoder.pt`
- `outputs/ssl/ssl_metrics.json`

## 4) Downstream evaluation (frozen encoder)

```bash
py -3 scripts/eval_downstream.py --data_dir data/generated --encoder_path outputs/ssl/ssl_encoder.pt --out_dir outputs/downstream --label_ratios 1.0,0.2,0.1 --epochs 30 --batch_size 256 --lr 1e-3
```

Outputs:

- `outputs/downstream/downstream_100pct.pt`
- `outputs/downstream/downstream_20pct.pt`
- `outputs/downstream/downstream_10pct.pt`
- `outputs/downstream/downstream_metrics.json`

## Notes

- Input tensor shape is `[N, 2, 128]` where channel 0/1 is I/Q.
- SNR split is intentionally mismatched (`train: high`, `test: low`).
- This is a V1 symbol-level pipeline. Pulse shaping/matched filtering and Rayleigh are out of scope.

