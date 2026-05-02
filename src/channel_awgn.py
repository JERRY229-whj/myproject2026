"""AWGN만 섞은 채널. 심볼 평균전력 1, Es/N0 기준."""

import numpy as np


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    snr_linear = 10 ** (snr_db / 10.0)  # dB → 선형
    # I·Q 따로 넣는 구현이랑 맞춰둔 분산값
    noise_var = 1.0 / (2.0 * snr_linear)
    # 실수축 잡음
    noise_i = rng.normal(0.0, np.sqrt(noise_var), size=signal.shape)
    # 허수축 잡음 
    noise_q = rng.normal(0.0, np.sqrt(noise_var), size=signal.shape)
    # I + jQ
    noise = noise_i + 1j * noise_q
    # complex64로 맞춰서 나중에 .real/.imag 때리기 편하게
    return (signal + noise).astype(np.complex64)
