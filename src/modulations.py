"""변조별 별자리 + 랜덤 심볼. 리스트 순서가 클래스 번호."""

import numpy as np

# 실험에 넣을 변조들 (순서 건드리면 라벨 매핑 전부 갈림)
MODULATION_NAMES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM"]
# 문자열 → 0..N-1
MOD_TO_IDX = {name: idx for idx, name in enumerate(MODULATION_NAMES)}
# Linear 출력 개수 등에서 쓰려고 한 번에 export
NUM_MODULATION_CLASSES = len(MODULATION_NAMES)


def _bpsk_constellation() -> np.ndarray:
    return np.array([-1 + 0j, 1 + 0j], dtype=np.complex64)  # ±1


def _qpsk_constellation() -> np.ndarray:
    points = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64)
    return points / np.sqrt(2.0)  # 평균전력 1


def _8psk_constellation() -> np.ndarray:
    k = np.arange(8)
    return np.exp(1j * 2 * np.pi * k / 8).astype(np.complex64)  # 단위원 8등분


def _16qam_constellation() -> np.ndarray:
    levels = np.array([-3, -1, 1, 3], dtype=np.float32)
    points = np.array([i + 1j * q for i in levels for q in levels], dtype=np.complex64)
    return points / np.sqrt(10.0)  # 정규화


def _64qam_constellation() -> np.ndarray:
    # 8-PAM 홀수 레벨, 평균 에너지 맞추려면 42로 나눔
    levels = np.arange(-7, 8, 2, dtype=np.float32)
    points = np.array([i + 1j * q for i in levels for q in levels], dtype=np.complex64)
    return points / np.sqrt(42.0) # 정규화


def _256qam_constellation() -> np.ndarray:
    levels = np.arange(-15, 16, 2, dtype=np.float32)  # 16-PAM
    points = np.array([i + 1j * q for i in levels for q in levels], dtype=np.complex64)
    return points / np.sqrt(170.0)  # 정규화


# import 시점에 한 번만 계산해 두기
CONSTELLATIONS = {
    "BPSK": _bpsk_constellation(),
    "QPSK": _qpsk_constellation(),
    "8PSK": _8psk_constellation(),
    "16QAM": _16qam_constellation(),
    "64QAM": _64qam_constellation(),
    "256QAM": _256qam_constellation(),
}


def random_symbols(mod_name: str, length: int, rng: np.random.Generator) -> np.ndarray:
    constellation = CONSTELLATIONS[mod_name]  # 복소 벡터 
    # 타임스텝마다 별자리에서 점 하나
    idx = rng.integers(low=0, high=len(constellation), size=length)
    return constellation[idx]
