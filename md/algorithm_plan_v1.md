# 알고리즘 계획서 (V1)

대상 범위: 데이터 생성 / Supervised 학습 / SSL 학습  
실험 범위: Symbol-level, AWGN-only, 6-class (BPSK/QPSK/8PSK/16QAM/64QAM/256QAM)

---

## 0) 공통 설정

- Classes: `BPSK`, `QPSK`, `8PSK`, `16QAM`, `64QAM`, `256QAM`
- Sequence length: `L = 128`
- Input tensor: `X [N, 2, L]` (I/Q)
- Label: `y [N]`
- SNR 설정:
  - Train/Val: `[8, 12, 16]`
  - Test: `[-4, 0, 4]`
- SNR 정의: `Es/N0`
- Constellation 정규화: `E[|s|^2] = 1`

---

## 1) 데이터 생성 알고리즘 (AWGN, Symbol-level)

### 1-1. 도식(흐름)

`설정 로드`  
→ `modulation 선택`  
→ `랜덤 심볼/비트 생성`  
→ `constellation mapping`  
→ `전력 정규화(E[|s|^2]=1)`  
→ `AWGN 추가(snr_db 기반)`  
→ `I/Q 분리 및 [2,L] 변환`  
→ `라벨/메타 부착`  
→ `조건 반복(mod × snr × K)`  
→ `train/val/test 저장(npz)`

### 1-2. 입력

- `mods, snr_list, L, samples_per_condition`

### 1-3. 출력

- `train.npz`, `val.npz`, `test.npz`
  - `X, y, snr_db, mod_idx`

### 1-4. 핵심 연산

- `SNR = 10^(snr_db/10)`
- `nI, nQ ~ N(0, 1/(2*SNR))`
- `r = s + (nI + j*nQ)`

---

## 2) Supervised(SL) 학습 알고리즘

### 2-1. 도식(흐름)

`train/val 로드`  
→ `1D CNN 초기화`  
→ `mini-batch forward`  
→ `CrossEntropy loss`  
→ `backprop + optimizer step`  
→ `epoch 종료 시 val 평가`  
→ `best checkpoint 저장`  
→ `test 평가(전체 acc + SNR별 acc + confusion matrix)`

### 2-2. 입력

- `train.npz, val.npz, test.npz`
- 하이퍼파라미터(배치 크기, learning rate, epoch)

### 2-3. 출력

- `sl_best.pt`
- 학습 로그(`loss`, `acc`)
- 평가 지표(전체 accuracy, SNR별 accuracy, confusion matrix)

### 2-4. 모델 블록 도식

`[B,2,L]`  
→ `Conv1d(2→32)+ReLU`  
→ `Conv1d(32→64)+ReLU`  
→ `Conv1d(64→128)+ReLU`  
→ `GlobalAvgPool`  
→ `Linear(128→6)`  
→ `logits`

---

## 3) SSL 학습 알고리즘 (Contrastive, SimCLR 스타일)

### 3-1. Pretraining 도식(흐름)

`unlabeled train X 로드`  
→ `샘플 x 선택`  
→ `x1=aug(x), x2=aug(x)`  
→ `encoder + projection head`  
→ `z1, z2 임베딩 계산`  
→ `NT-Xent(InfoNCE) loss`  
→ `backprop`  
→ `encoder 저장`

### 3-2. Augmentation 도식

`x`  
→ `small AWGN perturbation`  
→ `amplitude scaling`  
→ `small phase rotation`  
→ `augmented x`

참고: 2-view는 같은 `aug()`를 독립 랜덤으로 2회 적용해 생성한다.

### 3-3. 입력

- unlabeled `X_train`
- SSL 하이퍼파라미터(`tau`, `lr`, `epoch`)

### 3-4. 출력

- `ssl_encoder.pt`

---

## 4) SSL Downstream 분류 알고리즘 (평가 단계)

### 4-1. 도식(흐름)

`ssl_encoder 불러오기`  
→ `encoder freeze`  
→ `linear classifier 초기화`  
→ `labeled subset(100/20/10%)로 학습`  
→ `test 평가(전체/SNR별/confusion matrix)`  
→ `SL 결과와 비교`

### 4-2. 입력

- `ssl_encoder.pt`
- labeled train subset
- val/test set

### 4-3. 출력

- 라벨 비율별 성능표
- SL vs SSL 비교표

---

## 5) 최종 비교 알고리즘 (결과 정리)

### 5-1. 도식(흐름)

`SL 결과 수집` + `SSL-downstream 결과 수집`  
→ `라벨 비율별 정렬(100/20/10)`  
→ `SNR별 정확도 곡선 생성`  
→ `비교 표/그래프 출력`

### 5-2. 최종 산출물

- 표1: 방법별 overall accuracy
- 표2: 방법별/라벨비율별 accuracy
- 그래프1: Accuracy vs SNR
- 그림1: confusion matrix

---

## 6) 1페이지 요약 도식 (발표용)

`[Data Gen]`  
`bits/symbols → modulation → normalize → AWGN → I/Q tensor(npz)`  
↓  
`[SL]`  
`1D CNN supervised training → test metrics`  
↓  
`[SSL]`  
`2-view augmentation → contrastive pretraining`  
↓  
`[Downstream]`  
`freeze encoder + linear classifier`  
↓  
`[Comparison]`  
`overall acc / SNR별 acc / confusion matrix`

