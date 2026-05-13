# 한국어 일상 대화 연결

> AI 말평 대회 · 팀 프로젝트  · 2025  
> GitHub: [RYUSEOKHO1050/NLP-malpyeong](https://github.com/RYUSEOKHO1050/NLP-malpyeong)

---

## 한 줄 요약

한국어 일상 대화에서 **대상 발화**가 주어졌을 때, 그 앞(선행) 또는 뒤(후행)에 올 수 있는
**가장 자연스러운 발화를 3개의 후보 중 선택**하는 Multiple Choice NLP 과제.

**Pair-wise 이진 분류 + 특수 토큰 + 5-Fold CV + 양방향 앙상블**으로 Accuracy 75.73 → **87.45** 달성.

---

## 담당 역할

- 모델 아키텍처 설계 및 구현 (Exp B 시리즈)
- 학습/추론/앙상블 파이프라인 구현 (`src/` 전체)
- 추가 실험 3종 설계 및 수행 (Exp1·2·3)

*(팀원: 데이터 증강 + KcBERT·KoELECTRA 앙상블 실험(Exp A 시리즈))*

---

## 기술 스택

| 분류 | 내용 |
|------|------|
| 모델 | klue/roberta-large (1024-dim) |
| 프레임워크 | PyTorch, HuggingFace Transformers |
| 정규화 | Dropout(0.3) · Label Smoothing(0.1) · Gradient Clipping |
| 검증 전략 | Stratified 5-Fold CV |
| 앙상블 | Soft Voting (확률 기반 평균) |

---

## 핵심 설계 결정

### ① Pair-wise 이진 분류로 문제 재정의

3-class softmax는 세 선택지를 상대 비교해 학습 편향이 생길 수 있다.
대상발화와 후보를 **1:1로 독립 평가**하는 방식으로 재정의 → 각 선택지를 동등한 기준으로 판단.

```
(대상발화 + 발화1) → P = 0.23
(대상발화 + 발화2) → P = 0.71  ← argmax 선택
(대상발화 + 발화3) → P = 0.18
```

### ② EDA 기반 특수 토큰 설계

EDA에서 부정 표현(~52%)이 의미 대조의 핵심 단서임을 확인.
소규모 데이터(600개)에서 모델이 스스로 학습하기 어렵다 판단 → 사전 분석 결과를 토큰으로 직접 주입.

| 토큰 | 의미 |
|------|------|
| `[PREV]` / `[NEXT]` | 발화 위치 |
| `[POS]` / `[NEG]` / `[NEU]` | 감정 |
| `[NEGT]` / `[NONEGT]` | 부정 표현 여부 |

### ③ 선행문·후행문 모델 분리

EDA에서 발화위치별 정답 분포가 다름을 확인 (선행문 발화1 선호 37.6% vs 후행문 균등).
하나의 모델에 섞으면 편향 학습 위험 → **위치별 독립 모델 학습**.

### ④ 5-Fold CV 파이프라인

단일 train/val split 실험에서 Val Acc 97%, 리더보드 81% 역전 현상 발견 → 과적합 확인.
subprocess 기반 fold별 독립 학습 스크립트 구현, 10개 모델(선행 5 + 후행 5) 생성.

---

## 시스템 구조도

```
입력 (대상발화 + 발화위치 + 후보 3개)
    │
    ▼
전처리
  감정 분석 → [POS]/[NEG]/[NEU]
  부정 감지 → [NEGT]/[NONEGT]
  위치 태깅 → [PREV]/[NEXT]
    │
    ▼
Pair-wise 분해 (1개 샘플 → 3개 독립 샘플)
    │
    ▼
klue/roberta-large
  [CLS] → Dropout(0.3) → Linear(1024→2) → P(정답)
    │
    ├── 선행문 모델 × 5 (5-Fold CV)
    └── 후행문 모델 × 5 (5-Fold CV)
            │
            ▼
양방향 앙상블 (Exp1)
  선행문 샘플 → [선행문 모델 5] + [후행문 모델 5 역방향]
  후행문 샘플 → [후행문 모델 5] + [선행문 모델 5 역방향]
  Soft Voting → 최종 예측
```

---

## 구현 세부사항 (코드 기준)

| 파일 | 구현 내용 |
|------|----------|
| `src/model.py` | `DialogueConnectionModel` 클래스 — RoBERTa [CLS] → Dropout → Linear(1024→2), Label Smoothing |
| `src/data_loader.py` | Pair-wise 샘플 생성, 부정 표현 감지(`NegationDetector`), 감정 분석(`EmotionAnalyzer`), 특수 토큰 주입 |
| `src/train.py` | `EarlyStopping` 클래스, AdamW + Linear Warmup Scheduler, Gradient Clipping, 체크포인트 저장 |
| `src/train_cross_validation.py` | subprocess 기반 5-Fold CV 오케스트레이션, fold별 독립 프로세스 학습 |
| `src/ensemble_inference_cv.py` | `ReversedTestDialogueDataset` — 발화 순서 반전으로 양방향 앙상블 구현, Soft Voting |

---

## 실험 결과

| 실험 | 핵심 변경 | 리더보드 | Δ |
|------|----------|:-------:|:---:|
| Baseline | KcBERT | 75.73 | — |
| Exp B-1 | RoBERTa + 특수토큰 | 82.22 | +6.49 |
| Exp B-2 | 역번역 데이터만 | 63.60 | −12.13 |
| Exp B-3 | 원본+증강, CV 없음 | 81.17 | 과적합 |
| Exp B-4 | 원본+증강 + CV | 85.98 | +4.81 |
| Exp B-5 | 원본 + CV | 86.19 | +0.21 |
| Exp2 | 외부 말뭉치 추가 | 82.01 | −4.18 |
| Exp3 | ML 기반 감성 토큰 | 86.40 | −1.05 *(이전 최고 대비)* |
| **Exp1** | **양방향 앙상블** | **87.45** | **+1.26** ✅ |

**핵심 인사이트 3가지**

- **데이터 품질 > 양** — 역번역 증강이 구어체 특성을 희석, 원본 600개가 더 효과적
- **구조적 재설계** — 추가 학습 없이 추론 방식만 바꿔 +1.26%p (Exp1)
- **도메인 정합성** — 외부 데이터/감성 모델 모두 도메인 불일치로 역효과
