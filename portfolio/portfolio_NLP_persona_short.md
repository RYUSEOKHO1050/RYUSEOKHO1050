# Cyberpunk 2077 구매 의사 시뮬레이션 (Multi-Agent RAG)

> Multi-Agent RAG 연구 · 팀 프로젝트 · 2025  
> GitHub: [RYUSEOKHO1050/NLP-Persona](https://github.com/RYUSEOKHO1050/NLP-Persona)

---

## 한 줄 요약

LLM 기반 104명의 페르소나 에이전트가 **시계열 Steam 리뷰**에 노출될 때
대중의 구매 의사 변화를 얼마나 정확히 시뮬레이션할 수 있는지 검증.

**Time-Decay 가중 검색 + Candidate Expansion(300개)** 도입으로
Static RAG 대비 Pearson 상관계수 0.356 → **0.578** 달성 (SOTA).

---

## 담당 역할

- Static RAG 설계·구현 (`static_rag/`)
- Time-Aware RAG 설계·구현 (`time_aware_rag/`)

*(팀원: Zero-Shot Baseline 구현, 페르소나 생성 시스템, 쿼리 전략, LLM 설정 모듈)*

---

## 기술 스택

| 분류 | 내용 |
|------|------|
| LLM | `gpt-4o-mini` (Temperature 0.5) |
| Vector DB | ChromaDB (`PersistentClient`, batch_size=512) |
| 임베딩 (비교) | `all-MiniLM-L6-v2` vs `text-embedding-3-small` |
| 프레임워크 | Python, OpenAI API |
| 평가 지표 | Pearson (선형 정밀도), Spearman (순위 경향성) |

---

## 핵심 설계 결정

### ① Time-Decay 가중 검색

Static RAG는 2021년 환불 사태 리뷰와 2023년 Edgerunners 찬사를 동등하게 검색한다.
시뮬레이션 날짜 기준으로 최근 리뷰에 높은 가중치를 부여하는 지수 감쇠 함수 도입.

```python
time_factor = np.exp(-0.01 * days_diff)   # 반감기 ~70일
final_score = similarity * time_factor
```

날짜 필터(`date ≤ 시뮬레이션 날짜`)로 미래 리뷰 유입도 차단.

### ② Candidate Expansion + Reranking

유사도 Top-5를 바로 반환하면 시간 감쇠 전 단계에서 최신 리뷰가 걸러질 수 있다.
300개 후보 검색 → 시간 감쇠 점수 재정렬 → 중복 제거 → Top-5 반환.

```
Static RAG:   유사도 계산 → Top-5 반환
Time-Aware:   유사도 계산 → 300개 후보 → exp(−λ×days) 재점수 → Top-5 반환
```

### ③ 페르소나 기반 쿼리 전략 (8 유형 × 10개 + 공통 1개)

게이머 유형별로 10개의 전문화된 쿼리를 사전 정의. 시뮬레이션마다 유형별 쿼리에서 4개를 무작위 샘플링하고 공통 쿼리 1개를 추가해 총 5개로 검색.
Ultimate Gamer는 그래픽·성능 쿼리를, Cloud Gamer는 최적화·저사양 쿼리를 검색해 **페르소나 특성이 검색 단계부터 반영**됨.

### ④ 계층화된 페르소나 설계 (ESA 2024 인구 통계 기반)

실제 게이머 분포는 Time Filler(27%) vs Conventional Player(4%)처럼 극단적으로 쏠려 있어, 실제 분포대로 샘플링하면 소수 유형 분석이 불가능.
→ 8 유형 × 13명 **균등 추출**로 각 유형의 행동 차이를 동등한 통계적 기반에서 비교. 성별(남 54%/여 46%)·연령은 ESA 2024 기반 가중 샘플링.

### ⑤ 이중 Ground Truth 설계

Steam 긍정 비율(게이머 경험)과 주가(투기 심리) 두 축으로 동시 평가.
에이전트가 **Steam과 양의 상관, 주가와 음의 상관**을 보여야 '진짜 게이머'처럼 행동한 것.

### ⑥ LLM 설정 통일

모든 실험군(Exp 1·2·3)이 동일한 `gpt-4o-mini` (Temperature 0.5)을 사용. `utils/llm_config.py`로 단일 설정을 관리해 LLM 차이가 실험 결과에 개입하는 것을 차단.

---

## 시스템 구조도

```
시뮬레이션 날짜 (53개 시점)
  핵심 이벤트 7개 앵커 + 주간(2020.12~2021.02) + 월간(2021.03~2023.12)
        │
        ▼
페르소나 생성 — 8 유형 × 13명 = 104 에이전트 (총 5,512 스텝)
        │
  유형별 쿼리 선택 (4 random + 1 general)
        │
   ┌────┴────────────────────┐
   ▼                         ▼
Exp 2: Static RAG         Exp 3: Time-Aware RAG
유사도 Top-5 직접 반환     300개 후보 → exp(−λ×days) → Top-5
        │                         │
        └──────────┬──────────────┘
                   ▼
             gpt-4o-mini
    페르소나 + 검색 리뷰 → {"decision": "YES"/"NO", "reasoning": "..."}
                   │
                   ▼
         날짜별 YES 비율 집계
                   │
                   ▼
    Pearson / Spearman vs Steam GT / Stock GT
```

---

## 구현 세부사항 (코드 기준)

| 파일 | 핵심 구현 |
|------|----------|
| `static_rag/rag_modules.py` | `RAGRetriever.retrieve_reviews()` — `collection.query()`, cosine distance → similarity 변환, `$lte` 날짜 메타 필터 |
| `static_rag/build_chroma_db.py` | `PersistentClient` 초기화, 배치 512개 단위 upsert, 날짜를 YYYYMMDD 정수로 인덱싱 |
| `time_aware_rag/rag_modules.py` | `n_results=300` 후보 확장 → `dict` 기반 텍스트 중복 제거(최고 점수 유지) → `sorted()` 재정렬 → Top-5 슬라이싱 |
| `time_aware_rag/simulation_model_c.py` | 53 날짜 × 104 에이전트 중첩 루프 (5,512 API 호출), `json.loads()` + 예외 처리, 결과 CSV 누적 저장 |
| `evaluate_correlation.py` | `pd.merge(on='Date')`로 GT 정렬, `scipy.stats.pearsonr / spearmanr` 계산, Steam·Stock 동시 출력 |

---

## 기술적 도전 & 해결

### 중복 리뷰 문제
5개 쿼리 × 300개 후보 = 최대 1,500개 후보가 생성되는데, 동일 리뷰가 여러 쿼리에서 중복 검색됨.
`dict[text] = max_score` 방식으로 텍스트 기준 최고 점수만 유지해 해결.

```python
if text not in seen or final_score > seen[text]['final_score']:
    seen[text] = {'final_score': final_score, ...}
```

### 미래 데이터 유출 차단
날짜 필터 없이 검색하면 시뮬레이션 날짜 이후 리뷰가 컨텍스트에 포함돼
미래를 아는 에이전트가 됨. ChromaDB `where` 메타 필터로 원천 차단.

```python
where={"date": {"$lte": current_date_int}}  # YYYYMMDD 정수 비교
```

---

## 실험 결과

| 실험 | 아키텍처 | 임베딩 | Pearson | Spearman | Stock 상관 |
|------|---------|--------|:-------:|:--------:|:----------:|
| Exp 1 | Zero-Shot | — | NaN | NaN | NaN (구매율 37.5% 고정) |
| Exp 2 | Static RAG | MiniLM | 0.212 | 0.203 | -0.371 |
| Exp 2 | Static RAG | Emb-3-small | 0.356 | 0.379 | -0.479 |
| Exp 3 | Time-Aware | MiniLM | 0.462 | **0.568** | -0.362 |
| **Exp 3** | **Time-Aware** | **Emb-3-small** | **0.578** | 0.547 | **-0.578** ✅ |

**핵심 인사이트 3가지**

- **아키텍처 > 임베딩** — MiniLM 고정 시 아키텍처 개선 효과(+0.250)가 임베딩 개선(+0.116)의 2배
- **Spearman → Pearson 순서** — Time-Aware가 경향성(방향)을 잡고, 고품질 임베딩이 강도(정밀도)를 정교화
- **주가 음의 상관(-0.578)** — 에이전트가 투기 심리가 아닌 실제 게임 경험에 반응한 무결성 증명
