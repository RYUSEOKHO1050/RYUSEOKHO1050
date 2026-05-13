# WOODJUDGE — 유사 판례 기반 법률 조언 AI 서비스

> RAG 법률 AI 연구 · 팀 프로젝트 · 2025  
> GitHub: [RYUSEOKHO1050/wood-judge-apps](https://github.com/RYUSEOKHO1050/wood-judge-apps)

---

## 1. 문제 정의

### 연구 목적

자연어로 입력된 법적 상황을 실제 판례 데이터 기반의 RAG 파이프라인으로 검색하여,
일반인이 이해할 수 있는 구조화된 법률 조언을 제공하는 AI 서비스.

| 항목 | 내용 |
|------|------|
| 데이터 | 대한민국 법원 판례 전문 (MySQL + ChromaDB) |
| 임베딩 모델 | `nlpai-lab/KURE-v1` (한국어 특화 SentenceTransformer) |
| LLM 엔진 | `gpt-4o-mini` (Temperature 0.5) |
| 출력 구조 | 사실관계 / 판단 / 예상결과 / 대응전략 / 우려사항 / 관련판례 |
| 백엔드 | FastAPI (Python) |
| 프론트엔드 | React + TypeScript + TailwindCSS |

### 왜 이 문제인가?

법률 서비스의 접근 장벽은 두 가지다.

- **비용·심리적 장벽**: 로펌 방문은 일반인에게 높은 비용과 심리적 부담
- **기존 디지털 수단의 한계**:

| 기존 수단 | 한계 |
|-----------|------|
| GPT 기반 법률 상담 | 웹 문서 기반 → 판례 근거 없는 조언, 법적 신뢰성 부족 |
| 판례 검색 서비스 (bigcase.ai 등) | 키워드 검색만 가능, 자연어 처리 불가, 행동 추천 없음 |

**WOODJUDGE의 차별점**: 자연어 입력 → 실제 판례 벡터 검색 → 판례 번호와 인용문을 명시한 구조화 답변.

### 왜 어려운 문제인가?

1. **한국어 법률 도메인 검색** — 일반 임베딩 모델은 법률 용어의 의미적 유사성을 잘 포착하지 못함
2. **문단 vs 전문 분리** — 검색은 청크(문단) 단위, 답변 생성은 판례 전문이 필요한 이중 조회 구조
3. **다양성 있는 판례 검색** — 유사도만으로 검색하면 동일 판례가 반복 검색됨, 다양한 판례를 커버해야 함
4. **구조화 출력 일관성** — 6개 항목을 매번 일정한 형식으로 출력하도록 프롬프트 설계 필요

---

## 2. 담당 역할 (My Role)

| 담당 | 내용 |
|------|------|
| **나 (석호)** | RAG 파이프라인 설계·구현 (`utils/util.py`) — 검색 전략, 스트리밍, 대화 기능 전체 |
| 팀원  | DB 설계·구축 (`utils/createDB.py`) — 판례 CSV 전처리, 청크 분할, ChromaDB upsert |
|   | 프롬프트 엔지니어링 (`prompts/answer_synth.j2`, `conversation.j2`) — 6항목 구조화 출력 설계 |
|   | 백엔드 (`api_server.py`) — FastAPI 엔드포인트 구축, SSE 스트리밍 서버 |
|  | 프론트엔드 (`src/`) — React + TypeScript + TailwindCSS UI 구현 |

**Tech Stack**

| 분류 | 기술 |
|------|------|
| LLM 엔진 | `gpt-4o-mini` (Temperature 0.5) |
| 임베딩 | `nlpai-lab/KURE-v1` (SentenceTransformer, CUDA/CPU 자동 선택) |
| Vector DB | ChromaDB (`PersistentClient`, collection=`LAW_RAG_500_75`) |
| 관계형 DB | MySQL — 판례 전문(판례일련번호, 판례내용) 저장 |
| 검색 전략 | MMR (Maximal Marginal Relevance) + MultiQueryRetriever |
| 프롬프트 | Jinja2 템플릿 (`answer_synth.j2`, `conversation.j2`) |
| 백엔드 | FastAPI + Server-Sent Events (SSE 스트리밍) |
| 프론트엔드 | React + TypeScript + TailwindCSS |
| 대화 메모리 | LangChain `ConversationBufferMemory` |

---

## 3. 시스템 구조도 (Architecture Diagram)

```
사용자 자연어 입력
        │
        ▼
┌──────────────────────────────────────────────────────┐
│                  FastAPI (api_server.py)              │
│  /api/ask — 초기 질의                                  │
│  /api/conversation — 후속 대화 (ConversationChain)     │
│  /api/case — 판례 전문 조회                             │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│               RAG 파이프라인 (utils/util.py)            │
│                                                      │
│  1) MultiQueryRetriever                              │
│     GPT-4o-mini → 5개 다각도 검색 쿼리 생성              │
│                                                      │
│  2) ChromaDB 벡터 검색                                 │
│     KURE-v1 임베딩 + MMR (k=5, fetch_k=20, λ=0.85)   │
│     → 유사 판례 문단(chunk) 검색                         │
│                                                      │
│  3) MySQL 판례 전문 조회                                │
│     청크의 판례일련번호 → SELECT 판례내용 FROM 판례         │
│                                                      │
│  4) Jinja2 프롬프트 렌더링 (answer_synth.j2)           │
│     CONTEXT(청크) + FULL(전문) + QUESTION → 프롬프트     │
│                                                      │
│  5) GPT-4o-mini 스트리밍 생성 (astream)                │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
       SSE 스트리밍 → React 프론트엔드
                       │
                       ▼
     ┌─────────────────────────────────┐
     │       구조화된 법률 조언          │
     │  사실관계 / 판단 / 예상결과       │
     │  대응전략 / 우려사항 / 관련판례   │
     └─────────────────────────────────┘
```

---

## 4. 핵심 설계 결정

### ① 한국어 특화 임베딩: KURE-v1

일반 임베딩(OpenAI Embeddings, MiniLM 등)은 한국어 법률 문서의 의미적 유사도를 제대로 포착하지 못한다.
고려대 NLP 연구실에서 한국어 검색 태스크에 특화된 `nlpai-lab/KURE-v1`을 채택.
CUDA 가용 여부를 자동 감지하여 GPU/CPU 선택.

```python
# utils/util.py
device = "cuda" if torch.cuda.is_available() else "cpu"
vectorstore = Chroma(
    persist_directory=base_db_dir,
    embedding_function=SentenceTransformerEmbeddings(
        model_name='nlpai-lab/KURE-v1',
        model_kwargs={"device": device}
    ),
    collection_name='LAW_RAG_500_75'
)
```

### ② 이중 DB 구조 (ChromaDB + MySQL)

판례 전문은 수만 자에 달해 벡터 청크와 함께 저장하면 메모리 비효율적.
검색 인덱스(벡터 청크)와 전문 저장(MySQL)을 분리하여, 청크 → 판례번호 → 전문 순서로 조회.

```
ChromaDB (벡터 청크, chunk_size=500, overlap=75)
    └─ metadata: {source: 판례일련번호, case_type: 사건명}
          │
          ▼ 판례일련번호로 조인
MySQL (판례 전문)
    └─ SELECT 판례내용 FROM 판례 WHERE 판례일련번호 = {source}
```

### ③ MMR 검색 (다양성 확보)

단순 유사도(cosine) Top-k는 동일 판례의 유사한 청크가 반복 검색될 수 있음.
MMR(Maximal Marginal Relevance)로 **유사도와 다양성을 동시에 고려**하여 다양한 판례를 커버.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.85}
    # lambda_mult: 1.0 = 순수 유사도, 0.0 = 최대 다양성
)
```

### ④ MultiQueryRetriever (질의 다각화)

사용자의 구어체 법적 상황 설명은 판례 문서의 법률 용어와 표현이 다름.
GPT-4o-mini로 **5개의 다각도 법률 검색 쿼리**를 자동 생성하여 검색 커버리지 확대.

```python
# 예: "편의점에서 알바하다가 손님한테 맞았어요" →
# 1) 편의점 근로자 폭행 피해 손해배상 판례
# 2) 사용자 배상책임 피고용인 폭행 사건
# 3) 점포 내 고객 폭력 형사 처벌 판례 ...
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever, llm=llm, include_original=True
)
```

### ⑤ 구조화 프롬프트 + Jinja2 템플릿

6개 항목(사실관계, 판단, 예상결과, 대응전략, 우려사항, 관련판례)을 일관되게 출력하도록
역할 지시 + 출력 규칙 + 2개 예시(few-shot) + 컨텍스트를 담은 Jinja2 템플릿(`answer_synth.j2`) 설계.
판례 인용 시 `"..."[판례번호:ID]` 형식을 강제하여 신뢰성 확보.

### ⑥ SSE 스트리밍 (UX)

법률 조언 답변은 길이가 길어 생성 완료까지 사용자가 오래 기다림.
FastAPI + LangChain `astream()`으로 청크 단위 SSE 스트리밍 구현, 실시간 타이핑 효과 제공.

```python
# api_server.py
async def generate_stream():
    async for chunk in u.run_rag_stream(...):
        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
return StreamingResponse(generate_stream(), media_type="text/event-stream")
```

### ⑦ 대화 이어가기 (ConversationBufferMemory)

초기 답변 이후 "합의금은 얼마나 받을 수 있나요?"처럼 맥락을 이어가는 추가 질문을 지원.
LangChain `ConversationChain` + `ConversationBufferMemory`로 대화 히스토리를 유지하며 `/api/conversation` 엔드포인트에서 처리.

---

## 5. 구현 세부사항 (코드 기준)

내가 담당한 `utils/util.py` 기준.

| 함수 | 핵심 구현 |
|------|----------|
| `setup_db()` | Chroma 벡터스토어 초기화 — KURE-v1 임베딩 로드, CUDA/CPU 자동 감지 |
| `retrieve_db()` | MMR 검색 (`fetch_k=20, lambda_mult=0.85`) → 판례번호 추출 → MySQL `get_document()` 조회 |
| `multiquery_retrieve_db()` | MultiQueryRetriever + GPT-4o-mini로 5개 다각도 쿼리 생성 → 검색 커버리지 확대 |
| `docs2tpl()` | 검색 결과 k개를 Jinja2 `context{i}` / `full{i}` 변수로 동적 매핑 (k가 결과보다 많으면 빈 문자열로 패딩) |
| `run_rag_stream()` | 비동기 스트리밍 RAG — `retrieve_db()` → `docs2tpl()` → `llm.astream()` → 청크 `yield` |
| `run_conversation()` | ConversationChain에 MMR 재검색 결과(k=3)를 컨텍스트로 주입 + 스트리밍 |
| `set_conversation()` | ConversationBufferMemory 초기화 + 초기 질의·답변을 히스토리에 선적재 |
| `get_llm()` | `ChatOpenAI(model="gpt-4o-mini", temperature=0.5)` 팩토리 — 모든 호출 지점에서 재사용 |

---

## 6. 기술적 도전 & 해결

### 단순 유사도 검색의 판례 중복 문제

초기에 cosine 유사도 Top-k로 검색하면 동일 판례의 비슷한 청크가 k개를 차지하는 경우가 빈번했다.
5개 슬롯을 하나의 판례가 독점하면 다양한 법리를 커버하는 조언을 생성할 수 없었다.

MMR(`lambda_mult=0.85`)을 도입해 유사도와 다양성을 동시에 고려하도록 변경.
`fetch_k=20` — 후보 20개를 먼저 수집한 뒤 중복을 억제하며 최종 5개 선택.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.85}
)
```

### 구어체 입력과 법률 문서 표현의 간극

"친구한테 돈 빌려줬는데 안 갚아요" 같은 구어체 쿼리와 판례 문서의 법률 용어 사이 임베딩 유사도가 낮아 검색 품질이 불안정했다.

GPT-4o-mini 기반 `MultiQueryRetriever`를 도입해 단일 구어체 입력에서 5개의 법률 표현 쿼리를 자동 생성.
`include_original=True`로 원본 쿼리도 포함해 검색 커버리지를 최대화.

```python
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=ChatOpenAI(model="gpt-4o-mini"),
    include_original=True
)
# "친구한테 돈 못 받았어요" →
# 1) 금전 대여 채무불이행 손해배상 판례
# 2) 개인 간 금전거래 변제 이행 청구
# 3) 차용금 반환 소송 시효 판례 ...
```

### 청크 검색과 판례 전문 조회의 분리

벡터 검색 단위(청크)와 LLM에 주입할 컨텍스트(판례 전문)가 달라 이중 조회가 필요했다.
청크 메타데이터의 `source`(판례일련번호)로 MySQL을 조회하는 `get_document()`를 설계하고,
`retrieve_db()` 내 루프에서 청크 검색과 전문 조회를 하나의 흐름으로 처리.

```python
for i, doc in enumerate(results):
    source = doc.metadata['source']           # 청크에서 판례번호 추출
    full = get_document(conn, source)         # MySQL 전문 조회
    output.append({
        "유사문단": doc.page_content,
        "전문": full['판례내용']
    })
```

### 모호한 질문에서의 재질문 루프

"맞았어요", "억울해요" 같은 짧은 입력은 판례를 특정할 맥락이 없어 엉뚱한 판례가 검색됐다.
팀원이 설계한 프롬프트에 "조금 더 자세하게 설명해주세요" 조건을 명시하고,
`run_rag_stream()`의 응답을 RAG 파이프라인 단에서 체크하여 재질문 루프를 제어.

```python
# main.py — RAG 파이프라인이 내려보낸 신호를 루프 조건으로 사용
if '조금 더 자세하게 설명해주세요' not in answer \
        and '관련 판례가 존재하지 않습니다' not in answer:
    break
```

### 대화 이어가기: 초기 답변을 히스토리에 선적재

후속 질문("합의금은 얼마나 받을 수 있나요?") 시 ConversationChain의 메모리가 비어 있으면
앞서 생성된 법률 조언 맥락이 사라져 단절된 답변이 나왔다.
초기 질의·답변 쌍을 `memory.save_context()`로 선적재하여 첫 후속 질문부터 맥락이 유지되도록 설계.

```python
def set_conversation(query, answer, model):
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=model, memory=memory)
    memory.save_context({'input': query}, {'output': answer})  # 초기 맥락 선적재
    return conversation
```

---

## 7. 한계 및 개선 방향

### 한계

1. **정량 평가 부재** — 검색 품질과 출력 품질을 수치로 검증하지 않았음. 기능 동작 여부(`test_rag.py`)는 확인했으나, Retrieval 정밀도나 출력 형식 준수율 같은 자동화된 평가 파이프라인이 없음. 서비스 데모 수준의 수동 검토에 그쳤음.
2. **ConversationBufferMemory 무제한 성장** — 대화가 길어지면 컨텍스트 토큰이 무한정 증가. 슬라이딩 윈도우 또는 요약 메모리가 없음
3. **MMR λ 고정값** — `lambda_mult=0.85`는 경험적으로 설정된 값. 사건 유형별 최적값을 데이터 기반으로 탐색하지 않음
4. **청크 크기 단일 설정** — `chunk_size=500, overlap=75` 하나만 사용. 짧은 판례와 긴 판례 모두에 동일 분할 적용
5. **판례 최신성** — 데이터셋 수집 시점 이후 선고된 판례는 반영 불가

### 개선 방향

- **정량 평가 파이프라인 구축** — 측정 가능한 지표를 도입하여 검색·생성 품질을 체계적으로 검증

  | 평가 지표 | 측정 방법 |
  |-----------|-----------|
  | Retrieval 정확도 | 검색된 판례번호가 실제 DB에 존재하는 비율 |
  | 출력 형식 준수율 | 6개 항목(사실관계~관련판례)이 응답에 모두 포함되는 비율 |
  | 검색 다양성 | MMR vs 단순 유사도 간 Intra-list Diversity 비교 |
  | 판례 인용 적절성 | 인용된 판례와 사용자 상황의 사건 유형 일치 여부 |

- **Cross-Encoder Reranker** — `BAAI/bge-reranker-v2-m3` 기반 재순위로 상위 k개 정확도 개선 (코드에 주석으로 구현 준비됨)
- **HyDE (Hypothetical Document Embeddings)** — 쿼리 대신 가상 판례 문단을 생성하여 임베딩, 법률 도메인 검색 정밀도 향상
- **사건 유형별 라우팅** — 형사/민사/가사 유형을 분류하여 유형별 프롬프트와 판례 컬렉션 분리
- **변호사 연계 플랫폼** — AI 조언 → 변호사 상담 연결로 서비스 완결성 확보

---

*GitHub: [RYUSEOKHO1050/wood-judge-apps](https://github.com/RYUSEOKHO1050/wood-judge-apps)*
