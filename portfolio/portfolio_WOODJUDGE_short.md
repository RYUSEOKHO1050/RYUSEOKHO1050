# WOODJUDGE — 유사 판례 기반 법률 조언 AI 서비스

> RAG 법률 AI 연구 · 팀 프로젝트 · 2025  
> GitHub: [RYUSEOKHO1050/wood-judge-apps](https://github.com/RYUSEOKHO1050/wood-judge-apps)

---

## 한 줄 요약

자연어로 법적 상황을 입력하면 **실제 판례 전문 기반 RAG**로 유사 판례를 검색하고,
사실관계·예상결과·대응전략을 구조화하여 제공하는 AI 법률 조언 서비스.

---

## 담당 역할

| 담당 | 내용 |
|------|------|
| **나 (석호)** | RAG 파이프라인 설계·구현 (`utils/util.py`) — 검색 전략, 스트리밍, 대화 기능 전체 |
| 팀원 A | DB 설계·구축 (`utils/createDB.py`) — 판례 CSV 전처리, 청크 분할, ChromaDB upsert |
| 팀원 B | 프롬프트 엔지니어링 (`prompts/`) — 6항목 구조화 출력, few-shot 설계 |
| 팀원 C | 백엔드 (`api_server.py`) — FastAPI 엔드포인트, SSE 스트리밍 서버 |
| 팀원 D | 프론트엔드 (`src/`) — React + TypeScript + TailwindCSS UI |

---

## 기술 스택

| 분류 | 내용 |
|------|------|
| LLM | `gpt-4o-mini` (Temperature 0.5) |
| 임베딩 | `nlpai-lab/KURE-v1` (한국어 특화 SentenceTransformer) |
| Vector DB | ChromaDB (MMR 검색, k=5, fetch_k=20) |
| 관계형 DB | MySQL — 판례 전문 저장 |
| 프롬프트 | Jinja2 템플릿 + few-shot 2개 |
| 백엔드 | FastAPI + SSE 스트리밍 |
| 프론트엔드 | React + TypeScript + TailwindCSS |
| 대화 | LangChain ConversationBufferMemory |

---

## 핵심 설계 결정

### ① 이중 DB 구조 (ChromaDB + MySQL)
판례 전문은 수만 자로 길어 벡터와 함께 저장 시 비효율적.
청크(500자/75 overlap) 기반 벡터 검색 → 판례번호 추출 → MySQL 전문 조회 순서로 분리.

### ② MMR 검색 (다양성 확보)
단순 유사도 Top-k는 동일 판례의 중복 청크가 반복 검색됨.
`lambda_mult=0.85`로 유사도와 다양성을 동시에 최적화, 다양한 판례 커버.

### ③ MultiQueryRetriever (질의 다각화)
구어체 상황 설명 → GPT-4o-mini로 5개 법률 검색 쿼리 자동 생성 → 검색 커버리지 확대.

### ④ SSE 스트리밍
긴 법률 답변의 UX 개선. FastAPI + `astream()`으로 청크 단위 실시간 스트리밍.

### ⑤ 구조화 출력 (6항목)
Jinja2 프롬프트로 사실관계 / 판단 / 예상결과 / 대응전략 / 우려사항 / 관련판례 일관 출력.
모든 판례 인용에 `[판례번호:ID]` 명시 강제 → 법적 근거 투명성 확보.

---

## 시스템 구조

```
자연어 입력
    ↓
FastAPI (/api/ask)
    ↓
MultiQueryRetriever → 5개 쿼리 생성
    ↓
ChromaDB MMR 검색 → 유사 판례 청크 5개
    ↓
MySQL → 판례 전문 조회
    ↓
Jinja2 프롬프트 렌더링
    ↓
GPT-4o-mini astream → SSE 스트리밍
    ↓
[사실관계 / 판단 / 예상결과 / 대응전략 / 우려사항 / 관련판례]
```

---

## 한계 및 개선 방향

- **정량 평가 부재** — 기능 동작 확인 수준에 그쳤고, Retrieval 정밀도·출력 형식 준수율 같은 자동화 평가 파이프라인이 없음. 향후 아래 지표 기반 평가 체계 구축이 필요.

  | 평가 지표 | 내용 |
  |-----------|------|
  | Retrieval 정확도 | 검색 판례번호가 실제 DB에 존재하는 비율 |
  | 출력 형식 준수율 | 6개 항목이 응답에 모두 포함되는 비율 |
  | 검색 다양성 | MMR vs 단순 유사도 Intra-list Diversity 비교 |

- **ConversationBufferMemory** 무제한 성장 → 슬라이딩 윈도우 또는 요약 메모리 도입 필요
- **Cross-Encoder Reranker** (`BAAI/bge-reranker-v2-m3`) 적용 시 검색 정밀도 추가 향상 가능 (코드에 주석으로 구현 준비됨)
- 사건 유형별(형사/민사/가사) 라우팅으로 맞춤 프롬프트 분리
