# Portfolio — Ryu Seokho

> NLP · LLM · RAG 중심 프로젝트 3선

---

## 01. Cyberpunk 2077 구매 의사 시뮬레이션
`Multi-Agent RAG` · `팀 프로젝트` · `2025` · [GitHub →](https://github.com/RYUSEOKHO1050/NLP-Persona)

LLM 기반 104명의 페르소나 에이전트가 시계열 Steam 리뷰에 노출될 때,  
대중의 구매 의사 변화를 얼마나 정확히 시뮬레이션할 수 있는지 검증.

| | |
|:--|:--|
| **내 기여** | Static RAG + Time-Aware RAG 설계·구현 — Time-Decay 가중 검색(지수 감쇠), Candidate Expansion(300개 후보 → 재순위) 도입 |
| **핵심 기술** | Python · ChromaDB · GPT-4o-mini · text-embedding-3-small · OpenAI API |
| **결과** | Pearson 상관계수 0.356 → **0.578** (+0.222) / 주가와 음의 상관(-0.578)으로 에이전트 무결성 확인 |

---

## 02. WOODJUDGE — 유사 판례 기반 법률 조언 AI
`RAG Service` · `팀 프로젝트` · `2025` · [GitHub →](https://github.com/RYUSEOKHO1050/wood-judge-apps)

자연어로 법적 상황을 입력하면 실제 판례 전문 기반 RAG로 유사 판례를 검색하고,  
사실관계 · 예상결과 · 대응전략을 구조화하여 제공.

| | |
|:--|:--|
| **내 기여** | RAG 파이프라인 전체 설계·구현 — MMR 검색(다양성 확보), MultiQueryRetriever(구어체→법률 쿼리 변환), 비동기 스트리밍, 대화 맥락 유지 |
| **핵심 기술** | Python · ChromaDB · MySQL · KURE-v1(한국어 특화 임베딩) · LangChain · FastAPI · GPT-4o-mini |
| **결과** | 서비스 데모 구현 완료 / 정량 평가 미비가 한계 — Retrieval 정확도·출력 형식 준수율 기반 평가 파이프라인 구축이 향후 과제 |

---

## 03. 한국어 일상 대화 연결
`NLP Classification` · `팀 프로젝트` · `2025` · [GitHub →](https://github.com/RYUSEOKHO1050/NLP-malpyeong)

600개 소규모 구어체 데이터에서 대화 맥락에 맞는 발화를 3지선다로 선택하는 NLP 과제.  
*(AI 말평 대회)*

| | |
|:--|:--|
| **내 기여** | 모델 아키텍처 전체 설계·구현 — Pair-wise 이진 분류 재정의, 감정·부정 특수 토큰 주입, 선행·후행 모델 분리, 양방향 앙상블 |
| **핵심 기술** | Python · PyTorch · HuggingFace Transformers · klue/roberta-large · Stratified 5-Fold CV |
| **결과** | 리더보드 75.73 → **87.45 (+11.72%p)** / 역번역 증강보다 원본 데이터 품질 보존이 유효함을 실험으로 확인 |
