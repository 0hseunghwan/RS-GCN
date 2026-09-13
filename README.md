# RS-GCN

Graph Convolutional Network(GCN)으로 학습한 논문 임베딩을 이용해 유사 논문을 추천하는 프로젝트입니다. [Cora](https://paperswithcode.com/dataset/cora) 인용 네트워크 데이터셋을 사용해 GCN을 학습하고, 학습된 임베딩 사이의 코사인 유사도를 기반으로 논문 추천 API(FastAPI)를 제공합니다.

## 구성 요소

- `models/gcn.py` — GCN 모델 정의
  - `GCN`: 노드 분류용 2-layer GCN (`log_softmax` 출력)
  - `GCNEncoder`: 추천 시스템용 임베딩 인코더 (분류 헤드 없이 임베딩만 출력)
- `train.py` — Cora 데이터셋으로 `GCN` 노드 분류 모델을 학습하고 정확도를 평가하는 스크립트
- `train_RS.py` — `GCNEncoder`로 논문 임베딩을 학습하고, 추천 시스템에 필요한 데이터(`data/embeddings.npy`, `data/papers.json`, `data/edges.json`)를 생성하는 스크립트
- `RS.py` — 학습된 임베딩을 로드해 키워드/카테고리 기반 유사 논문 검색을 수행하는 `PaperRecommender` 클래스
- `app.py` — 추천 시스템을 서빙하는 FastAPI 애플리케이션
- `requirements.txt` — 의존성 패키지 목록

## 설치

```bash
pip install -r requirements.txt
```

> `torch`, `torch-geometric`과 관련 확장(`torch_scatter`, `torch_sparse`)은 사용 중인 PyTorch/CUDA 버전에 맞는 빌드가 필요할 수 있습니다. 환경에 따라 [PyG 공식 설치 가이드](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)를 참고해 별도로 설치하세요.

## 사용 방법

### 1. GCN 노드 분류 학습 (선택)

Cora 데이터셋으로 기본 GCN 분류 성능을 확인합니다.

```bash
python train.py
```

### 2. 추천 시스템용 임베딩 생성 (필수)

`GCNEncoder`를 학습시켜 논문 임베딩과 메타데이터를 `data/` 디렉터리에 생성합니다. 추천 API를 실행하기 전에 반드시 먼저 실행해야 합니다.

```bash
python train_RS.py
```

실행이 완료되면 다음 파일이 생성됩니다.

- `data/embeddings.npy` — 논문(노드) GCN 임베딩
- `data/papers.json` — 논문 메타데이터 (id, title, label, category)
- `data/edges.json` — 인용 관계(엣지) 목록

### 3. 추천 API 서버 실행

```bash
python app.py
```

서버는 기본적으로 `http://localhost:8000` 에서 실행됩니다.

## API 엔드포인트

| Method | Endpoint | 설명 |
| --- | --- | --- |
| GET | `/` | 메인 페이지 (`static/index.html`) |
| GET | `/api/search?q={query}&top_k={n}` | 키워드/카테고리로 유사 논문 검색 |
| GET | `/api/paper/{paper_id}` | 특정 논문 상세 정보 및 유사 논문 목록 |
| GET | `/api/categories` | 전체 카테고리 목록 및 논문 수 |
| GET | `/api/graph?limit={n}` | 시각화용 노드/엣지 그래프 데이터 |

예시:

```bash
curl "http://localhost:8000/api/search?q=neural+networks&top_k=5"
```

## 데이터셋

[Planetoid Cora](https://paperswithcode.com/dataset/cora) 데이터셋을 사용하며, 7개 카테고리로 분류됩니다.

- Case Based
- Genetic Algorithms
- Neural Networks
- Probabilistic Methods
- Reinforce Learning
- Rule Learning
- Theory

Cora 데이터셋에는 실제 논문 제목이 포함되어 있지 않아, `train_RS.py`에서 카테고리 기반으로 가상의 논문 제목을 생성해 사용합니다.
