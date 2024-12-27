
# Python FastAPI와 관련된 명령어 및 개념 정리

---

## 주요 명령어

### 1. **FastAPI 실행**
FastAPI 앱을 실행하는 두 가지 방법:
- **명령어로 실행**:
  ```bash
  uvicorn src.main:app --host 0.0.0.0 --port 8090 --reload
  ```
- **Python 코드로 실행**:
  ```python
  if __name__ == "__main__":
      uvicorn.run(app, host="0.0.0.0", port=8090, reload=True)
  ```

### 2. **의존성 관리**
- 현재 환경의 의존성을 파일로 저장:
  ```bash
  pip freeze > requirements.txt
  ```
- 저장된 의존성을 설치:
  ```bash
  pip install -r 파일이름.txt
  ```

---

## 용어 정리

### 1. **CGI (Common Gateway Interface)**
- **정의**: 웹 서버와 외부 애플리케이션 간의 동적 요청 처리를 위한 공통 규약(인터페이스).
- **특징**: 요청마다 새로운 프로세스를 생성하는 방식으로 작동.

### 2. **WSGI (Web Server Gateway Interface)**
- **정의**: CGI의 단점을 개선하기 위해 개발된 인터페이스.
- **특징**:
  - 요청마다 프로세스를 생성하지 않고, **callable object**를 사용해 요청을 처리.
  - Python 웹 서버와 애플리케이션 간의 표준 인터페이스.

### 3. **ASGI (Asynchronous Server Gateway Interface)**
- **정의**: WSGI의 단점을 보완하여 비동기 처리를 지원하는 인터페이스.
- **특징**:
  - FastAPI와 같은 비동기 기반 프레임워크에서 사용.
  - **Uvicorn**은 ASGI를 지원하는 서버.

---

## FastAPI에서 경로 변수 처리

### 경로 변수 정의
FastAPI는 경로 변수로 문자열, 정수, 경로 등을 사용할 수 있습니다. 예를 들어:
```python
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}
```

- **`{file_path:path}`**:
  - `path`는 슬래시(`/`)가 포함된 경로까지 처리 가능.
  - 예: `/files/my/folder/file.txt`와 같은 요청을 처리.

---


```

├── app  # Contains the main application files.
│   ├── __init__.py   # this file makes "app" a "Python package"
│   ├── main.py       # Initializes the FastAPI application.
│   ├── dependencies.py # Defines dependencies used by the routers
│   ├── routers
│   │   ├── __init__.py
│   │   ├── news.py  # Defines routes and endpoints related to items.
│   │   └── users.py  # Defines routes and endpoints related to users.
│   ├── crud
│   │   ├── __init__.py
│   │   ├── news.py  # Defines CRUD operations for items.
│   │   └── user.py  # Defines CRUD operations for users.
│   ├── schemas
│   │   ├── __init__.py
│   │   ├── news.py  # Defines schemas for items.
│   │   └── user.py  # Defines schemas for users.
│   ├── models
│   │   ├── __init__.py
│   │   ├── news.py  # Defines database models for items.
│   │   └── user.py  # Defines database models for users.
├── tests
│   ├── __init__.py
│   ├── test_main.py
│   ├── test_items.py  # Tests for the items module.
│   └── test_users.py  # Tests for the users module.
├── requirements.txt
├── .gitignore
└── README.md

```


```
fastapi-project
├── alembic/
├── src
│   ├── auth
│   │   ├── router.py         # auth main router with all the endpoints
│   │   ├── schemas.py        # pydantic models
│   │   ├── models.py         # database models
│   │   ├── dependencies.py   # router dependencies
│   │   ├── config.py         # local configs
│   │   ├── constants.py      # module-specific constants
│   │   ├── exceptions.py     # module-specific errors
│   │   ├── service.py        # module-specific business logic
│   │   └── utils.py          # any other non-business logic functions
│   ├── aws
│   │   ├── client.py  # client model for external service communication
│   │   ├── schemas.py
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── exceptions.py
│   │   └── utils.py
│   └── posts
│   │   ├── router.py
│   │   ├── schemas.py
│   │   ├── models.py
│   │   ├── dependencies.py
│   │   ├── constants.py
│   │   ├── exceptions.py
│   │   ├── service.py
│   │   └── utils.py
│   ├── config.py      # global configs
│   ├── models.py      # global database models
│   ├── exceptions.py  # global exceptions
│   ├── pagination.py  # global module e.g. pagination
│   ├── database.py    # db connection related stuff
│   └── main.py
├── tests/
│   ├── auth
│   ├── aws
│   └── posts
├── templates/
│   └── index.html
├── requirements
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── .env
├── .gitignore
├── logging.ini
└── alembic.ini
```


# NLP 모델과 데이터 셋 크기별 전략

자연어 처리(NLP)에서 사용할 모델과 기법은 데이터 셋의 크기에 따라 크게 달라질 수 있습니다. 데이터 크기에 따른 적합한 기술과 접근 방법을 아래와 같이 정리했습니다.

---

## 데이터 셋 크기와 권장 접근 방식

### 1. **작은 데이터 셋 (5,000 ~ 10,000 문장)**
- **적합한 기법:**
  - **BoW** (Bag of Words) 또는 **TF-IDF**와 같은 간단한 피처 변환 방식
  - **로지스틱 회귀(Logistic Regression)** 또는 **SVM(Support Vector Machine)**과 같은 단순 모델
- **데이터 증강:**  
  - Synonym Replacement (동의어 치환)
  - Back Translation (역번역)
- **사전 학습된 모델 활용:**
  - Word2Vec, GloVe와 같은 사전 학습된 임베딩
  - BERT, GPT와 같은 사전 학습된 언어 모델을 Fine-tuning

---

### 2. **중간 크기 데이터 셋 (50,000 ~ 100,000 문장)**
- **적합한 기법:**
  - **LSTM** (Long Short-Term Memory)
  - 랜덤 포레스트(Random Forest)와 같은 더 복잡한 모델

---

### 3. **대규모 데이터 셋 (100,000 ~ 수백만 문장)**
- **적합한 기법:**
  - Word Embedding을 직접 학습
  - Transformer 기반 모델 (예: BERT, GPT 등)
  - 사전 학습 없이 Transformer 모델 학습

---

## 데이터 크기에 따른 모델 요구사항

| **기법/모델**                  | **요구 데이터 크기**         |
|---------------------------------|-----------------------------|
| BoW, TF-IDF                    | 5,000 ~ 10,000 문장         |
| Word Embedding (직접 학습)      | 수십만 문장 이상            |
| Word Embedding (사전 학습)      | 5,000 ~ 10,000 문장         |
| 딥러닝 모델                    | 50,000 ~ 100,000 문장       |
| Transformer Fine-tuning        | 10,000 ~ 50,000 문장        |
| Transformer 학습 (사전 학습 없이)| 수백만 문장                 |

---

## 작은 데이터 셋에 적합한 전략

1. **간단한 모델 사용**
   - BoW, TF-IDF와 같은 간단한 피처 변환 방식
   - Logistic Regression, SVM 같은 비교적 가벼운 모델

2. **데이터 증강**
   - Synonym Replacement: 동의어로 단어를 대체
   - Back Translation: 문장을 다른 언어로 번역 후 다시 원문으로 역번역하여 새로운 데이터 생성

3. **사전 학습된 모델 활용**
   - Word2Vec, GloVe, BERT, GPT 등 사전 학습된 임베딩이나 언어 모델을 활용하여 Fine-tuning 수행

4. **Transfer Learning**
   - 대규모 데이터셋에서 학습된 모델을 작은 데이터셋에 맞게 Fine-tuning하여 사용

---

이 글은 데이터 크기에 따라 어떤 접근법이 적합한지 간단히 정리하며, 데이터 증강 및 사전 학습된 모델 활용의 중요성을 강조합니다. 모델 및 데이터 셋 전략을 선택할 때 참고하시기 바랍니다.

---

---

# 🌟 Section Divider 🌟

---

# NLP(Natural Language Processing) 자연어 처리

자연어 처리(NLP)는 인간의 언어를 컴퓨터가 이해하고 처리할 수 있도록 만드는 기술입니다. 그 첫 단계는 텍스트 데이터를 정제하고 구조화하는 **전처리 과정**으로 시작됩니다. 이번 글에서는 텍스트 전처리의 주요 기법들을 살펴보겠습니다.

---

## 텍스트 전처리 종류

### 1. **텍스트 정제 (Text Cleaning)**
텍스트 데이터를 정리하여 분석하기 적합한 형태로 만드는 과정입니다. 주요 작업으로는 다음과 같은 것들이 포함됩니다.
- 숫자, 구두점(`.` 및 `,` 등) 제거
- 불필요한 공백 제거 (예: double 공백 → single 공백)
- 소문자화 (모든 텍스트를 소문자로 변환)

### 2. **불용어 제거 (Stop Words Removal)**
불용어는 텍스트에서 자주 등장하지만, 분석에 중요한 정보를 제공하지 않는 단어를 의미합니다. 예를 들어, 영어에서는 "the", "is", "in" 등이 불용어로 간주됩니다. 이를 제거함으로써 텍스트 데이터의 품질을 향상시킬 수 있습니다.

### 3. **표제어 추출 (Lemmatization)**
표제어 추출은 단어를 사전의 기본 형태(base form)로 변환하는 작업입니다. 예를 들어:
- "studying", "studies", "studied" → "study"

표제어 추출은 문맥 정보를 고려하며, 단어가 문장에서 사용된 **품사(POS: Part of Speech)**에 따라 처리됩니다. 이는 의미 있는 어휘를 유지하며 데이터를 정제하는 데 적합합니다.

### 4. **어간 추출 (Stemming)**
어간 추출은 단어의 어근(root form)을 찾는 작업으로, 단순히 접사를 제거하는 방식으로 이루어집니다. 예를 들어:
- "studying", "studies", "studied" → "studi"

어간 추출은 표제어 추출에 비해 문맥을 고려하지 않으며, 다소 기계적인 방식으로 진행됩니다.

---

### 어간 추출 vs. 표제어 추출
두 방법 모두 단어의 변형된 형태를 단순화하려는 목적을 가집니다. 하지만 그 사용 목적과 특성이 다릅니다.
- **어간 추출 (Stemming)**: 빠르고 간단하며, 문맥을 고려하지 않아 Sentiment Analysis(감정 분석) 같은 작업에 적합합니다.
- **표제어 추출 (Lemmatization)**: 문맥 정보를 반영하며, Chatbot(챗봇) 개발처럼 높은 정확도가 요구되는 작업에 적합합니다.

표제어 추출은 더 복잡한 형태소 분석이 필요하며, 이를 위해 품사 태깅(POS tagging)이 중요한 역할을 합니다.

---

## 토큰화 (Tokenizing)

토큰화는 텍스트를 작은 단위(단어, 문장, 하위 단어 등)로 나누는 과정입니다. 예를 들어:
- "I love NLP!" → ["I", "love", "NLP"]

영어의 경우, 대표적인 토큰화 도구로 **NLTK (Natural Language Toolkit)**가 많이 사용됩니다.

---

## 주요 어간 추출 알고리즘

1. **Porter Stemmer**  
   어간 추출을 위한 가장 널리 알려진 알고리즘으로, 단순하고 효율적인 방식이 특징입니다.

2. **Lancaster Stemmer**  
   Porter Stemmer보다 더 공격적인 방식으로 어근을 추출합니다. 하지만 과도한 추출로 인해 단어가 지나치게 변형될 가능성이 있습니다.

---

## 표제어 추출과 품사 태깅

표제어 추출은 단어의 문맥에 따라 적절한 기본 형태를 찾아내는 데 중점을 둡니다. 이 과정에서 **품사 태깅(POS tagging)**이 중요한 역할을 합니다. 품사 태깅은 단어가 문장에서 사용된 품사를 식별하며, 이는 표제어 추출의 정확도를 높이는 데 필수적입니다.

---

### 마무리

자연어 처리는 데이터의 품질에 크게 의존합니다. 전처리 과정에서 텍스트를 정제하고 적합한 기법을 적용함으로써, NLP 모델의 성능을 극대화할 수 있습니다. 어간 추출과 표제어 추출의 차이를 이해하고 적재적소에 활용하는 것이 성공적인 NLP 프로젝트의 열쇠입니다.
