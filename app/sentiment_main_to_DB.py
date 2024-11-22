import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from scipy.stats import loguniform
import math
import pickle
import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db

from ai.SentimentModel import SentimentModel
import logging

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/finance_sentiment_corpus/main/finance_data.csv", filename="finance_data.csv")
df_data = pd.read_csv('finance_data.csv', keep_default_na=False)

# 레이블 데이터 정수 인코딩
df_data['labels'] = df_data['labels'].replace(['neutral', 'positive', 'negative'], [0,1,2])

# 영어 기사 제거
del df_data['sentence']

# 중복 데이터 확인
duplicate = df_data[df_data.duplicated()]
duplicate

# 중복 데이터 제거
df_data.drop_duplicates(subset=['kor_sentence'], inplace=True)
print('총 샘플의 수 :', len(df_data))

# 훈련데이터 레이블링 데이터 분리
x_data = df_data['kor_sentence'].values
y_data = df_data['labels'].values

# 훈련, 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0, stratify=y_data)


# 한국어 형태소 분석기
okt = Okt()

# TF-IDF = 단어들 마다 중요한 정보를 가중치로 계산하는 방법
# TfidfVectorizer는 기본적으로 공백 기준으로 구분함.
# okt.morphs 메서드를 전달해서 형태소 분석기로 토큰화 수행
tfidf = TfidfVectorizer(ngram_range=(1, 2),     # 유니그램과 바이그램 사용
                        min_df=3,               # 3회 미만으로 등장하는 토큰 무시
                        max_df=0.9,             # 상위 10% 토큰 무시
                        tokenizer=okt.morphs,   # 토큰화를 위한 사용자 정의 함수 전달
                        token_pattern=None)


# fit = 훈련, 학습
tfidf.fit(x_train)
x_train_okt = tfidf.transform(x_train)
x_test_okt = tfidf.transform(x_test)


# 데이터 분류기 (확률적 경사 하강법)
sgd = SGDClassifier(loss='log_loss', random_state=1)
param_dist = {'alpha' : loguniform(0.0001, 100.0)}

# 분류기 결정하고 최적의 하이퍼 파라미터를 찾기(오차 값이 가장 적은 것)
rsv_okt = RandomizedSearchCV(estimator=sgd,
                            param_distributions=param_dist, # 파라미터 입력
                            n_iter=50,                      # Random Search 탐색 횟수
                            random_state=1,
                            verbose=1)                      # 진행 상황

# 모델 학습
rsv_okt.fit(x_train_okt, y_train)

print(rsv_okt.score(x_test_okt, y_test))


# 모델 직렬화
bdata = pickle.dumps(rsv_okt, protocol=5)

logger = logging.getLogger("Sentiment_Logger")

db_session = get_db()

def insertSentiment(data: bytes, db: Session):
    try:
        sentiment_model = SentimentModel(content="testDescription", model=data)
        db.add(sentiment_model)
        db.commit()
        db.refresh(sentiment_model)
        return sentiment_model
    except Exception as e:
        logger.error(str(e))
        return None


def selectSentiment(db: Session):
    try:
        model = db.query(SentimentModel).first()
        return model
    except Exception as e:
        logger.error(e)
        return None

db_session_instance = next(db_session)

insertSentiment(bdata, db_session_instance)

model = selectSentiment(db_session_instance)

reverse = pickle.loads(model.model)

# 모델 역직렬화


neutral1 = "Gran에 따르면, 그 회사는 회사가 성장하고 있는 곳이지만, 모든 생산을 러시아로 옮길 계획이 없다고 한다."
negative1 = "국제 전자산업 회사인 엘코텍은 탈린 공장에서 수십 명의 직원을 해고했으며, 이전의 해고와는 달리 회사는 사무직 직원 수를 줄였다고 일간 포스티메스가 보도했다."
positive1 = "새로운 생산공장으로 인해 회사는 예상되는 수요 증가를 충족시킬 수 있는 능력을 증가시키고 원자재 사용을 개선하여 생산 수익성을 높일 것이다."

new_data = [positive1]

prediction_data = tfidf.transform(new_data)

prediction = reverse.predict(prediction_data)
label = {0: 'neutral', 1: 'positive', 2: 'negative'}

print(prediction)

print(label[prediction[0]])