import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from scipy.stats import loguniform
import pickle
import os
import joblib

# 데이터 다운로드 및 로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/finance_sentiment_corpus/main/finance_data.csv", filename="finance_data.csv")
df_data = pd.read_csv('finance_data.csv', keep_default_na=False)

# 레이블 데이터 정수 인코딩
df_data['labels'] = df_data['labels'].replace(['neutral', 'positive', 'negative'], [0, 1, 2])

# 영어 기사 제거
del df_data['sentence']

# 중복 데이터 제거
df_data.drop_duplicates(subset=['kor_sentence'], inplace=True)
print('총 샘플의 수 :', len(df_data))

# 훈련 데이터와 레이블 분리
x_data = df_data['kor_sentence'].values
y_data = df_data['labels'].values

# 훈련, 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0, stratify=y_data)

# 한국어 형태소 분석기
oct = Okt()

# 데이터 토큰화 (형태소 분석)
tokenized_train = [" ".join(oct.morphs(sent)) for sent in x_train]
tokenized_test = [" ".join(oct.morphs(sent)) for sent in x_test]

# TF-IDF 벡터라이저 초기화 및 학습
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)
tfidf.fit(tokenized_train)
x_train_okt = tfidf.transform(tokenized_train)
x_test_okt = tfidf.transform(tokenized_test)

# 벡터라이저 직렬화
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# 데이터 분류기 (확률적 경사 하강법)
sgd = SGDClassifier(loss='log_loss', random_state=1)
param_dist = {'alpha': loguniform(0.0001, 100.0)}

# RandomizedSearchCV를 사용하여 최적의 하이퍼파라미터 찾기
rsv_okt = RandomizedSearchCV(estimator=sgd,
                             param_distributions=param_dist,  # 파라미터 입력
                             n_iter=50,                       # Random Search 탐색 횟수
                             random_state=1,
                             verbose=1)                       # 진행 상황 출력

# 모델 학습
rsv_okt.fit(x_train_okt, y_train)

# 학습된 모델 점수 출력
print(rsv_okt.score(x_test_okt, y_test))

# 모델 직렬화 및 저장
cur_dir = os.path.dirname(__file__)
dest = os.path.join(cur_dir, 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(rsv_okt, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)

# 저장된 모델 불러오기
clf = pickle.load(open(os.path.join(dest, 'classifier.pkl'), 'rb'))
print(f'복원한 클래스 점수: {clf.score(x_test_okt, y_test)}')

# 새로운 데이터 예측
positive1 = "새로운 생산공장으로 인해 회사는 예상되는 수요 증가를 충족시킬 수 있는 능력을 증가시키고 원자재 사용을 개선하여 생산 수익성을 높일 것이다."
new_data = [" ".join(oct.morphs(positive1))]

# 저장된 TF-IDF 벡터라이저 불러오기
loaded_tfidf = joblib.load('tfidf_vectorizer.pkl')

# 새로운 데이터 변환 및 예측
prediction_data = loaded_tfidf.transform(new_data)
prediction = clf.predict(prediction_data)
label = {0: 'neutral', 1: 'positive', 2: 'negative'}

print(prediction)
print(label[prediction[0]])
