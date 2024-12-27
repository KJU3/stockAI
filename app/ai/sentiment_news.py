import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform
from scipy.stats import loguniform
import pickle
import os
from sentiment.models import TrainedModel
from util.model_to_db import insert_trained_model
from ai.training_model_template import training_model_template_class
from sklearn.metrics import classification_report
from soynlp.tokenizer import LTokenizer

class sentiment_news_model_v1(training_model_template_class):
    # 데이터 불러오기
    def load_data(self):
        urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/finance_sentiment_corpus/main/finance_data.csv", filename="app/data/finance_data.csv")
        df_data = pd.read_csv('app/data/finance_data.csv', keep_default_na=False)
        
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

        return df_data
    
    def split_x_y_data(self, data):
        # 훈련, 레이블링 데이터 분리
        x_data = data['kor_sentence'].values
        y_data = data['labels'].values
        return (x_data, y_data)
    
    def split_train_and_test_data(self, data):
        x_data, y_data = data

        # 훈련, 테스트 데이터 분리
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0, stratify=y_data)
        return (x_train, x_test, y_train, y_test)

    def preprocess_data(self, data):
        x_train, x_test, y_train, y_test = data
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

        return (tfidf, x_train_okt, x_test_okt, x_train, y_train, y_test)
    
    def trainig_model(self, data):
        tfidf, x_train_okt, x_test_okt, x_train, y_train, y_test = data

        # 데이터 분류기 (확률적 경사 하강법:SGD)
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

        score = rsv_okt.score(x_test_okt, y_test)

        # 예측값 생성
        y_pred = rsv_okt.predict(x_test_okt)

        # 정밀도, 재현율, F1 점수 출력
        print("Classification Report:")
        report = classification_report(y_test, y_pred)
        print(report)

        return (rsv_okt, f"score : {score}, classification_report : {report}")

    def save_preprocessing_data(self, data):
        tfidf, x_train_okt, x_test_okt, x_train, y_train, y_test = data
        # 전처리 데이터 저장
        pickle.dump(x_train,
            open(os.path.join('app/data', 'x_train.pkl'), 'wb'),
            protocol=4)

    def save_model(self, model_and_score):
        model, score = model_and_score

        version = 'v1'
        name = '감성분석모델'
        description = """
        감성분석모델 v1 전처리, 학습 이력
        
        # step 전처리
        1. 레이블 데이터 정수 인코딩
        2. 한글 기사만 추출
        3. 중복 데이터 제거
        4. 훈련/테스트 데이터 분리 비율 : 8/2
        5. 토큰화 : 형태소 분석기로 토큰화. Okt의 morphs 
        6. 벡터화 : TF-IDF 방식 사용(3회 미만, 상위 10% 토큰 무시)

        # step 학습
        1. 선형분류 (SGDClassifier) 알고리즘 사용
        2. 손실함수 log_loss (Logistic regression) 사용
        3. 교차검증 (Randomized Search Cross Validation) 사용
        """
        # 모델 직렬화
        model = pickle.dumps(model, protocol=4)
        result = f'Test Data Score : {score}'

        trainde_model = TrainedModel(
            version=version,
            name=name,
            description=description,
            model=model,
            training_result=result
        )

        insert_trained_model(trainde_model)



class sentiment_news_model_v2(sentiment_news_model_v1):
    def trainig_model(self, data):
        tfidf, x_train_okt, x_test_okt, x_train, y_train, y_test = data

        # 데이터 분류기 (확률적 경사 하강법:SGD)
        sgd = SGDClassifier(loss='hinge', random_state=1)
        param_dist = {'alpha' : loguniform(0.0001, 100.0)}

        # 분류기 결정하고 최적의 하이퍼 파라미터를 찾기(오차 값이 가장 적은 것)
        rsv_okt = RandomizedSearchCV(estimator=sgd,
                                    param_distributions=param_dist, # 파라미터 입력
                                    n_iter=50,                      # Random Search 탐색 횟수
                                    random_state=1,
                                    verbose=1)                      # 진행 상황

        # 모델 학습
        rsv_okt.fit(x_train_okt, y_train)

        score = rsv_okt.score(x_test_okt, y_test)

        # 예측값 생성
        y_pred = rsv_okt.predict(x_test_okt)

        # 정밀도, 재현율, F1 점수 출력
        print("Classification Report:")
        report = classification_report(y_test, y_pred)
        print(report)

        return (rsv_okt, f"score : {score}, classification_report : {report}")
    
    def save_model(self, model_and_score):
        model, score = model_and_score

        version = 'v2'
        name = '감성분석모델_hinge'
        description = """
        감성분석모델 v2 전처리, 학습 이력
        
        # step 전처리
        1. 레이블 데이터 정수 인코딩
        2. 한글 기사만 추출
        3. 중복 데이터 제거
        4. 훈련/테스트 데이터 분리 비율 : 8/2
        5. 토큰화 : 형태소 분석기로 토큰화. Okt의 morphs 
        6. 벡터화 : TF-IDF 방식 사용(3회 미만, 상위 10% 토큰 무시)

        # step 학습
        1. 선형분류 (SGDClassifier) 알고리즘 사용
        2. 손실함수 hinge (SVM: Suport Vector Machine) 사용
        3. 교차검증 (Randomized Search Cross Validation) 사용
        """
        # 모델 직렬화
        model = pickle.dumps(model, protocol=4)
        result = f'Test Data Score : {score}'

        trainde_model = TrainedModel(
            version=version,
            name=name,
            description=description,
            model=model,
            training_result=result
        )

        insert_trained_model(trainde_model)



class sentiment_news_model_v3(sentiment_news_model_v2):

    def preprocess_data(self, data):
        x_train, x_test, y_train, y_test = data
        # 한국어 형태소 분석기
        lto = LTokenizer()

        # TF-IDF = 단어들 마다 중요한 정보를 가중치로 계산하는 방법
        # TfidfVectorizer는 기본적으로 공백 기준으로 구분함.
        # okt.morphs 메서드를 전달해서 형태소 분석기로 토큰화 수행
        tfidf = TfidfVectorizer(ngram_range=(1, 2),     # 유니그램과 바이그램 사용
                                min_df=3,               # 3회 미만으로 등장하는 토큰 무시
                                max_df=0.9,             # 상위 10% 토큰 무시
                                tokenizer=lto.tokenize,   # 토큰화를 위한 사용자 정의 함수 전달
                                token_pattern=None)
        
        # fit = 훈련, 학습
        tfidf.fit(x_train)
        x_train_okt = tfidf.transform(x_train)
        x_test_okt = tfidf.transform(x_test)

        return (tfidf, x_train_okt, x_test_okt, x_train, y_train, y_test)
    
    def save_model(self, model_and_score):
        model, score = model_and_score

        version = 'v3'
        name = '감성분석모델_hinge'
        description = """
        감성분석모델 v3 전처리, 학습 이력
        
        # step 전처리
        1. 레이블 데이터 정수 인코딩
        2. 한글 기사만 추출
        3. 중복 데이터 제거
        4. 훈련/테스트 데이터 분리 비율 : 8/2
        5. 토큰화 : 형태소 분석기로 토큰화. soynlp LTokenizer
        6. 벡터화 : TF-IDF 방식 사용(3회 미만, 상위 10% 토큰 무시)


        # step 학습
        1. 선형분류 (SGDClassifier) 알고리즘 사용
        2. 손실함수 hinge(SVM: Suport Vector Machine) 사용
        3. 교차검증 (Randomized Search Cross Validation) 사용
        """

        # 모델 직렬화
        model = pickle.dumps(model, protocol=4)
        result = f'Test Data Score : {score}'

        trainde_model = TrainedModel(
            version=version,
            name=name,
            description=description,
            model=model,
            training_result=result
        )

        insert_trained_model(trainde_model)

class sentiment_news_model_v4(sentiment_news_model_v1):
    def trainig_model(self, data):
        tfidf, x_train_okt, x_test_okt, x_train, y_train, y_test = data

        # 랜덤 포레스트 분류기
        rfc = RandomForestClassifier(random_state=2, class_weight='balanced')

        # 하이퍼파라미터 분포 설정
        dists = {
            'n_estimators': randint(50, 300),  # 트리의 개수
            'max_depth': [5, 10, 15, 20],  # 트리 최대 깊이
            'max_features': uniform(0, 1)  # 사용할 최대 피처 비율
        }

        # RandomizedSearchCV로 최적의 하이퍼파라미터 탐색
        rsv_okt = RandomizedSearchCV(estimator=rfc,
                                    param_distributions=dists,
                                    n_iter=30,  # 랜덤 탐색 횟수
                                    cv=3,  # 교차 검증 폴드 수
                                    scoring='accuracy',  # 평가 지표
                                    verbose=1,
                                    n_jobs=-1,
                                    error_score='raise')

        # 모델 학습
        rsv_okt.fit(x_train_okt, y_train)

        score = rsv_okt.score(x_test_okt, y_test)

        # 예측값 생성
        y_pred = rsv_okt.predict(x_test_okt)

        # 정밀도, 재현율, F1 점수 출력
        print("Classification Report:")
        report = classification_report(y_test, y_pred)
        print(report)

        return (rsv_okt, f"score : {score}, classification_report : {report}")
    
    def save_model(self, model_and_score):
        model, score = model_and_score

        version = 'v4'
        name = '감성분석모델_RandomForest'
        description = """
        감성분석모델 v4 전처리, 학습 이력
        
        # step 전처리
        1. 레이블 데이터 정수 인코딩
        2. 한글 기사만 추출
        3. 중복 데이터 제거
        4. 훈련/테스트 데이터 분리 비율 : 8/2
        5. 토큰화 : 형태소 분석기로 토큰화. Okt의 morphs 
        6. 벡터화 : TF-IDF 방식 사용(3회 미만, 상위 10% 토큰 무시)

        # step 학습
        1. 랜덤포레스트 (RandomForestClassifier) 알고리즘 사용
        2. 교차검증 (Randomized Search Cross Validation) 사용
        """
        # 모델 직렬화
        model = pickle.dumps(model, protocol=4)
        result = f'Test Data Score : {score}'

        trainde_model = TrainedModel(
            version=version,
            name=name,
            description=description,
            model=model,
            training_result=result
        )

        insert_trained_model(trainde_model)



