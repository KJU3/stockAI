from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os


# 전처리 서비스

# 전처리 분류기 선택
# 분류기 주입
# 분류기 검증

# 로그 생성

# 전처리 클래스
class PreProcessing:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls.tfidf = cls.training()
            
        return cls._instance

    def training():
        # 모델 역직렬화
        x_train = pickle.load(open(os.path.join('app/data/',
                                            'x_train.pkl'), 'rb'))

        # 훈련, 테스트 데이터 분리
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

        tfidf.fit(x_train)

        return tfidf


# 파일 이름, 직렬화 대상 데이터 받아서 저장하는 함수
def to_pickle(path, name, file):
    pickle.dump(file,
            open(os.path.join(path, name), 'wb'),
            protocol=4)
    return