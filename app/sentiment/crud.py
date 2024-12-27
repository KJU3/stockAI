from sqlalchemy.orm import Session
from sentiment.models import *
from sentiment.exception import NotFoundException
from sentiment.constants import ErrorMessage
from typing import List

# News CRUD

# 뉴스 저장
def save_news(news: News, session: Session):
    session.add(news)
    session.commit()
    session.refresh(news)

# 뉴스 삭제
def delete_news(news: News, session: Session):
    session.delete(news)
    session.commit()

# 뉴스 상세
def find_news_by_news_id(news_id, session: Session):
    news = session.query(News).filter(News.news_id == news_id).first()
    stock_list = session.query(Stock)\
                        .join(NewsStock)\
                            .filter(Stock.stock_id == NewsStock.stock_id)\
                        .join(News)\
                            .filter(NewsStock.news_id == news_id).all()
    prediction = session.query(NewsSentimentPrediction).filter(NewsSentimentPrediction.news_id == news_id).first()
    return {"news": {**news.asdict()}, "stock_list": stock_list, "prediction":prediction.prediction}

# 뉴스 목록
def find_news_list(skip: int, limit: int, session: Session):
    news_list = session.query(News).offset(skip).limit(limit).all()
    if not news_list:
        raise NotFoundException(ErrorMessage.NOT_FOUND)
    return news_list

# Stock CRUD

# 전체 종목 목록 조회

def get_stock_list(session: Session):
    stock_list = session.query(Stock).order_by(Stock.stock_id.desc()).all()
    return stock_list

# NewsStock CRUD

# 뉴스 종목 저장
def save_news_stocks(news_id, stock_list: list[Stock], session: Session):
    for stock in stock_list:
        new_relation = NewsStock(news_id=news_id, stock_id=stock.stock_id)
        session.add(new_relation)
    session.commit()

# 뉴스 아이디로 연관 종목 삭제
def delete_by_news_id(news_id, session: Session):
    session.query(NewsStock).filter(NewsStock.news_id == news_id).delete()

# TrainedModel CRUD

# 모델 고유번호로 모델 찾기
def find_model_by_model_id(model_id, session: Session):
    model = session.query(TrainedModel).filter(TrainedModel.model_id == model_id).first()
    return model

# 모델명으로 가장 최근 버전 모델 찾기
def find_latest_model_by_name(model_name, session: Session):
    model = session.query(TrainedModel).filter(TrainedModel.name == model_name).first()
    return model

# 전체 모델 조회
def get_model_list(session: Session) -> List[TrainedModel]:
    model_list = session.query(TrainedModel).all()
    return model_list

# NewsSentimentPredictioin CRUD

# 예측 결과 저장
def save_prediction(prediction: NewsSentimentPrediction, session: Session):
    session.add(prediction)
    session.commit()
    session.refresh(prediction)