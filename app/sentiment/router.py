from fastapi import APIRouter, Depends, HTTPException, File, Form, UploadFile, Body
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import get_db
from sentiment.schemas import *
from sentiment.models import *
import logging
import pickle
import io
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sentiment.service import PreProcessing
from sentiment import crud
import threading
from datetime import date, datetime
from collections import Counter, defaultdict
from konlpy.tag import Okt

# 로거 생성
logger = logging.getLogger("router")

# 라우터 생성
router = APIRouter()

# 전처리기 불러오기
tfidf = PreProcessing().tfidf

model_lock = threading.Lock()
current_model = None
model_info = None

# 종목 추출
def extract_stocks(title: str, content:str, stockList: List[Stock]):
    return [stock for stock in stockList 
            if stock.stock_name in content or stock.stock_name in title]

# News
# 뉴스 데이터 JSON n개 저장
@router.post("/news", response_model=NewsResponse, tags=["News"], description="뉴스 데이터를 저장하고 관련된 종목 및 예측 정보를 처리합니다.")
def register_news(newsList: MultipleNewsRequest, db: Session = Depends(get_db)):
    if model_info is None or current_model is None:
        raise HTTPException(status_code=400, detail=f"No model loaded model_info : {model_info}, current_model : {current_model}")
    
    # 뉴스 데이터
    request = newsList.news[0]
    news = News(
        title=request.title,
        content=request.content,
        published_at=request.published_at,
        media=request.media,
        reporter=request.reporter
    )

    # 뉴스 테이블 저장
    crud.save_news(news=news, session=db)

    # 종목 리스트
    stockList = crud.get_stock_list(session=db)

    # 종목 추출
    matched_stocks = extract_stocks(request.title, request.content, stockList)

    # 뉴스종목 교차 테이블 저장
    crud.save_news_stocks(news_id=news.news_id, stock_list=matched_stocks, session=db)

    # 예측 데이터 전처리
    prediction_data = tfidf.transform([request.content])

    prediction = current_model.predict(prediction_data)

    label = {0: 'neutral', 1: 'positive', 2: 'negative'}

    # 예측 결과 저장
    crud.save_prediction(prediction = NewsSentimentPrediction(
                                    news_id = news.news_id,
                                    model_id = model_info.model_id,
                                    prediction = prediction), 
                    session = db)
    
    return {
        "news_id": news.news_id,
        "title": news.title,
        "content": news.content,
        "published_at": news.published_at,
        "media": news.media,
        "reporter": news.reporter,
        "stock_list": [{"stock_id": stock.stock_id, "stock_name": stock.stock_name, "stock_code": stock.stock_code} 
                    for stock in matched_stocks],
        "prediction": f'label: {prediction[0]}, score?: {label[prediction[0]]}'
    }

# News
# 뉴스 목록
@router.get("/news", tags=["News"], description="저장된 모든 뉴스 데이터를 페이징하여 반환합니다.")
def get_news_list(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    if skip < 0 or limit <= 0:
        raise HTTPException(status_code=400, detail="Invalid pagination parameters")
    news_list = crud.find_news_list(skip=skip, limit=limit, session=db)
    return news_list

# News
# 뉴스 상세
@router.get("/news/{news_id}", tags=["News"], description="특정 뉴스 ID에 대한 상세 정보를 반환합니다.")
def get_news(news_id, db: Session = Depends(get_db)):
    news_with_stocks = crud.find_news_by_news_id(news_id=news_id, session=db)
    if not news_with_stocks:
        raise HTTPException(status_code=404, detail=f"News not found : {news_id}")
    return news_with_stocks

# News
# 뉴스 삭제
@router.delete("/news/{news_id}", tags=["News"], description="특정 뉴스 ID를 삭제합니다.")
def remove_news(news_id: int, db: Session = Depends(get_db)):
    news = crud.find_news_by_news_id(news_id=news_id, session=db)
    if not news:
        raise HTTPException(status_code=404, detail=f"News not found : {news_id}")
    try:
        # 교차 테이블 정보 삭제
        crud.delete_by_news_id(news_id=news.news_id, session=db)
        # 뉴스 삭제
        crud.delete_news(news=news, session=db)
        return {"code":200, "data":news_id, "message":f"Success Delete : {news_id}"}
    except Exception as e:
        logger.error(e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete news")

# News
# 뉴스 수정
@router.put("/news/{news_id}", tags=["News"], description="특정 뉴스 ID의 데이터를 수정합니다.")
def update_news(news_id: int, news_data: SingleNewsRequest, db: Session = Depends(get_db)):
    news = crud.find_news_by_news_id(news_id=news_id, session=db)
    if not news:
        raise HTTPException(status_code=404, detail=f"News not found : {news_id}")
    try:
        for key, value in news_data.dict(exclude_unset=True).items():
            setattr(news, key, value)
        db.commit()
        db.refresh(news)

        return {"code":200, "data":{**news.asdict()}, "message":f"Success update : {news_id}"}
    except Exception as e:
        logger.error(e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update news")

# Model
# 모델 목록 반환
@router.get("/model/list", response_model=List[TrainedModelResponse], tags=["Model"], description="현재 훈련된 모델 목록을 반환합니다.")
async def get_model_list(db: Session = Depends(get_db)):
    model_list = crud.get_model_list(session=db)
    if model_list is None:
        raise HTTPException(status_code=400, detail=f"No model list available")
    
    # 데이터베이스 객체를 응답 모델로 변환
    response = [
        TrainedModelResponse(
            id=model.model_id,
            version=model.version,
            name=model.name,
            description=model.description,
            created_at=model.created_at,
            training_result=model.training_result,
        ) for model in model_list
    ]

    # description 및 training_result 포맷 적용
    for model in response:
        if model.description:
            model.description = model.description.strip().replace("\n", "\n> ")
        if model.training_result:
            model.training_result = model.training_result.strip().replace("\n", "\n> ")

    return model_list

# Model
# 모델 상태 반환
@router.get("/model/status", tags=["Model"], description="현재 로드된 모델 상태를 반환합니다.")
async def get_model_status():
    if model_info is None or current_model is None:
        raise HTTPException(status_code=400, detail=f"No model loaded model_info : {model_info}, current_model : {current_model}")
    return {"status": f"Model is loaded id : {model_info.model_id}, name : {model_info.name}, model : {current_model}"}

# Model
# 모델 테스트
@router.get("/model/test", tags=["Model"], description="현재 로드된 모델을 테스트하고 상태를 확인합니다.")
async def get_model_test():
    if model_info is None or current_model is None:
        raise HTTPException(status_code=400, detail=f"No model loaded model_info : {model_info}, current_model : {current_model}")
    
    return {"test": f"Model score : {model_info.description}"}

# Model
# 모델 셋팅 - 모델아이디로 설정
@router.post("/model/load/{model_id}", tags=["Model"], description="주어진 모델 ID로 모델을 동적으로 로드합니다.")
async def load_model(model_id, db: Session = Depends(get_db)):
    with model_lock:
        try:
            model_entity = crud.find_model_by_model_id(model_id=model_id, session=db)
            new_model = pickle.loads(model_entity.model)
            global current_model
            global model_info

            current_model = new_model
            model_info = model_entity
            return {"message": f"Model loaded from id : {model_entity.model_id}, name : {model_entity.name}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# File
# 뉴스 데이터 CSV n개 저장
@router.post("/files/news", tags=["Files"], description="CSV 파일을 업로드하여 뉴스 데이터를 저장합니다.")
async def register_news_csv(
    file: UploadFile = File(),
    db: Session = Depends(get_db)
):
    try:
        file_content = await file.read()
        df = pd.read_csv(io.StringIO(file_content.decode(encoding="utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    df.dropna(axis=0)

    try:
        db.bulk_save_objects([
            News(
                title=row['title'],
                content=row['content'],
                published_at=row['published_at'],
                media=row['media'],
            )
            for _, row in df.iterrows()
        ])
        db.commit()

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving to database: {str(e)}")

# Batch
# 뉴스 데이터 전체 감성 분석 후 집계 테이블 저장
@router.put("/sentiment", tags=["Sentiment"], description="저장된 모든 뉴스 데이터를 대상으로 감성 분석을 수행합니다.")
async def sentiment_news_all(db: Session = Depends(get_db)):
    if model_info is None or current_model is None:
        raise HTTPException(status_code=400, detail=f"No model loaded model_info : {model_info}, current_model : {current_model}")
    
    try:
        news_list = db.query(News).all()

        prediction_list = []

        for news in news_list:
            prediction_data = tfidf.transform([news.content])

            prediction = current_model.predict(prediction_data)

            prediction_list.append(
                NewsSentimentPrediction(
                    news_id=news.news_id,
                    model_id=model_info.model_id,
                    prediction=prediction
                )
            )

        db.bulk_save_objects(prediction_list)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving to database: {str(e)}")

# File
# 종목 일별 데이터 CSV n개 저장
@router.post("/files/daily-stock-history", tags=["Files"], description="CSV 파일을 업로드하여 종목의 일별 데이터를 저장합니다.")
async def register_stock_history_csv(
    file: UploadFile = File(),
    db: Session = Depends(get_db)
):
    try:
        file_content = await file.read()
        df = pd.read_csv(io.StringIO(file_content.decode(encoding="utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    df = df.dropna(axis=0)
    
    try:
        records = df.to_dict(orient='records')
        db.bulk_save_objects([DailyStockPriceHistory(**record) for record in records])
        db.commit()

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving to database: {str(e)}")

# Batch
# 주가 일별 데이터 등락 순위, 상승/하락 Labeling
@router.post("/stock/history/update_ranking_direction", tags=["Stock"], description="일별 등락률을 기반으로 순위 및 UP/DOWN Labeling합니다.")
async def update_ranking_and_direction(db: Session = Depends(get_db)):
    try:
        all_stocks = db.query(DailyStockPriceHistory).all()
        grouped_stocks = defaultdict(list)
        for stock in all_stocks:
            grouped_stocks[stock.stock_date].append(stock)

        update_data = []
        for stock_date, stocks in grouped_stocks.items():
            sorted_up = sorted(
                [s for s in stocks if s.updown_ratio > 0],
                key=lambda x: x.updown_ratio, reverse=True
            )
            for ranking, stock in enumerate(sorted_up, start=1):
                update_data.append({"id": stock.id, "ranking": ranking, "direction": "UP"})

            sorted_down = sorted(
                [s for s in stocks if s.updown_ratio < 0],
                key=lambda x: x.updown_ratio
            )
            for ranking, stock in enumerate(sorted_down, start=1):
                update_data.append({"id": stock.id, "ranking": ranking, "direction": "DOWN"})

        db.bulk_update_mappings(DailyStockPriceHistory, update_data)
        db.commit()

        return {"message": "Ranking과 Direction이 성공적으로 업데이트되었습니다."}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Batch
# 뉴스 빅데이터 일자별 상위 키워드 집계
@router.post("/news/keywords/rank", tags=["Keywords"], description="뉴스 데이터를 분석해 일자별 상위 키워드를 추출하고 저장합니다.")
async def extract_and_save_keyword_rank(db: Session = Depends(get_db), top_n: int = 10):
    try:
        db.query(DailyNewsKeywordRank).delete()
        db.commit()

        all_news = db.query(News).all()
        grouped_news = defaultdict(list)
        for news in all_news:
            grouped_news[news.published_at].append(news.content)

        okt = Okt()
        bulk_insert_data = []

        for published_at, contents in grouped_news.items():
            keyword_counter = Counter()
            for content in contents:
                keywords = okt.nouns(content)
                filtered_keywords = [word for word in keywords if len(word) >= 2]
                keyword_counter.update(filtered_keywords)

            top_keywords = keyword_counter.most_common(top_n)

            for ranking, (keyword, freq) in enumerate(top_keywords, start=1):
                bulk_insert_data.append(DailyNewsKeywordRank(
                    published_at=published_at,
                    keyword=keyword,
                    ranking=ranking,
                    frequency=freq
                ))

        db.bulk_save_objects(bulk_insert_data)
        db.commit()

        return {"message": "일자별 키워드 랭킹이 성공적으로 저장되었습니다."}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Batch
# 일자별 키워드와 종목 연관관계 집계
@router.post("/news/keywords/stock-association", tags=["Keywords"], description="일자별 키워드와 종목 간의 관계를 저장합니다.")
async def save_keyword_stock_association(db: Session = Depends(get_db)):
    try:
        db.query(DailyKeywordStockAssociation).delete()
        db.commit()

        daily_keywords = db.query(DailyNewsKeywordRank).all()
        daily_stock_ranks = db.query(DailyStockPriceHistory).filter(DailyStockPriceHistory.ranking.between(1, 5)).all()

        bulk_insert_data = []
        for daily_keyword in daily_keywords:
            for stock in daily_stock_ranks:
                if daily_keyword.published_at == stock.stock_date:
                    bulk_insert_data.append(DailyKeywordStockAssociation(
                        published_at=daily_keyword.published_at,
                        keyword=daily_keyword.keyword,
                        stock_code=stock.stock_code,
                        stock_name=stock.stock_name
                    ))

        db.bulk_save_objects(bulk_insert_data)
        db.commit()

        return {"message": "뉴스 키워드와 종목 관계 데이터가 저장되었습니다."}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Batch
# 키워드와 종목의 연관 횟수 집계
@router.post("/keywords/stock-summary", tags=["Keywords"], description="키워드와 종목의 연결 횟수를 집계하여 저장합니다.")
async def save_keyword_stock_summary(db: Session = Depends(get_db)):
    try:
        db.query(KeywordStockAssociationScore).delete()
        db.commit()

        results = (
            db.query(
                DailyKeywordStockAssociation.keyword,
                DailyKeywordStockAssociation.stock_code,
                DailyKeywordStockAssociation.stock_name,
                func.count(DailyKeywordStockAssociation.id).label("count")
            )
            .group_by(
                DailyKeywordStockAssociation.keyword,
                DailyKeywordStockAssociation.stock_code,
                DailyKeywordStockAssociation.stock_name
            )
            .all()
        )

        bulk_insert_data = [
            KeywordStockAssociationScore(
                keyword=row.keyword,
                stock_code=row.stock_code,
                stock_name=row.stock_name,
                association_score=row.count
            )
            for row in results
        ]

        db.bulk_save_objects(bulk_insert_data)
        db.commit()

        return {"message": "키워드-종목 집계가 성공적으로 저장되었습니다."}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# 뉴스 감성분석, 키워드 기반 종목 추천
@router.post("/news/sentiment/keywords/recommend", tags=["News"], description="뉴스 감성 분석을 수행하고 키워드 관련 종목을 추천합니다.")
async def analyze_news_and_recommend_stocks(
    news_request: SingleNewsRequest = Body(
        ...,
        example={
            "title": "지난해 기업 투자, 글로벌 시장에서 빛을 발하다",
            "content": """지난해 주요 기업들이 글로벌 시장에서 대규모 투자에 나서며 경제 성장을 견인했다.특히, 첨단 기술 분야에서의 투자가 두드러져 AI와 클라우드 시장이 급성장했다.전문가들은 이러한 투자 활동이 향후 몇 년간 시장의 판도를 바꿀 것으로 전망하고 있다.하지만 일부 기업은 높은 비용과 불확실한 시장 상황 속에서 신중한 접근을 요구받고 있다.
            """,
            "published_at": "2024-12-13",
            "media": "데브캠프 1기",
            "reporter": "아무개 기자",
        }
    ),
    db: Session = Depends(get_db)):
    try:
        news_content = news_request.content

        label = {0: 'neutral', 1: 'positive', 2: 'negative'}
        prediction_data = tfidf.transform([news_content])
        sentiment_score = current_model.predict(prediction_data)

        okt = Okt()
        keyword_counter = Counter()
        keywords = okt.nouns(news_content)
        filtered_keywords = [word for word in keywords if len(word) >= 2]
        keyword_counter.update(filtered_keywords)

        unique_keywords = list(set(filtered_keywords))
        if not unique_keywords:
            raise HTTPException(status_code=400, detail="키워드를 추출하지 못했습니다.")

        all_recommended_stocks = []
        for keyword in unique_keywords:
            top_stocks = (
                db.query(KeywordStockAssociationScore)
                .filter(KeywordStockAssociationScore.keyword == keyword)
                .order_by(KeywordStockAssociationScore.association_score.desc())
                .all()
            )

            for stock in top_stocks:
                all_recommended_stocks.append({
                    "keyword": keyword,
                    "stock_code": stock.stock_code,
                    "stock_name": stock.stock_name,
                    "association_score": stock.association_score
                })

        sorted_stocks = sorted(all_recommended_stocks, key=lambda x: x["association_score"], reverse=True)
        top_5_stocks = sorted_stocks[:5]

        return {
            "sentiment_score": label[sentiment_score[0]],
            "recommended_stocks": top_5_stocks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# 삭제 예정 메서드

# # 특정일자 상승, 하락 Top5 추출
# @router.get("/stock/rank", tags=["Stock"], description="특정 일자에 대한 상승/하락 Top5 종목을 반환합니다.")
# async def get_daily_stock_rank(
#     daily: date,
#     top: int = 5,
#     db: Session = Depends(get_db)
# ):
#     up = (
#         db.query(DailyStockPriceHistory)
#         .filter(DailyStockPriceHistory.stock_date == daily)
#         .order_by(DailyStockPriceHistory.updown_ratio.desc())
#         .limit(top)
#         .all()
#     )
#     down = (
#         db.query(DailyStockPriceHistory)
#         .filter(DailyStockPriceHistory.stock_date == daily)
#         .order_by(DailyStockPriceHistory.updown_ratio.asc())
#         .limit(top)
#         .all()
#     )
#     return {'up': up, 'down': down}

# # Batch
# # 모든 종목, 일자별 상승/하락 순위 계산
# @router.post("/stock/rank/save_all", tags=["Stock"], description="모든 데이터를 일자별 상승/하락 순위로 계산하여 저장합니다.")
# async def save_all_daily_stock_rank(db: Session = Depends(get_db)):
#     try:
#         db.query(DailyStockUpDownRank).delete()
#         db.commit()

#         all_stocks = db.query(DailyStockPriceHistory).all()

#         grouped_stocks = defaultdict(list)
#         for stock in all_stocks:
#             grouped_stocks[stock.stock_date].append(stock)

#         bulk_insert_data = []
#         for stock_date, stocks in grouped_stocks.items():
#             sorted_up = sorted(
#                 [s for s in stocks if s.updown_ratio > 0],
#                 key=lambda x: x.updown_ratio, reverse=True
#             )
#             for ranking, stock in enumerate(sorted_up, start=1):
#                 bulk_insert_data.append(DailyStockUpDownRank(
#                     stock_date=stock.stock_date,
#                     stock_code=stock.stock_code,
#                     stock_name=stock.stock_name,
#                     close_price=stock.close_price,
#                     updown=stock.updown,
#                     updown_ratio=stock.updown_ratio,
#                     ranking=ranking,
#                     direction="UP",
#                     volume=stock.volume
#                 ))

#             sorted_down = sorted(
#                 [s for s in stocks if s.updown_ratio < 0],
#                 key=lambda x: x.updown_ratio
#             )
#             for ranking, stock in enumerate(sorted_down, start=1):
#                 bulk_insert_data.append(DailyStockUpDownRank(
#                     stock_date=stock.stock_date,
#                     stock_code=stock.stock_code,
#                     stock_name=stock.stock_name,
#                     close_price=stock.close_price,
#                     updown=stock.updown,
#                     updown_ratio=stock.updown_ratio,
#                     ranking=ranking,
#                     direction="DOWN",
#                     volume=stock.volume
#                 ))

#         db.bulk_save_objects(bulk_insert_data)
#         db.commit()

#         return {"message": "모든 데이터를 일자별 상승/하락 순위로 계산 후 저장하였습니다."}

#     except Exception as e:
#         db.rollback()
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")