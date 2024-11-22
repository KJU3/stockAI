from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from news_models import News
from news_dto import NewsCreateDto, NewsUpdateDto, NewsResponse, ApiResponse
import logging

logger = logging.getLogger("news_router")

router = APIRouter()

# 유틸리티 함수: 뉴스 조회
def get_news_or_404(newsId: int, db: Session):
    news = db.query(News).filter(News.newsId == newsId).first()
    if not news:
        raise HTTPException(status_code=404, detail=f"News not found : {newsId}")
    return news

# 뉴스 등록
@router.post("/", response_model=ApiResponse[NewsResponse])
def upload_news(news_data: NewsCreateDto, db: Session = Depends(get_db)):
    new_news = News(**news_data.dict())
    try:
        db.add(new_news)
        db.commit()
        db.refresh(new_news)
        return ApiResponse(code=200, data=NewsResponse.from_orm(new_news), message=f"Success Upload News : {new_news.newsId}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to upload news: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload news")

# 뉴스 수정
@router.put("/{newsId}", response_model=ApiResponse[NewsResponse])
def update_news(newsId: int, news_data: NewsUpdateDto, db: Session = Depends(get_db)):
    news_to_update = get_news_or_404(newsId, db)
    try:
        for key, value in news_data.dict(exclude_unset=True).items():
            setattr(news_to_update, key, value)
        db.commit()
        db.refresh(news_to_update)
        return ApiResponse(code=200, data=NewsResponse.from_orm(news_to_update))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update news")

# 뉴스 삭제
@router.delete("/{newsId}", response_model=ApiResponse)
def delete_news(newsId: int, db: Session = Depends(get_db)):
    news_to_delete = get_news_or_404(newsId, db)
    try:
        db.delete(news_to_delete)
        db.commit()
        return ApiResponse(code=200, data=newsId, message=f"Seccess Delete : {newsId}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete news")

# 뉴스 상세
@router.get("/{newsId}", response_model=ApiResponse[NewsResponse])
def get_news(newsId: int, db: Session = Depends(get_db)):
    news = db.query(News).filter(News.newsId == newsId).first()
    
    if not news:
        raise HTTPException(status_code=404, detail=f"News not found : {newsId}")
    
    return ApiResponse(code=200, data=NewsResponse.from_orm(news))

# 뉴스 목록
@router.get("/", response_model=list[NewsResponse])
def get_news_list(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    if skip < 0 or limit <= 0:
        raise HTTPException(status_code=400, detail="Invalid pagination parameters")
    return db.query(News).offset(skip).limit(limit).all()

# 뉴스 감성분석 결과
@router.post("/sentiment")
def post_sentiment_source(content: str):
    return content


# 감성 평가
@router.post("/sentiment/{newsId}")
def post_sentiment_result(score: int):
    return True
