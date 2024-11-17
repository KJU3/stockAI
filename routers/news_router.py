from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Union, List
from sqlalchemy.orm import Session
from src.database import get_db
from src.models import News as ORMNews
from datetime import datetime
from fastapi.encoders import jsonable_encoder

news = APIRouter(prefix='/news')

# Pydantic 모델 (데이터 검증용)
class NewsDto(BaseModel):
    title: str
    content: str
    publishedDate: datetime
    mediaCompany: str
    sentimentIndex: Union[float, None] = None

class MessageResponse(BaseModel):
    message: str

# 유틸리티 함수: 뉴스 조회
def get_news_or_404(newsId: int, db: Session):
    news = db.query(ORMNews).filter(ORMNews.id == newsId).first()
    if not news:
        raise HTTPException(status_code=404, detail="News not found")
    return news

# 뉴스 등록
@news.post("/", tags=['news'])
def upload_news(news_data: NewsDto, db: Session = Depends(get_db)):
    new_news = ORMNews(**news_data.dict())
    db.add(new_news)
    db.commit()
    db.refresh(new_news)
    return jsonable_encoder(new_news)

# 뉴스 수정
@news.put("/{newsId}", tags=['news'])
def update_news(newsId: int, news_data: NewsDto, db: Session = Depends(get_db)):
    news_to_update = get_news_or_404(newsId, db)
    for key, value in news_data.dict().items():
        setattr(news_to_update, key, value)
    db.commit()
    db.refresh(news_to_update)
    return jsonable_encoder(news_to_update)

# 뉴스 삭제
@news.delete("/{newsId}", tags=['news'], response_model=MessageResponse)
def delete_news(newsId: int, db: Session = Depends(get_db)):
    news_to_delete = get_news_or_404(newsId, db)
    db.delete(news_to_delete)
    db.commit()
    return {"message": "News deleted successfully"}

# 뉴스 상세
@news.get("/{newsId}", tags=['news'])
def get_news(newsId: int, db: Session = Depends(get_db)):
    news = get_news_or_404(newsId, db)
    return jsonable_encoder(news)

# 뉴스 목록
@news.get("/", tags=['news'])
def get_news_list(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    return db.query(ORMNews).offset(skip).limit(limit).all()
