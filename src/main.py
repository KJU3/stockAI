from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, List
from enum import Enum
import uvicorn

app = FastAPI()

class News(BaseModel): 
    newsId: int
    title: str
    content: str
    publishedDate: str
    mediaCompany: str
    sentimentIndex: Union[float, None] = None

# class MediaCompany(str, Enum):
#     한국경제 = "한국경제"


newsTable: List[News]

def insertNews(news: News):
    newsTable.append(news)
    return newsTable

def updateNews(news: News):
    return news

def deleteNews(newsId: int):
    return

def sentimentService(news: News):
    return 0.0

def findNews(newsId: int):
    
    return News()

def findNewsList():
    return newsTable

# 뉴스 등록
@app.post("/news")
def uploadNews(news: News):
    insertNews(news)
    return {"message": "News uploaded successfully",
            "newsId": 123}

# 뉴스 수정
@app.put("/news/{newsId}")
def updateNews(news: News):
    return updateNews(news)

# 뉴스 삭제
@app.delete("/news/{newsId}")
def deleteNews(newsId: int):
    return deleteNews(newsId)

# 뉴스 감성 확인
@app.get("/sentimentIndex")
def getSentimentIndex(news: News):
    return sentimentService(news)

# 뉴스 상세
@app.get("/news/{newsId}")
def getNews(newsId: int):
    return findNews(newsId)

# 뉴스 목록
@app.get("/news")
def getNewsList():
    return findNewsList()