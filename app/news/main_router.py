from fastapi import APIRouter
from news.routers import news_router

api_router = APIRouter()
api_router.include_router(news_router.router, prefix="/news", tags=["news"])
