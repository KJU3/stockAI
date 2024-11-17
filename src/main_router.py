from fastapi import FastAPI
from routers.news_router import news

app = FastAPI()

app.include_router(news)