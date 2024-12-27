import uvicorn
# from sentiment.router import router
from sentiment.router import router
from fastapi import FastAPI
from ai.sentiment_news import sentiment_news_model_v4
from starlette.middleware.base import BaseHTTPMiddleware

class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        print(f"Request: {request.method} {request.url}")
        response = await call_next(request)
        response.headers["X-Custom-Header"] = "Custom value"
        return response


SWAGGER_HEADERS = {
    "title": "AI 종목추천 서비스",
    "version": "v1.0.0",
    "description": "## AI 종목 추천 서비스 입니다. \n - 뉴스 감성분석 결과. \n - 키워드 기반 종목 추천 \n - 원하는 AI 모델 교체",
    "contact": {
        "name": "J",
        "url": "https://sysmetic.kr",
        "email": "kju21351@gmail.com",
        "license_info": {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    },
}

app = FastAPI(
    swagger_ui_parameters={
        "deepLinking": True,
        "displayRequestDuration": True,
        "docExpansion": "none",
        "operationsSorter": "method",
        "filter": True,
        "tagsSorter": "alpha",
        "syntaxHighlight.theme": "tomorrow-night",
    },
    **SWAGGER_HEADERS,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(CustomMiddleware)
app.include_router(router)

def main():
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8090,
        reload=True
    )

if __name__ == "__main__":
    main()



# 키워드 추출 -> 1. 키워드 2. 분석 결과 3. 종목
# if __name__ == "__main__":
#     model = sentiment_news_model_v4()
#     model.training_model_template_method()