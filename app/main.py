import uvicorn
from news.main_router import api_router
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(api_router)

def main():
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8090,
        reload=True
    )

if __name__ == "__main__":
    main()
