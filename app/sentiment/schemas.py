from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# 단일 뉴스 요청
class SingleNewsRequest(BaseModel):
    title: str = Field(..., example="삼성전자가 신제품 발표")
    content: str = Field(..., example="삼성전자는 오늘 신제품 발표회를 열고...")
    published_at: datetime = Field(..., example="2024-11-23T12:00:00")
    media: str = Field(..., example="매일경제")
    reporter: Optional[str] = Field(None, example="홍길동")

# 여러 뉴스 요청
class MultipleNewsRequest(BaseModel):
    news: List[SingleNewsRequest]

# 종목 데이터 응답
class StockResponse(BaseModel):
    stock_id: int
    stock_name: str
    stock_code: str

# 단일 뉴스 응답
class NewsResponse(BaseModel):
    news_id: int
    title: str
    content: str
    published_at: datetime
    media: str
    reporter: Optional[str]  # 기자는 선택적 필드
    stock_list: List[StockResponse]  # 연결된 종목 목록
    prediction: Optional[str]
    
    class Config:
        from_attributes = True
        populate_by_name = True

# 여러 뉴스 응답
class NewsListResponse(BaseModel):
    news: List[NewsResponse]

# 훈련 모델 응답
class TrainedModelResponse(BaseModel):
    id: int = Field(..., alias='model_id')
    version: Optional[str]
    name: str
    description: Optional[str]
    created_at: datetime
    training_result: Optional[str]

    # 포맷팅된 출력 제공
    def format_description(self):
        if self.description:
            return self.description.strip().replace("\n", "\n> ")

    def format_training_result(self):
        if self.training_result:
            return self.training_result.strip().replace("\n", "\n> ")
    
    class Config:
        from_attributes = True
        populate_by_name = True
