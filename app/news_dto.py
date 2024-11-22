from pydantic import BaseModel, Field
from datetime import datetime
from typing import Union, Generic, TypeVar

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    code: int
    message: Union[str, None] = None
    data: Union[T, None] = None

class NewsResponse(BaseModel):
    newsId: int
    title: str
    content: str
    publishedDate: datetime
    mediaCompany: str
    sentimentIndex: Union[float, None] = None

    class Config:
        from_attributes = True  # Pydantic v2에서 ORM 변환 활성화

class NewsCreateDto(BaseModel):
    title: str
    content: str
    publishedDate: datetime
    mediaCompany: str

class NewsUpdateDto(BaseModel):
    title: Union[str, None] = None
    content: Union[str, None] = None
    publishedDate: Union[datetime, None] = None
    mediaCompany: Union[str, None] = None
