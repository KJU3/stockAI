from sqlalchemy import Column, BigInteger, Text, DateTime, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class News(Base):
    __tablename__ = "news"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    publishedDate = Column(DateTime, nullable=False)
    mediaCompany = Column(String(255), nullable=False)  # VARCHAR의 길이 지정
    sentimentIndex = Column(Float, nullable=True)
