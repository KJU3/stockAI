from sqlalchemy import Column, BigInteger, Text, DateTime, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class News(Base):
    __tablename__ = "news"
    
    newsId = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    publishedDate = Column(DateTime, nullable=False)
    mediaCompany = Column(String(255), nullable=False)
    sentimentIndex = Column(Float, nullable=True)