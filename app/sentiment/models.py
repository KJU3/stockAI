from sqlalchemy import (
    Column, Integer, Double, String, Text, DateTime, Date, ForeignKey, UniqueConstraint, BigInteger
)
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime, timezone

Base = declarative_base()

# 종목 테이블
class Stock(Base):
    __tablename__ = "stock"
    
    stock_id = Column(Integer, primary_key=True, autoincrement=True)   # 고유 식별자
    stock_code = Column(String(255), nullable=False, unique=True)  # 종목 코드
    stock_type = Column(String(255), nullable=False)    # 종목 타입
    stock_name = Column(String(255), nullable=False, unique=True)  # 종목 이름
    issued_share = Column(BigInteger, nullable=False)
    industry_name = Column(String(255), nullable=False)
    market_cap = Column(BigInteger, nullable=False)

    # 관계 설정 (NewsStock에서 참조)
    news_associations = relationship("NewsStock", back_populates="stock")


# 뉴스 테이블
class News(Base):
    __tablename__ = "news"
    
    news_id = Column(Integer, primary_key=True, autoincrement=True)  # 고유 식별자
    title = Column(String(255), nullable=False)  # 뉴스 제목
    content = Column(Text, nullable=False)  # 뉴스 본문
    published_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)  # 게시일
    media = Column(String(255), nullable=False)  # 언론사
    reporter = Column(String(255), nullable=True)  # 기자 이름

    # 관계 설정 (NewsStock에서 참조)
    stock_associations = relationship("NewsStock", back_populates="news")

    # 관계 설정 (NewsSentimentPrediction에서 참조)
    model_associations = relationship("NewsSentimentPrediction", back_populates="news")

    def asdict(self):
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}


# 뉴스-종목 관계 테이블
class NewsStock(Base):
    __tablename__ = "news_stock"

    id = Column(Integer, primary_key=True, autoincrement=True)  # 고유 ID
    news_id = Column(Integer, ForeignKey("news.news_id", ondelete="CASCADE"), nullable=False)  # 뉴스 ID (외래키), 뉴스나 종목이 삭제되면 같이 삭제
    stock_id = Column(Integer, ForeignKey("stock.stock_id", ondelete="CASCADE"), nullable=False)  # 종목 ID (외래키)

    # 관계 설정
    news = relationship("News", back_populates="stock_associations")
    stock = relationship("Stock", back_populates="news_associations")

    # 뉴스와 종목의 조합이 유일하도록 제약 조건 추가
    __table_args__ = (UniqueConstraint("news_id", "stock_id", name="_news_stock_uc"),)


class NewsSentimentPrediction(Base):
    __tablename__ = "news_sentiment_prediction"

    id = Column(Integer, primary_key=True, autoincrement=True)  # 고유 ID
    news_id = Column(Integer, ForeignKey("news.news_id", ondelete="CASCADE"), nullable=False)  # 뉴스 ID (외래키), 뉴스나 종목이 삭제되면 같이 삭제
    model_id = Column(Integer, ForeignKey("trained_model.model_id", ondelete="CASCADE"), nullable=False)  # 모델 ID (외래키)
    prediction = Column(String(255))
    prediction_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # 관계 설정
    news = relationship("News", back_populates="model_associations")
    trained_model = relationship("TrainedModel", back_populates="news_associations")


class TrainedModel(Base):
    __tablename__ = "trained_model"

    model_id = Column(Integer, primary_key=True, autoincrement=True)  # 고유 ID
    version = Column(String(255))
    name = Column(String(255))
    description = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    modified_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    model = Column(LONGBLOB)
    training_result = Column(Text)

    # 관계 설정 (NewsSentimentPrediction에서 참조)
    news_associations = relationship("NewsSentimentPrediction", back_populates="trained_model")

class DailyStockPriceHistory(Base):
    __tablename__ = "daily_stock_price_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(255), nullable=False)
    stock_name = Column(String(255))
    stock_date = Column(Date, nullable=False)
    updown = Column(Integer)
    updown_ratio = Column(Double)
    open_updown_ratio = Column(Double)
    open_price = Column(Integer)
    high_price = Column(Integer)
    low_price = Column(Integer)
    close_price = Column(Integer)
    volume = Column(BigInteger)
    day10_price = Column(Integer)
    day20_price = Column(Integer)
    day60_price = Column(Integer)
    day120_price = Column(Integer)
    ranking = Column(Integer)
    direction = Column(String(10))

class DailyStockUpDownRank(Base):
    __tablename__ = "daily_stock_up_down_rank"

    id = Column(Integer, primary_key=True, autoincrement=True)  # 기본 키
    stock_date = Column(Date, nullable=False)  # 거래 날짜
    stock_code = Column(String(255), nullable=False)  # 종목 코드
    stock_name = Column(String(255), nullable=False)  # 종목명
    close_price = Column(Integer, nullable=False)  # 종가
    updown = Column(Integer, nullable=False)  # 전일 대비 상승/하락 금액
    updown_ratio = Column(Double, nullable=False)  # 전일 대비 상승/하락 비율 (%)
    ranking = Column(Integer, nullable=False)  # 상승/하락 순위 (1~5위)
    direction = Column(String(10), nullable=False)  # 'UP' 또는 'DOWN'
    volume = Column(BigInteger, nullable=True)  # 거래량 (선택 사항)

class DailyNewsKeywordRank(Base):
    __tablename__ = "daily_news_keyword_rank"

    id = Column(Integer, primary_key=True, autoincrement=True)  # 기본 키
    published_at = Column(Date, nullable=False)  # 뉴스 날짜
    keyword = Column(String(255), nullable=False)  # 키워드
    ranking = Column(Integer, nullable=False)  # 키워드 순위
    frequency = Column(Integer, nullable=False)  # 키워드 등장 빈도수

class DailyKeywordStockAssociation(Base):
    __tablename__ = "daily_keyword_stock_association"

    id = Column(Integer, primary_key=True, autoincrement=True)  # 기본 키
    published_at = Column(Date, nullable=False)  # 키워드 등장 날짜
    keyword = Column(String(255), nullable=False)  # 뉴스 키워드
    stock_code = Column(String(255), nullable=False)  # 종목 코드
    stock_name = Column(String(255), nullable=False)  # 종목명

class KeywordStockAssociationScore(Base):
    __tablename__ = "keyword_stock_association_score"

    id = Column(Integer, primary_key=True, autoincrement=True)
    keyword = Column(String(255), nullable=False)
    stock_code = Column(String(255), nullable=False)  # 종목 코드
    stock_name = Column(String(255), nullable=False)  # 종목명
    association_score = Column(Double, nullable=False)
