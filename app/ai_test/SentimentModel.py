from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mysql import *
from sqlalchemy import Column

Base = declarative_base()

class SentimentModel(Base) :
    __tablename__ = "sentiment_model"

    id = Column(BIGINT, primary_key=True, autoincrement=True)
    content = Column(VARCHAR(255))
    model = Column(LONGBLOB, nullable=False)

