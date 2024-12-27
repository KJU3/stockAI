from database import get_db
from sentiment.models import *
import logging

logger = logging.getLogger("into_db")

# 1. 데이터베이스 연결 설정
db = next(get_db())

# 2. 모델 저장
def insert_trained_model(model: TrainedModel):
    db.add(model)
    db.commit()
    logger.info("모델이 성공적으로 데이터베이스에 저장되었습니다.")
