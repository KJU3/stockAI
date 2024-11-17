from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

# DB_URL = 'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}'
DB_URL = 'mysql+pymysql://root:1234@localhost:3306/aiTest'

# Engine 및 Session 설정
engine = create_engine(DB_URL, pool_recycle=500)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

# 세션 제공 함수
def get_db():
    """
    FastAPI 의존성 주입을 위한 DB 세션 함수
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
