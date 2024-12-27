import pandas as pd
from database import get_db
from sentiment.models import *

# 1. 데이터베이스 연결 설정
db = next(get_db())

# 2. 테이블 생성

# 3. 엑셀 파일 로드
excel_file_path = r"C:/Dev/private/ai_python/app/stock.xlsx"  # 엑셀 파일 경로
df = pd.read_excel(excel_file_path)

# 4. 데이터 삽입
for _, row in df.iterrows():
    stock = Stock(
        stock_code=row['stock_code'],
        stock_type=row['stock_type'],
        stock_name=row['stock_name'],
        issued_share=row['issued_share'],
        industry_name=row['industry_name'],
        market_cap=row['market_cap'],
    )
    db.add(stock)

# 5. 변경사항 커밋
db.commit()

print("엑셀 데이터가 성공적으로 데이터베이스에 저장되었습니다.")
