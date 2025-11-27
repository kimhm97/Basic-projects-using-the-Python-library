import pandas as pd
import os

# 원본 CSV 경로
input_csv = "../data/raw/train.csv"

# 저장할 작은 CSV 경로
output_csv_small = "../data/processed/train_small.csv"

# CSV 불러오기
for enc in ['utf-8', 'cp949', 'latin1']:
    try:
        df = pd.read_csv(input_csv, encoding=enc)
        print(f"CSV 로드 성공! (encoding={enc})")
        break
    except Exception as e:
        print(f"encoding={enc} 실패: {e}")
else:
    raise ValueError(f"CSV 파일을 로드할 수 없습니다: {input_csv}")

# 일부만 추출 (예: 처음 500개)
df_small = df.head(500)

# 폴더 없으면 생성
os.makedirs(os.path.dirname(output_csv_small), exist_ok=True)

# 작은 CSV 저장
df_small.to_csv(output_csv_small, index=False, encoding='utf-8-sig')
print(f"작은 CSV 생성 완료! 경로: {output_csv_small}")
