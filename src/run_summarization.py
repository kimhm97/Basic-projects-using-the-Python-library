import pandas as pd
from tqdm import tqdm
from summarizer import summarize_text
import os

# 경로 설정
RAW_CSV = "../data/raw/train.csv"
OUTPUT_CSV = "../data/processed/train_summarized.csv"

# CSV 불러오기
df = pd.read_csv(RAW_CSV, encoding="utf-8")

# 요약 처리
summaries = []
for passage in tqdm(df['passage'], desc="Summarizing"):
    try:
        summaries.append(summarize_text(passage))
    except:
        summaries.append("")

# 결과 저장
df['generated_summary'] = summaries
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"✅ 요약 완료! 저장 경로: {OUTPUT_CSV}")
