# src/run_summarization_small.py
import pandas as pd
from summarizer import batch_summarize
import os

input_csv = "../data/processed/train_small.csv"
output_csv = "../data/processed/summarized_small.csv"

df = pd.read_csv(input_csv, encoding="utf-8")
print("작은 CSV 로드 완료!")

possible_cols = ['document', 'text', 'content']
for col in possible_cols:
    if col in df.columns:
        text_col = col
        break
else:
    raise ValueError(f"CSV에 요약할 컬럼이 없습니다. 가능한 컬럼: {possible_cols}, 실제 컬럼: {df.columns.tolist()}")

df['summary'] = batch_summarize(df[text_col].tolist(), batch_size=8)

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"작은 CSV 요약 완료! 저장 경로: {output_csv}")
