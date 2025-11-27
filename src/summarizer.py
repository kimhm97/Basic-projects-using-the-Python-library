# src/summarizer.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import math
from tqdm import tqdm

# --------------------------
# 모델 초기화
# --------------------------
MODEL_NAME = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --------------------------
# 배치 요약 함수
# --------------------------
def batch_summarize(texts, batch_size=8, max_input_tokens=1024, max_length=200, min_length=50):
    summaries = []
    num_batches = math.ceil(len(texts) / batch_size)

    for i in tqdm(range(num_batches), desc="Summarizing Batches"):
        batch_texts = texts[i*batch_size : (i+1)*batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_input_tokens)
        inputs = {k:v.to(device) for k,v in inputs.items()}

        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                do_sample=False
            )

        batch_summaries = [tokenizer.decode(g, skip_special_tokens=True) for g in summary_ids]
        summaries.extend(batch_summaries)

    return summaries
