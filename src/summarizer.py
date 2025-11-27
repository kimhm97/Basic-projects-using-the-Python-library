from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# 모델 초기화
MODEL_NAME = "facebook/bart-large-cnn"  # 빠른 요약 모델, LED보다 가벼움
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# GPU 사용 여부
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)


def summarize_text(text, max_input_tokens=1024, max_length=150, min_length=50):
    """
    텍스트 요약 함수
    """
    text = text.replace("\n", " ").strip()
    if len(text) == 0:
        return ""

    # 토크나이즈 후 GPU 전송
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(model.device)

    # 요약 생성
    summary_ids = model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        length_penalty=2.0,
        do_sample=False
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
