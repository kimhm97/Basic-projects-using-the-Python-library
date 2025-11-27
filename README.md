---

# 📝 Transformers Text Summarizer

---

## 📌 프로젝트 소개

Hugging Face Transformers 모델을 활용해 한국어 뉴스 기사를 자동으로 요약 하는 프로젝트입니다.

* 데이터셋: `naver-news-summarization-ko`
* 입력: 뉴스 기사 (`passage`)
* 출력: 요약문 (`generated_summary`)

-뉴스 요약, 데이터 처리 및 자연어 처리(NLP)-

---

## 🗂 프로젝트 구조

```
transformers_Text_Summarizer/
│
├─ data/
│   ├─ raw/                # 원본 CSV
│   │   └─ train.csv
│   └─ processed/          # 요약 결과 저장
│       └─ train_summarized.csv
│
├─ src/
│   ├─ summarizer.py       # 요약 함수
│   └─ run_summarization.py # 실행 코드

│
├─ .gitignore
├─ requirements.txt
└─ README.md
```

---

## ⚙️ 설치 방법

```bash
# 깃허브에서 프로젝트 클론
git clone <your-repo-url>
cd transformers_Text_Summarizer

# 가상환경 생성
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 라이브러리 설치
pip install -r requirements.txt
```

---

## 🚀 사용 방법

```bash
cd src
python run_summarization.py
```

* 처리 완료 시 `data/processed/train_summarized.csv`에 요약 결과 저장
* GPU 사용 가능 시 자동으로 GPU로 실행

---

## 🧠 모델 정보

| 항목 | 내용                        |
| -- | ------------------------- |
| 모델 | `facebook/bart-large-cnn` |
| 유형 | Seq2Seq (텍스트 요약)          |
| 장점 | 빠른 요약, 메모리 효율적            |
| 주의 | 입력 길이 1024 토큰 제한          |

---

