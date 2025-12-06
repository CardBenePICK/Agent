import sys
import io
from dotenv import load_dotenv
# 표준 출력을 UTF-8로 강제 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- 기존 코드 시작 ---
import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
# 또는 구버전: from langchain_community.embeddings import HuggingFaceEndpointEmbeddings
load_dotenv()
# API 키 및 모델 설정
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2" # 예시 모델

# 1. 간결해진 임베딩 객체 생성
embeddings = HuggingFaceEndpointEmbeddings(
    model=EMBEDDING_MODEL_ID,
    task="feature-extraction",
    huggingfacehub_api_token=HF_API_KEY,
)

# 2. 테스트 실행 (한번만 호출해보기)
text = "This is a test sentence."

try:
    # 쿼리 임베딩 (단일 문장)
    query_result = embeddings.embed_query(text)
    
    # 문서 임베딩 (여러 문장 리스트)
    # doc_results = embeddings.embed_documents([text])

    print(f"임베딩 성공! 벡터 길이: {len(query_result)}")
    print(f"벡터 앞부분 예시: {query_result[:5]}")

except Exception as e:
    print(f"에러 발생: {e}")