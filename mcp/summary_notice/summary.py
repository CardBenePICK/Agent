import os
import asyncio
import pymysql
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

# 1. 환경 설정 로드
load_dotenv()

# DB 설정 (환경변수에서 가져오거나 직접 입력)
DB_HOST = os.getenv("MYSQL_SERVER", "localhost")
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "password")
DB_NAME = os.getenv("MYSQL_DB", "card_db")
HF_API_KEY = os.getenv("HF_API_KEY")

# 2. LLM 및 프롬프트 설정 (이전 대화 내용 반영)
chat = ChatOpenAI(
    model="openai/gpt-oss-120b",
    openai_api_key=HF_API_KEY,
    base_url="https://router.huggingface.co/v1",
    temperature=0
)

# 최종 합의된 "Smart Filtering" 프롬프트
summary_system_prompt = """
# Role
당신은 신용카드 안내문 데이터를 정제하여 **'혜택 계산용 데이터'**로 변환하는 전문 AI입니다.

# Goal
입력 텍스트에서 **노이즈(법적 고지, 인사말 등)를 제거**하고, 크롤링 과정에서 구조가 깨진 **'표(Table) 형태의 혜택 조건'을 논리적으로 복원**하여 요약하십시오.

# Processing Logic (Strict)
1. **표 구조 복원 (Table Reconstruction)**:
   - 입력 텍스트가 "구분, 실적, 한도" 등의 헤더와 함께 나열된 경우, 이를 **[혜택명] - [실적조건] - [제공한도]**의 3단 구조로 재조립하십시오.
   - **셀 병합 처리**: 상위 카테고리(예: '추가 할인')가 한 번만 나오고 뒤이어 조건들이 나열되면, 해당 카테고리가 계속 적용되는 것으로 간주하십시오.
   - (예: "추가할인 40만 1만 80만 2만" -> "추가할인(40만):1만", "추가할인(80만):2만")

2. **숫자 데이터 보존 (Data Preservation)**:
   - 금액(1만원), 비율(10%), 횟수(월 5회), 실적 기준(40만원 이상) 등 **계산에 필요한 숫자는 절대 생략하거나 변형하지 마십시오.**

3. **노이즈 완전 삭제 (Smart Filtering)**:
   - "카드사의 사정으로 변경 가능", "약관 참조", "준법감시인 심의필" 등 계산과 무관한 문장은 흔적도 남기지 말고 삭제하십시오.
   - 혜택 정보가 전무하다면 `카드 전체 유의사항 없음`만 출력하십시오.

4. **제외 조건 필수 포착 (Critical)**:
   - 텍스트 내에 **"제외", "포함되지 않음", "미적용", "실적에서 차감"** 등의 표현이 있으면, 해당 항목들을 **반드시 '혜택 제외' 또는 '실적 제외' 카테고리에 포함**시키십시오.
   - (예: "무이자할부 이용금액은 제외매출입니다" -> 혜택 제외: 무이자할부)
   - (예: "상품권 구입은 실적 산정에서 제외" -> 실적 제외: 상품권)

# Output Format
유효한 정보가 있을 경우 아래 형식을 따르십시오.

## 1. 실적별 혜택 한도 (Mapping)
- 형식: `[혜택명] ([실적조건]) : [기간] [금액] [한도여부]`
- **반드시 '통합 한도' 또는 '최대'라는 표현을 사용하여 상한선(Cap)임을 명시하십시오.**
- (예: 추가 할인 (40만원 이상) : 월 **최대** 1만원 **통합 한도**)
- (예: 기본 할인 (실적무관) : 한도 없음)

## 2. 혜택 제외 및 제한 (할인, 혜택 관련) (Constraints)
- **실적 제외**: (실적 산정에 포함되지 않는 항목)
- **혜택 제외**: (할인/적립 대상이 아닌 항목 - 예: 무이자할부, 상품권 등)
- **무이자 할부**: (혜택 적용 여부 및 실적 포함 여부 O/X)
- **필수 조건**: (건당 최소 결제액 등)

"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", summary_system_prompt),
    ("human", "{input_text}")
])

summary_chain = prompt_template | chat | StrOutputParser()

# 3. MySQL 연결 및 처리 함수
def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

def merge_context(items):
    """
    복잡한 중첩 구조(리스트 내 딕셔너리, 테이블 등)를 평탄한 텍스트로 변환합니다.
    """
    parts = []
    
    # 내부 재귀 함수 정의
    def extract_text(data):
        if isinstance(data, str):
            return data.strip()
        
        elif isinstance(data, list):
            # 리스트 내부의 각 항목을 재귀적으로 처리
            # (예: ["문자열", {"table": ...}, "문자열"])
            extracted_list = [extract_text(item) for item in data]
            return "\n".join([t for t in extracted_list if t]) # 빈 문자열 제거
            
        elif isinstance(data, dict):
            # 딕셔너리 내부의 모든 값(value)을 재귀적으로 처리
            # (예: {"subtitle": "제목", "descriptions": [...]})
            extracted_values = []
            for k, v in data.items():
                # table 같은 특수 구조도 결국 list of list 형태일 테니 재귀로 풀림
                text = extract_text(v)
                if text:
                    extracted_values.append(text)
            return "\n".join(extracted_values)
            
        return str(data) # 숫자나 기타 타입은 문자열로 변환

    # 메인 로직
    if isinstance(items, list):
        for item in items:
            text = extract_text(item)
            if text:
                parts.append(text)
    else:
        # items가 리스트가 아니라 단일 객체일 경우
        text = extract_text(items)
        if text:
            parts.append(text)

    return "\n".join(parts)

async def process_card_summaries():
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # A. 요약이 필요한 데이터 가져오기 (이미 요약된 건 제외)
        # 가정: 테이블명 'cards', 원문 컬럼 'notice_raw', 요약 컬럼 'notice_summary'
        print("🔍 요약 대상 데이터를 조회합니다...")
        # sql_select = "SELECT card_id, json_notice FROM card_master where card_id = 51 or card_id = 2719 or card_id = 13 or card_id = 2346"
        sql_select = "SELECT card_id, json_notice FROM card_master"
        cursor.execute(sql_select)
        rows = cursor.fetchall()
        total_count = len(rows)
        
        # for r in rows:
        #     print(r)
        # a = merge_context(json.loads(rows[0]['json_notice']))
        # print(a)
        print(f"📋 총 {total_count}건의 처리할 데이터를 발견했습니다.")

        if total_count == 0:
            return

        # B. 배치 처리 설정 (한 번에 5개씩 처리)
        BATCH_SIZE = 5
        
        # 진행률 바와 함께 배치 처리 시작
        # 리스트를 BATCH_SIZE 만큼 잘라서 순회
        with tqdm(total=total_count, desc="Processing Cards", unit="row") as pbar:
            for i in range(0, total_count, BATCH_SIZE):
                batch_rows = rows[i : i + BATCH_SIZE]

                # 1. LLM에 보낼 텍스트 리스트 준비
                inputs = [{"input_text": merge_context(json.loads(row['json_notice']))} for row in batch_rows]
                
                # 2. 비동기 병렬 요청 (abatch)
                # tqdm을 사용하지 않을 경우: summaries = await summary_chain.abatch(inputs)
                # print(f"🔄 Processing batch {i//BATCH_SIZE + 1}/{(total_count//BATCH_SIZE)+1}...")
                summaries = await summary_chain.abatch(inputs)

                # 3. DB 업데이트 (Batch 단위로 Commit)
                LOG_FILE = "summary_log.md"
                for row, summary_text, origin_text in zip(batch_rows, summaries, inputs):
                    card_id = row['card_id']
                    
                    # print(f"--------card_id : {card_id}--------")
                    # print(f"원래 유의사항 : \n {origin_text['input_text']} \n 요약본 : \n {summary_text}")
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(f"# Card ID: {card_id}\n")
                        f.write("## 1. 원본 유의사항\n")
                        f.write("```\n")
                        f.write(f"{origin_text['input_text']}\n")
                        f.write("```\n\n")
                        f.write("## 2. AI 요약본\n")
                        f.write("```\n")
                        f.write(f"{summary_text}\n")
                        f.write("```\n\n")
                        f.write("\n---\n\n") # 구분선

                    # ⭐️ 요청하신 포맷: [{"content": "요약내용"}]
                    result_json_obj = [{"content": summary_text}]
                    
                    # DB 저장을 위해 문자열로 변환
                    result_json_str = json.dumps(result_json_obj, ensure_ascii=False)

                    # 업데이트 쿼리
                    sql_update = "UPDATE card_master SET summarized_notice = %s WHERE card_id = %s"
                    cursor.execute(sql_update, (result_json_str, card_id))
                
                conn.commit()  # 배치 하나 끝날 때마다 저장 (안전성 확보)
                pbar.update(len(batch_rows))
            
        print(f"\n✅ {total_count}건의 요약 및 업데이트가 완료되었습니다!")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        conn.rollback()
    finally:
        conn.close()

# 4. 메인 실행
if __name__ == "__main__":
    asyncio.run(process_card_summaries())
