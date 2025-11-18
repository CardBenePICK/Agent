import requests
import os
from dotenv import load_dotenv

load_dotenv()
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL")

API_URL = f"http://{LOCAL_LLM_URL}/api/chat"  # Docker 호스트/로컬에서 테스트
API_URL = "http://182.226.200.232:11434/api/chat"
# API_URL = "http://localhost:11434/api/chat"  # Docker 호스트/로컬에서 테스트

def invoke_question(llm_model : str, prompt, context, question):
    context = context_str # 횬재는 임의로 사용.
    try:
        
        payload = {
            "model": llm_model,
            "messages": [
                # 시스템 프롬프트 추가: 모델의 역할을 정의
                {"role": "system", "content": prompt},
                
                # 컨텍스트가 추가된 사용자 메시지
                {"role": "user", "content": f""" 
                    Context : {context} \n

                    --- 


                    Question : {question} \n
                   

                """}
            ],
            "stream": False,
            "format": "json"
        }

        response = requests.post(API_URL, json=payload)
        ans = response.json()
        return ans['message']["content"]
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return {}
    
    



# https://www.card-gorilla.com/card/detail/2851 이건 유의 사항 대분류에만 할인 한도, 전월실적 등이 적혀있음. 

context_str = """
결제날짜 : 2025-09-06 토요일 (KST)
결제시간 : 19:08:54

1. 
card name : 신한카드 Mr.Life

- 고객 혜택 사용 정보 :

월간에 받은 혜택 누적합 : 4800원
전월 실적 달성 금액 : 38만 2000원
일간 혜택 사용 횟수 : 0회
주간 혜택 사용 횟수 : 3회
월간 혜택 사용 횟수 : 6회
년간 혜택 사용 횟수 : 7회

-카드 혜택 및 제한 내용 :

편의점 - 편의점 10% 할인

TIME 할인서비스 10% 할인 

편의점 10% 할인

- 편의점 업종 

- 서비스 영역별 각각 일 1회 할인 적용

- 1회 이용 금액 1만원까지 할인 적용 (1회 최대 1천원 할인)

- 월 5회 할인 적용



TIME 할인서비스 할인한도안내

- 전월실적 30만원~50만원 : 1만원

- 전월실적 50만원 ~100만원 : 2만원

- 전월실적 100만원 이상 : 3만원 



유의사항

- 서비스 대상 거래건 중 신한카드 전표 매입 순서대로 결제일 할인이 적용됩니다.

- TIME 할인서비스는 전월 이용금액에 따라 제공된 월 할인 한도 내에서 서비스가 제공됩니다.

- 신규 발급 회원에 대해서는 카드사용 등록월의 익월말(등록월+1개월)까지 1만원의 할인한도가 제공됩니다.

- 편의점, 병원/약국, 세탁소, 식음료는 신한카드 가맹점 업종 기준으로 할인이 제공됩니다.

- Time 할인서비스는 주중/주말 상관없이 제공되는 서비스이며, Night Time 할인서비스는 승인시간 기준으로 오후 9시부터 오전 9시까지 제공됩니다.

- 병원/약국 10% 할인서비스 대상 가맹점에서 동물병원은 제외되며, 치과/한의원은 포함 제공됩니다.

- 티켓몬스터, 위메프, 쿠팡 사이트 직접 접속 시 혹은 대상가맹점 앱을 통해서 접속 시에만 할인이 적용됩니다.

- 전월 이용실적은 일시불+할부 금액 기준이며 (전월 할인거래 실적 포함) 교통이용금액은 전전월 이용금액 기준, 해외이용금액은 매입일자를 기준으로 적용됩니다.

- 월 기준 : 매월 1일 ~ 말일

"""

if __name__ == "__main__":
    import json
    with open('C:/ITStudy/Project/Final/CardBenePICK/mcp/prompt/prompt.json', 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)

        print("prompt_json을 불러왔습니다." + prompt_data["get_sale_local"][:20])
    question ="GS25에서 80000원 사용 예정. 카드 추천 해줘."
    answer = invoke_question(llm_model="qwen3:8b", prompt=prompt_data["get_sale_local"], context="", question=question)
    print(answer)