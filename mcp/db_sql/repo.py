from typing import Any, Dict, Iterable, Optional
import pandas as pd
from db_mysql import get_conn

def one_col(sql: str, params: Optional[Iterable[Any]] = None) -> map:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())
        rows = cur.fetchall()
    return map(lambda x: x[0], rows)

def df(sql: str, params: Optional[Iterable[Any]] = None) -> pd.DataFrame:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())
        rows = cur.fetchall()
    return pd.DataFrame(rows)

def one(sql: str, params: Optional[Iterable[Any]] = None) -> Optional[Dict]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())
        row = cur.fetchone()
    return row

# --- 테이블별 헬퍼들 ---
def get_user_master(user_id: int):
    return one("SELECT user_id, uuid FROM user_master WHERE user_id=%s", (user_id,))


def get_user_assets(user_id: int, limit:int=20):
    return df("""
        SELECT * FROM user_assets
        WHERE user_id=%s ORDER BY updated_at DESC LIMIT %s
    """, (user_id, limit))

# mcc table 전체 정보
def get_mcc_map():
    return df("SELECT mcc_id, mcc_code, merchant_name FROM mcc")

# card id로 카드가 가지고 있는 혜택들 얻기
def get_card_benefit_by_card(card_id: int):
    return df("""
        SELECT BENEFIT_ID, card_id, category, summary, json_rawdata
        FROM card_benefit
        WHERE card_id=%s
        ORDER BY BENEFIT_ID
    """, (card_id))

def get_mcc_code_by_merchant(merchant : str) -> int:
    """
        DB에 있는 영업점 이름, 검색 이름의 공백을 없애서 검색
        만약 찾고자 하는 것이 없다면 None을 반환한다.
        사용법 :
        mcc_code = get_mcc_code_by_merchant('gs 25')
        
    """
    mcc_code = one("""
        SELECT mcc_code FROM mcc 
        WHERE REPLACE(merchant_name, ' ', '') = 
            REPLACE(%s, ' ', '');
    """, (merchant))
    if mcc_code :
        return mcc_code[0]
    return None
    

def get_total_cardbenefit_by_mcc(user_id : int, mcc : int) -> pd.DataFrame:
    """
        get_mcc_code_by_merchant 함수로 mcc를 구하여서 관련된 모든 혜택 내용을 검색한다.
    """
    user_cardlist = get_user_card_list(user_id)
    
    benefit_df = df("""
        SELECT *
        FROM card_benefit
        WHERE JSON_CONTAINS(mcc_code, %s, '$') and card_id in %s;
    """, (str(mcc), user_cardlist))

    return benefit_df

def get_user_card_list(user_id : int) -> list:
    """
        user_id 를 사용해서 사용자가 가지고 있는 모든 카드 리스트를 얻습니다.
    """

    return tuple(one_col("""
            SELECT card_id
            FROM user_card_temp
            WHERE user_id = %s;
        """, (user_id)))