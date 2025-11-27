from typing import Any, Dict, Iterable, Optional
import pandas as pd
from .db_mysql import get_conn
import json

def one_col(sql: str, params: Optional[Iterable[Any]] = None) -> list:
    print(f"ONE_COL SQL: {sql}")
    print(f"ONE_COL PARAMS: {params}")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())
        rows = cur.fetchall()
        print(f"ONE_COL RAW ROWS: {rows}")
    print("row도 궁금하다", rows)
    result = [row[0] for row in rows]
    print(f"ONE_COL FINAL RESULT: {result}")
    return result

def df(sql: str, params: Optional[Iterable[Any]] = None) -> pd.DataFrame:
    print(f"EXECUTING SQL: {sql}")
    print(f"WITH PARAMS: {params}")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())
        rows = cur.fetchall()
        # 컬럼명 가져오기
        columns = [desc[0] for desc in cur.description]
    print(f"QUERY RETURNED {len(rows)} rows")
    print(f"COLUMNS: {columns}")
    return pd.DataFrame(rows, columns=columns)

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
    if mcc_code:
        return mcc_code[0]
    return None
    

def get_total_cardbenefit_by_mcc(user_id : int, mcc : int) -> pd.DataFrame:
    """
        get_mcc_code_by_merchant 함수로 mcc를 구하여서 관련된 모든 혜택 내용을 검색한다.
    """
    user_cardlist = get_user_card_list(user_id)
    print("이거야",user_cardlist)
    
    # JSON_CONTAINS를 위해 '"1111"' 형식으로 변환
    mcc_json_param = f'"{str(mcc)}"'
    print(f"MCC JSON PARAM: {mcc_json_param}")
    
    benefit_df = df("""
        SELECT *
        FROM card_benefit
        WHERE JSON_CONTAINS(mcc_code, %s, '$') and card_id in %s;
    """, (mcc_json_param, user_cardlist))

    return benefit_df

def get_user_card_list(user_id : int) -> list:
    """
        user_id 를 사용해서 사용자가 가지고 있는 모든 카드 리스트를 얻습니다.
    """

    return tuple(one_col("""
            SELECT external_account_id
            FROM user_assets
            WHERE user_id = %s;
        """, (user_id,)))

def get_benefits_by_user_assets_and_mcc(user_id: int, mcc: int) -> pd.DataFrame:
    """
    user_assets에서 주어진 user_id의 external_card_id를 서브쿼리로 사용하여
    card_benefit와 card_master, benefit_sum을 조인 후, 주어진 mcc가 포함된 혜택 행을 반환합니다.
    benefit_sum 테이블과 조인하여 사용자별 혜택 적용 현황도 함께 가져옵니다.

    반환되는 컬럼: card_benefit의 모든 컬럼 + card_master의 card_name, json_notice + benefit_sum의 사용 현황
    """
    # Use the exact SQL query form that works in MySQL Workbench
    # JSON_CONTAINS needs the second parameter as a JSON string literal like '"1111"'
    mcc_json_str = f'"{str(mcc)}"'  # converts 1111 to '"1111"'
    sql = """
        SELECT cm.card_name, cb.benefit_id, cb.card_id, cb.category, cb.summary, cb.json_rawdata, cb.mcc_code, cm.summarized_notice,
               COALESCE(bs.day_amount, 0) as day_amount, 
               COALESCE(bs.day_count, 0) as day_count, 
               COALESCE(bs.week_amount, 0) as week_amount, 
               COALESCE(bs.week_count, 0) as week_count, 
               COALESCE(bs.month_amount, 0) as month_amount, 
               COALESCE(bs.month_count, 0) as month_count, 
               COALESCE(bs.year_amount, 0) as year_amount, 
               COALESCE(bs.year_count, 0) as year_count,
               COALESCE(ct.prev_month_total, 0) as prev_month_total
        FROM card_benefit cb
        JOIN (
            SELECT DISTINCT external_account_id AS card_id
            FROM user_assets
            WHERE user_id = %s
        ) ua ON cb.card_id = ua.card_id
        JOIN card_master cm ON cb.card_id = cm.card_id
        LEFT JOIN benefit_sum bs ON bs.user_id = %s AND bs.benefit_id = cb.benefit_id
        LEFT JOIN (
            SELECT card_id, SUM(amount_krw) as prev_month_total
            FROM card_transactions
            WHERE user_id = %s
              AND transaction_date >= DATE_FORMAT(CURDATE() - INTERVAL 1 MONTH, '%%Y-%%m-01')
              AND transaction_date < DATE_FORMAT(CURDATE(), '%%Y-%%m-01')
            GROUP BY card_id
        ) ct ON ct.card_id = cb.card_id
        WHERE JSON_CONTAINS(cb.mcc_code, %s, '$')
        ORDER BY cb.benefit_id
    """
    benefit_df = df(sql, (user_id, user_id, user_id, mcc_json_str))
    benefit_df.to_csv("debug_benefit_df.csv")
    return benefit_df

def get_user_benefit_limit_in_benefit_sum(user_id: int) -> pd.DataFrame:
    """
    해당 user가 이번 기간에 적용받은 모든 혜택의 금액과 횟수를 조회해서 반환합니다.
    """
    sql = """
    SELECT * FROM benefit_sum
    WHERE user_id = %s"""

    benefit_df = df(sql, (user_id,))
    return benefit_df

def get_transaction_sum_by_user(user_id: int) -> pd.DataFrame:
    """
    해당 유저의 전월실적을 반환합니다.
    현재 API 요청 시각 기준으로 전월의 거래만 조회합니다.
    """
    sql = """
    SELECT user_id, card_id, SUM(amount_krw) as prev_month_total
    FROM card_transactions
    WHERE user_id = %s 
      AND transaction_date >= DATE_FORMAT(CURDATE() - INTERVAL 1 MONTH, '%%Y-%%m-01')
      AND transaction_date < DATE_FORMAT(CURDATE(), '%%Y-%%m-01')
    GROUP BY user_id, card_id;"""
    
    benefit_df = df(sql, (user_id,))
    return benefit_df