from typing import Any, Dict, Iterable, Optional
import pandas as pd
from db_mysql import get_conn

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

def get_mcc_map():
    return df("SELECT mcc_id, mcc_code, merchant_name FROM mcc")

def get_card_benefit_by_card(card_id: int):
    return df("""
        SELECT BENEFIT_ID, card_id, category, summary, json_rawdata
        FROM card_benefit
        WHERE card_id=%s
        ORDER BY BENEFIT_ID
    """, (card_id,))
