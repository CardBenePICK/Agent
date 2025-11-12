from datetime import date, timedelta
import pandas as pd
from db_mysql import get_conn

def month_bounds(d: date):
    start = d.replace(day=1)
    # naive month add
    if start.month == 12:
        nxt = start.replace(year=start.year+1, month=1, day=1)
    else:
        nxt = start.replace(month=start.month+1, day=1)
    return start, nxt

def prev_month_bounds(d: date):
    first, _ = month_bounds(d)
    if first.month == 1:
        pfirst = first.replace(year=first.year-1, month=12, day=1)
    else:
        pfirst = first.replace(month=first.month-1, day=1)
    return month_bounds(pfirst)

def build_benefit_context(user_id: int, ref: date) -> pd.DataFrame:
    m_s, m_e   = month_bounds(ref)
    pm_s, pm_e = prev_month_bounds(ref)
    d_s, d_e   = ref, ref + timedelta(days=1)

    sql = """
    WITH
    hist AS (
      SELECT
        user_id,
        benefit_id,
        applied_amount,
        COALESCE(useage_date, DATE(datetime)) AS use_date,
        SUBSTRING_INDEX(benefit_id, '_', 1) AS card_id
      FROM benefit_history
      WHERE user_id = %s
    ),
    cb AS (  -- card_benefit에서 card_id만
      SELECT DISTINCT card_id FROM card_benefit
    ),
    keys_union AS (
      SELECT DISTINCT card_id FROM cb
      UNION
      SELECT DISTINCT card_id FROM hist
      UNION
      SELECT DISTINCT SUBSTRING_INDEX(benefit_id,'_',1) FROM benefit_limit WHERE user_id=%s
    ),
    prev_m AS (
      SELECT h.card_id, SUM(h.applied_amount) AS prev_month_amount
      FROM hist h
      WHERE h.use_date >= %s AND h.use_date < %s
      GROUP BY h.card_id
    ),
    this_m AS (
      SELECT h.card_id,
             SUM(h.applied_amount) AS month_benefit_amount,
             COUNT(*) AS month_benefit_count
      FROM hist h
      WHERE h.use_date >= %s AND h.use_date < %s
      GROUP BY h.card_id
    ),
    today AS (
      SELECT h.card_id,
             SUM(h.applied_amount) AS day_benefit_amount,
             COUNT(*) AS day_benefit_count
      FROM hist h
      WHERE h.use_date >= %s AND h.use_date < %s
      GROUP BY h.card_id
    ),
    limits_m AS (
      SELECT
        SUBSTRING_INDEX(benefit_id,'_',1) AS card_id,
        SUM(COALESCE(limit_amount,0)) AS monthly_limit_amount,
        SUM(COALESCE(limit_count,0))  AS monthly_limit_count
      FROM benefit_limit
      WHERE user_id=%s AND limit_period='monthly'
      GROUP BY SUBSTRING_INDEX(benefit_id,'_',1)
    ),
    limits_d AS (
      SELECT
        SUBSTRING_INDEX(benefit_id,'_',1) AS card_id,
        SUM(COALESCE(limit_amount,0)) AS daily_limit_amount,
        SUM(COALESCE(limit_count,0))  AS daily_limit_count
      FROM benefit_limit
      WHERE user_id=%s AND limit_period='daily'
      GROUP BY SUBSTRING_INDEX(benefit_id,'_',1)
    )
    SELECT
      k.card_id,
      COALESCE(pm.prev_month_amount, 0)       AS prev_month_amount,
      COALESCE(tm.month_benefit_amount, 0)    AS month_benefit_amount,
      COALESCE(tm.month_benefit_count,  0)    AS month_benefit_count,
      COALESCE(td.day_benefit_amount,   0)    AS day_benefit_amount,
      COALESCE(td.day_benefit_count,    0)    AS day_benefit_count,
      COALESCE(lm.monthly_limit_amount, 0)    AS monthly_limit_amount,
      COALESCE(lm.monthly_limit_count,  0)    AS monthly_limit_count,
      COALESCE(ld.daily_limit_amount,   0)    AS daily_limit_amount,
      COALESCE(ld.daily_limit_count,    0)    AS daily_limit_count
    FROM keys_union k
    LEFT JOIN prev_m  pm ON pm.card_id = k.card_id
    LEFT JOIN this_m  tm ON tm.card_id = k.card_id
    LEFT JOIN today   td ON td.card_id = k.card_id
    LEFT JOIN limits_m lm ON lm.card_id = k.card_id
    LEFT JOIN limits_d ld ON ld.card_id = k.card_id
    ORDER BY CAST(k.card_id AS UNSIGNED);
    """

    params = (
        user_id,         # hist
        user_id,         # keys_union from benefit_limit
        pm_s, pm_e,      # prev_m
        m_s,  m_e,       # this_m
        d_s,  d_e,       # today
        user_id,         # limits_m
        user_id,         # limits_d
    )

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    df = pd.DataFrame(rows)
    if df.empty:
        # 빈 사용자라도 컬럼은 유지
        df = pd.DataFrame(columns=[
            "card_id","prev_month_amount","month_benefit_amount","month_benefit_count",
            "day_benefit_amount","day_benefit_count","monthly_limit_amount",
            "monthly_limit_count","daily_limit_amount","daily_limit_count"
        ])

    # 잔여치 계산
    for col_sum, col_lim, col_left in [
        ("month_benefit_amount","monthly_limit_amount","monthly_amount_left"),
        ("month_benefit_count","monthly_limit_count","monthly_count_left"),
        ("day_benefit_amount","daily_limit_amount","daily_amount_left"),
        ("day_benefit_count","daily_limit_count","daily_count_left"),
    ]:
        if col_sum in df and col_lim in df:
            df[col_left] = df[col_lim].fillna(0) - df[col_sum].fillna(0)

    return df
