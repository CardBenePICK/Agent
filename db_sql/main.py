from datetime import date
from repo import get_user_master, get_user_assets, get_mcc_map, get_card_benefit_by_card
from services import build_benefit_context

USER_ID = 101

print("user_master:", get_user_master(USER_ID))

print("\nuser_assets(top5):")
print(get_user_assets(USER_ID, 5))

print("\nmcc(top5):")
print(get_mcc_map().head())

print("\nbenefit context (오늘 기준):")
ctx = build_benefit_context(USER_ID, date.today())
print(ctx.head())

print("\ncard_benefit for card_id=115 (top5):")
print(get_card_benefit_by_card(115).head())
