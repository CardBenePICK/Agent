from datetime import date
from repo import get_user_card_list, get_user_master, get_user_assets, get_mcc_map, get_card_benefit_by_card, get_mcc_code_by_merchant, get_total_cardbenefit_by_mcc
from services import build_benefit_context

USER_ID = 101

# print("user_master:", get_user_master(USER_ID))

# print("\nuser_assets(top5):")
# print(get_user_assets(USER_ID, 5))

# print("\nmcc(top5):")
# print(get_mcc_map().head())

# print("\nbenefit context (오늘 기준):")
# ctx = build_benefit_context(USER_ID, date.today())
# print(ctx.head())

# print("\ncard_benefit for card_id=115 (top5):")
# print(get_card_benefit_by_card(115).head())

# mer = get_mcc_code_by_merchant("써브웨이")

# print(mer)
# print(type(mer))

# card_benefit = get_total_cardbenefit_by_mcc(str(mer))
# type(card_benefit[4][0])

# b = list(map(lambda x : json.loads(x)[0], card_benefit[4]))
# b
# for bb in b:
#     print(bb)
# import json
# aa = json.loads(card_benefit[4][0])[0]
# aa
# type(aa)
# len(aa)
# for aaa in aa:
#     print(aaa)
# type(a)


card_benefit = get_total_cardbenefit_by_mcc(USER_ID, 4011)
print(card_benefit)

import json
b = list(map(lambda x : json.loads(x), card_benefit[4]))

def merge_context(items):
    parts = []
    for item in items:
        for key, value in item.items():
            if isinstance(value, list):
                # 리스트일 경우 각 항목을 줄바꿈으로 합침
                if value:
                    parts.append('\n'.join(value))
            elif value:  # 문자열 등 일반 값
                parts.append(str(value))
    return '\n'.join(parts)

for idx, bb in enumerate(b):
    context = merge_context(bb)
    print(idx)
    print(context)



for bb in b:
    print(bb)

card_list = get_user_card_list(USER_ID)

print(card_list)



a = (1,)
str(a)
type(a)