import json

# Define the input and output file names
input_file = "credit_card_top100_v4.json"
output_file = "card_ids_top100.json"

# --- 1. Read the input JSON file ---
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: The input file '{input_file}' was not found. Please ensure it is uploaded.")
    # Stop execution if the file is not found
    data = []

# --- 2. Extract and convert to integers ---
card_ids_int = []
for card in data:
    if 'card_id' in card:
        try:
            # Convert string ID to integer
            card_ids_int.append(int(card['card_id']))
        except ValueError:
            # Handle cases where card_id might not be an integer string
            print(f"Warning: Skipping non-integer card_id value: {card['card_id']}")
            pass

# --- 3. Sort the list of integers ---
card_ids_int.sort()

# --- 4. Print count and preview ---
count = len(card_ids_int)
print(f"총 카드 ID 개수: {count}개")

# Prepare a preview string
preview_count = min(7, count)
preview_list = card_ids_int[:preview_count]
preview_output = json.dumps(preview_list, indent=None)

# Format the output preview exactly as requested
if count > preview_count:
    preview_output = preview_output[:-1] + ", ...]"
else:
    preview_output = preview_output
    
print(f"정렬된 카드 ID 리스트의 미리보기 (앞 {preview_count}개): {preview_output}")

# --- 5. Save the sorted list to a new JSON file ---
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(card_ids_int, f, indent=2)

print(f"정렬된 카드 ID 리스트가 '{output_file}' 파일에 저장되었습니다.")