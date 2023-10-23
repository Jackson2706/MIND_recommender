import pandas as pd
import json

news_df = pd.read_csv('data/mind-demo/train/news.tsv', sep='\t', header=None)
print(news_df.get(2).head())

elements = list(news_df.get(2))

element_to_id = {'pad': 0,
                 'unk': 1,}
id_counter = 2


for element in elements:
    if element not in element_to_id:
        element_to_id[element] = id_counter
        id_counter += 1
# print("Danh sách gốc:", elements)
# print("Danh sách đã mã hóa:", encoded_list)
# print("Từ điển ánh xạ:", element_to_id)

# Tên tệp JSON để lưu dữ liệu
json_filename = "category2id.json"

# Mở tệp JSON để ghi
with open(json_filename, "w") as json_file:
    json.dump(element_to_id, json_file)

print(f"Dữ liệu đã lưu vào tệp {json_filename}")