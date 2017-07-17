import pandas as pd
import json
from IPython import embed

all_data = pd.read_csv("./data/all_data/train_data.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")

video_data = pd.read_csv("./data/all_data/videos_metadata.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")

video_id = all_data["video_id"]
rel_data = all_data[["video_id", "uploader", "age", "category", "length", "views", "rate", "ratings", "comments"]]
print("joining_2")
rel_data = pd.merge(rel_data, video_data, on='video_id', how='left')

video_category = rel_data[["video_id", "category"]]


video_category_dict = {}
category_videos_dict = {}
for i, row in video_category.iterrows():
    if not i % 1000:
        print(i)
    video_id = str(row[0])
    category = str(row[1])

    if category in category_videos_dict:
        category_videos_dict[category].append(video_id)
    else:
        category_videos_dict[category] = [video_id]

    video_category_dict[video_id] = category

with open('video_to_category_dict.json', 'w') as fp:
    json.dump(video_category_dict, fp)

with open('category_to_videos_dict.json', 'w') as fp:
    json.dump(category_videos_dict, fp)
