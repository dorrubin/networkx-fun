import pandas as pd
import json
from IPython import embed

all_data = pd.read_csv("./data/all_data/train_data.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")

# create a dict of source video ids that map to an uploader. the uploader then needs to map to a list of video_ids that user has uploaded
user_dict = {}
src_dict = {}
for i, row in all_data.iterrows():
    if not i % 1000:
        print(i)
    source = str(row[0])
    uploader = str(row[1])

    if uploader in user_dict:
        user_dict[uploader].append(source)
    else:
        user_dict[uploader] = [source]

    src_dict[source] = uploader

embed()
with open('user_to_sources_dict.json', 'w') as fp:
    json.dump(user_dict, fp)

with open('source_to_user_dict.json', 'w') as fp:
    json.dump(src_dict, fp)
