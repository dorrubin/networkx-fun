import networkx as nx
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from IPython import embed

# # ON TRAINING DATA
proc_test = pd.read_csv("./data/preprocessed/test_preprocessed_data.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")
proc_train = pd.read_csv("./data/preprocessed/train_preprocessed_data.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")

test = pd.read_csv("./data/all_data/test_data.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")

for i, row in test.iterrows():

    src_node = row["video_id"]
    trgt_node = row["target"]

    foo = proc_train.loc[proc_train['video_id'] == src_node]
    bar = proc_train.loc[proc_train['video_id'] == trgt_node]
    # print(i, foo.empty, bar.empty)
    if (not foo.empty and not bar.empty):
        foo = foo.drop(["video_id"], axis=1)
        bar = bar.drop(["video_id"], axis=1)
        dist = cosine_similarity(foo, bar)

        print(src_node, ",", trgt_node, ",", dist)



# JUST ON VIDEO METADATA
# video_data = pd.read_csv("./data/all_data/videos_metadata.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")
# test = pd.read_csv("./data/all_data/test_data.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")

# video_ids = video_data["video_id"]
# scaled_video = video_data.drop(["video_id"], axis=1)
# X_scaled = preprocessing.scale(scaled_video)
# X_std = StandardScaler().fit_transform(X_scaled)

# final_video = pd.concat(video_ids, X_std)
# embed()
# for i, row in test.iterrows():

#     src_node = row["video_id"]
#     trgt_node = row["target"]

#     foo = video_data.loc[video_data['video_id'] == src_node]
#     bar = video_data.loc[video_data['video_id'] == trgt_node]
#     # print(i, foo.empty, bar.empty)
#     if (not foo.empty and not bar.empty):
#         foo = foo.drop(["video_id"], axis=1)
#         bar = bar.drop(["video_id"], axis=1)
#         dist = cosine_similarity(foo, bar)

#         print(src_node, ",", trgt_node, ",", dist)
