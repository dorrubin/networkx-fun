import networkx as nx
import pandas as pd
import numpy as np
import json
from IPython import embed

G = nx.read_gpickle("ranked_graph.gpickle")

all_data = pd.read_csv("./data/all_data/train_data.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")
# row_1 = all_data.iloc[1]
# row_2 = all_data.iloc[2]
# embed()

actual = []
for index, row in all_data.iterrows():
    bunch = []
    for j in range(1, 21):
        if not str(row[str(j)]) == "nan":
            # bunch.append((row[0], row[str(j)]))
            preds = nx.adamic_adar_index(G, [(row["video_id"], row[str(j)])])
            # preds = nx.jaccard_coefficient(G, [(row["video_id"], row[str(j)]), (row["video_id"], row[str(j)])])
            # embed()
            for u, v, p in preds:
                print(u, v, p)
            preds = nx.jaccard_coefficient(G, [(row["video_id"], row[str(j)])])
            for u, v, p in preds:
                print(u, v, p)
    if index == 3:
        break

print(actual)
# print(np.mean(actual))
