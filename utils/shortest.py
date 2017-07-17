import networkx as nx
import json
import pandas as pd
from IPython import embed


path_lengths = pd.read_csv("path_lengths.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")
foo = path_lengths.loc[path_lengths['source'] == 'lUjVSKfdCWU'].loc[path_lengths['target'] == 'yL2faZW8Htc']['path_length']
bar = int(foo)
embed()

# all_data = pd.read_csv("./data/all_data/train_data.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")
# print("createGraph...")
# G = nx.Graph()
# for i, row in all_data.iterrows():
#     if not i % 1000:
#         print(i)
#     source = str(row[0])
#     G.add_node(source)
#     for j in range(9, 29):
#         target = str(row[j])
#         if not target == "nan":
#             G.add_node(target)
#             try:
#                 if G[source][target]['weight'] > j:
#                     # we added this one before, update to shorter-weight path
#                     G[source][target]['weight'] = j
#             except:
#                 # Add the new edge with a weight equal to its rank
#                 G.add_edge(source, target, weight=j)
# nx.write_gpickle(G, "ranked_graph.gpickle")



# G = nx.read_gpickle("ranked_graph.gpickle")
# total_shortest_paths = {}

# i = 0
# for n, d in G.nodes_iter(data=True):
#     length = nx.single_source_dijkstra_path_length(G, n, cutoff=22)
#     total_shortest_paths[n] = length
#     if i % 1000 == 0:
#         print(i)
#     i = i + 1

# embed()
# with open('shortest.json', 'w') as fp:
#     json.dump(total_shortest_paths, fp)


