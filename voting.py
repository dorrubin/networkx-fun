import numpy as np
import pandas as pd
import networkx as nx
import os
import json
import community
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
from scipy import stats
from IPython import embed


all_data = pd.read_csv("./data/all_data/train_data.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")


def every1000(i):
    if not i % 1000:
        print(i)


def clusterCommunity(src_node, src_cluster, trgt_node, community_dict):
    if trgt_node in community_dict:
        dest_cluster = community_dict[trgt_node]
    else:
        return anyRoom(src_node)
    if src_cluster == dest_cluster:
        return 1
    else:
        return 0


def withinMaxPath(trgt_node, src_lengths):
    length = int(src_lengths.loc[src_lengths['target'] == trgt_node]['path_length'])
    if length < 25 or length == 1000:
        return 1
    else:
        return 0


def commonNeighbors(GG, src_node, trgt_node):
    if trgt_node in GG:
        length = len(list(nx.common_neighbors(GG, src_node, trgt_node)))
        if length > 0:
            return 1
        else:
            return 0
    else:
        return 1


def anyRoom(src_node):
    row = all_data.loc[all_data["video_id"] == src_node]
    count = 0
    for j in range(1, 21):
        if str(row[str(j)]) == "nan":
            count = count + 1
    if count > 9 and count < 20:
        return 1
    else:
        return 0


def createSubmission(prefix, GG, community_dict, shortest_dict, source_to_user_dict, user_to_sources_dict):
    print("createSubmission...")
    test = pd.read_csv(prefix + "test_data.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")
    path_lengths = pd.read_csv("path_lengths.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")
    n_test = len(test.index)
    submission = pd.DataFrame(index=np.arange(0, n_test), columns=('edge_id', 'edge_present'))
    i = 0
    grouped = test.groupby('video_id')
    for src_node, targets in grouped:
        src_cluster = community_dict[src_node]
        src_lengths = path_lengths.loc[path_lengths['source'] == src_node]
        uploader = source_to_user_dict[src_node]
        videos_by_uploader = user_to_sources_dict[uploader]
        in_train = all_data.loc[all_data["video_id"] == src_node]
        connections = in_train[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']]
        count = connections.count(axis=1)
        # print("count - ", int(count))
        # print("num_targets - ", len(targets.index))
        n_r = (count == 20).bool()
        remaining = int(20 - count)
        for j, row in targets.iterrows():
            if n_r:
                final_predict = 0
            else:
                trgt_node = row[1]
                if trgt_node in videos_by_uploader or trgt_node in connections:
                    final_predict = 1
                else:
                    commPredict = clusterCommunity(src_node, src_cluster, trgt_node, community_dict)
                    pathPredict = withinMaxPath(trgt_node, src_lengths)
                    neigPredict = commonNeighbors(GG, src_node, trgt_node)
                    final_predict = stats.mode([commPredict, pathPredict, neigPredict])[0][0]
                    # print("target - ", j , str(commPredict + pathPredict + neigPredict))
            submission.loc[i] = [
                row[2],        # edge_id
                final_predict  # predict edge
            ]
            i += 1
            every1000(i)
            if i == 1:
                break
    submission.to_csv('submission.csv', sep=',', index=False, encoding='utf-8')


def designValidation(validation_data):
    src_target_dict = {}
    for i, row in validation_data.iterrows():
        for j in range(9, 29):
            src_target_dict[(row[0], row[j])] = 1
    return src_target_dict


def validate(validation_data, graph):
    """ Currently the same as the designValidation """
    print("validating...")
    # print(nx.shortest_path(graph, 1, 3))
    validation_dict = {}
    for i, row in validation_data.iterrows():
        for j in range(9, 29):
            validation_dict[(row[0], row[j])] = 1
    return validation_dict


def calculateROC(prediction_dict, validation_dict, trial_results):
    y_true = []
    y_pred = []
    for key in prediction_dict:
        y_pred.append(prediction_dict[key])
        if key in validation_dict:
            y_true.append(validation_dict[key])
        else:
            y_true.append(0)
    # Errors if only one class so forced these values
    y_pred[0] = 0
    y_true[0] = 0
    roc = roc_auc_score(y_true, y_pred)
    trial_results.append(roc)


def createGraph(train_data):
    print("training...")
    MG = nx.MultiGraph()
    for i, row in train_data.iterrows():
        if not i % 1000:
            print(i)
        for j in range(9, 29):
            if not str(row[j]) == "nan":
                MG.add_weighted_edges_from([(row[0], row[j], j)])
    MG.degree(weight='weight')
    GG = nx.Graph()
    for n, nbrs in MG.adjacency_iter():
        for nbr, edict in nbrs.items():
            minvalue = min([d['weight'] for d in edict.values()])
            GG.add_edge(n, nbr, weight=minvalue)
    nx.write_gpickle(GG, "graph.gpickle")
    return GG


def main():
    size = 2
    if size == 1:
        prefix = './data/small_data/'
    elif size == 2:
        prefix = './data/all_data/'
    max_size = 2

    with open('shortest.json', 'r') as fp:
        shortest_dict = json.load(fp)
    if not (os.path.isfile("community.json")):
        parts = community.best_partition(GG)
        with open('community.json', 'w') as fp:
            json.dump(parts, fp)
    with open('community.json', 'r') as fp:
        community_dict = json.load(fp)
    with open('source_to_user_dict.json', 'r') as fp:
        source_to_user_dict = json.load(fp)
    with open('user_to_sources_dict.json', 'r') as fp:
        user_to_sources_dict = json.load(fp)


    all_data = pd.read_csv(prefix + "train_data.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")
    if size < max_size:
        trial_results = []
        ss = ShuffleSplit(n_splits=n_trials, test_size=0.2)
        for train_index, vald_index in ss.split(all_data):
            train_data = all_data.iloc[train_index]
            validation_data = all_data.iloc[vald_index]
            validation_dict = designValidation(validation_data)
            graph = createGraph(train_data)
            prediction_dict = validate(validation_data)
            calculateROC(prediction_dict, validation_dict, trial_results)
        mean_results = np.mean(trial_results)
        # print(trial_results)
        # print(mean_results)
    else:
        if not (os.path.isfile("ranked_graph.gpickle")):
            train_data = all_data
            graph = createGraph(train_data)
        else:
            graph = nx.read_gpickle("ranked_graph.gpickle")
        createSubmission(prefix, graph, community_dict, shortest_dict, source_to_user_dict, user_to_sources_dict)
    # print(nx.average_shortest_path_length(graph))

if __name__ == '__main__':
    np.random.seed(3)
    main()
