from orange_cb_recsys.evaluation.eval_model import EvalModel
from orange_cb_recsys.evaluation.partitioning import KFoldPartioning
from orange_cb_recsys.recsys.config import RecSysConfig
from orange_cb_recsys.recsys.ranking_algorithms.ranking_algorithm import TopNRanking
from orange_cb_recsys.recsys.recsys import RecSys
from orange_cb_recsys.recsys.score_prediction_algorithms.ratings_based import CentroidVector

import pandas as pd

spa = CentroidVector('Plot', '0')
top_n = TopNRanking(10)

rating_frame = pd.DataFrame.from_records([
    ("A000", "Balto_tt0112453", "sdfgd", 5.0, "54654675"),
    ("A000", "Dracula Dead and Loving It_tt0112896", "sdfgd", 3.0, "54654675"),
    ("A000", "GoldenEye_tt0113189", "sdfgd", 2.0, "54654675"),
    ("A000", "Money Train_tt0113845", "sdfgd", 1.0, "54654675")
], columns=["user_id", "item_id", "original_rating", "derived_score", "timestamp"])

config = RecSysConfig('contents/users_test1591285008.134619',
                      'movielens_test1591028175.9454775',
                      spa,
                      top_n,
                      rating_frame
                      )

recsys = RecSys(config)

eval_model = EvalModel(config, KFoldPartioning(2), True, False, True)
a, b, c = eval_model.fit()
print(a)

"""

import csv

import networkx as nx

graph = nx.DiGraph()

with open('orange_cb_recsys/content_analyzer/result_frame.csv', newline='', encoding='utf-8-sig') as csv_file:
    reader = csv.DictReader(csv_file, quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        graph.add_node(row["imdbID"])

        row_nodes = [row[key] for key in row.keys() if key != "imdbID" and row[key] != '']
        graph.add_nodes_from(row_nodes)

        row_edges = [(row["imdbID"], row_node) for row_node in row_nodes]
        graph.add_edges_from(row_edges)

print(len(graph.nodes()))
page_rank = nx.pagerank(graph)
print(page_rank)

        items = [filename for filename in os.listdir(items_directory)]

        features_bag_list = []
        rated_items_index_list = []
        for item in items:
            item_filename = items_directory + '/' + item
            with open(item_filename, "rb") as content_file:
                content = pickle.load(content_file)

                features_bag_list.append(content.get_field("Plot").get_representation("1").get_value())

        features_bag_list.append(item.get_field("Plot").get_representation("1").get_value())
        v = DictVectorizer(sparse=False)

        X_tmp = v.fit_transform(features_bag_list)

        X = X_tmp[:-1]

import os
import pickle
import random

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

v = DictVectorizer(sparse=False)

items_directory = "contents/movielens_test1591028175.9454775"

items = [filename for filename in os.listdir(items_directory)]

features_bag_list = []
for i, item in enumerate(items):
    item_filename = items_directory + '/' + item
    with open(item_filename, "rb") as content_file:
        content = pickle.load(content_file)

        features_bag_list.append(content.get_field("Plot").get_representation("1").get_value())

    if i==3:
        break

X_tmp = v.fit_transform(features_bag_list)
Y = [int(random.randint(-1, 1)) for i in range(0, 3)]
CLF = DecisionTreeClassifier(random_state=42)

print(Y)

X = X_tmp[1:]

CLF.fit(X, Y)

to_predict = X_tmp[0]

print(CLF.predict([to_predict]))

import os
from pathlib import Path
home = str(Path.home())

with open(os.path.join(home, "test.txt"), "w") as file:
    file.write("gang")

"""
