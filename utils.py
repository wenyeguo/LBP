import copy
import csv
import math
import pickle
import sys

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold


class GraphPlot:
    def __init__(self, graph, data):
        self.graph = graph
        self.train_set = data['train']
        self.test_set = data['test']
        self.predicts = self.initial_predicts()

    def initial_predicts(self):
        dic = {}
        for node in self.test_set:
            dic[node] = -1
        return dic

    def check_if_converge(self, iteration):
        newPredict = self.predict_nodes()
        num = 0
        for node, label in self.predicts.items():
            if node in self.test_set:
                updated_label = newPredict[node]
                if label == updated_label:
                    num += 1
        # print(f'Iteration {iteration} not changed node {num}')
        self.predicts = copy.deepcopy(newPredict)
        return num
        # if num == self.graph.number_of_nodes() - len(self.train_set):
        #     return True
        # return False

    def update_predict(self, data):
        for node in self.predicts:
            self.predicts[node] = data[node]
        print('updated')

    def predict_nodes(self):
        predict_labels = {}
        for node in self.graph.nodes:
            if node not in self.train_set:
                prior_probability = math.log(1 - self.graph.nodes[node]["label"])
                cost = [prior_probability + msg for msg in self.graph.nodes[node]['msg_sum']]
                if cost[0] < cost[1]:
                    predict_labels[node] = 0
                else:
                    predict_labels[node] = 1
        return predict_labels
        # self.store_predict_result(predict_labels)

    def store_predict_result(self, data):
        with open(f'iteration_{self.iteration}_predict.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Hidden Node', 'Predict Label'])
            for node in self.test_set:
                writer.writerow([node, data[node]])

            for node, label in data.items():
                if node not in self.test_set:
                    writer.writerow([node, label])

    def draw_graph(self, iteration):
        self.set_node_colors()
        b, m = self.set_node_sizes()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        pos = nx.spring_layout(self.graph, seed=30, scale=0.1)
        # pos_m = nx.spring_layout(self.graph, seed=30, scale=0.1)
        labels_benign = self.set_node_labels(b)
        self.networkx_draw('benign_size', pos, labels_benign, axes[0])
        axes[0].set_title('Graph Benign Message Sum')

        labels_malicious = self.set_node_labels(m)

        self.networkx_draw('malicious_size', pos, labels_malicious, axes[1])
        axes[1].set_title('Graph Malicious Message Sum')
        labels_predict_label = self.set_node_label_as_predict_label()
        self.networkx_draw('malicious_size', pos, labels_predict_label, axes[2])
        axes[2].set_title(f'Iteration {iteration}, Graph With Predict Label')
        plt.tight_layout()

        plt.show()

    def set_node_labels(self, size):
        labels = {}
        for node in self.graph.nodes:
            if node in self.train_set:
                labels[node] = 'label - ' + str(self.graph.nodes[node]['label'])
            else:
                labels[node] = str(size[node])
        return labels

    def set_node_label_as_predict_label(self):
        labels = {}
        for node in self.graph.nodes:
            if node in self.train_set:
                labels[node] = str(self.graph.nodes[node]['label'])
            else:
                msg_sum = self.graph.nodes[node]['msg_sum']
                if msg_sum[0] < msg_sum[1]:
                    labels[node] = 0
                else:
                    labels[node] = 1
        return labels

    def networkx_draw(self, size, pos, labels, ax):
        nx.draw(self.graph, pos, with_labels=True,
                node_color=list(nx.get_node_attributes(self.graph, 'color').values()),
                node_size=list(nx.get_node_attributes(self.graph, size).values()),
                labels=labels, font_weight='bold', font_size=10, ax=ax)

    def set_node_colors(self):
        node_colors = {}
        for node in self.graph.nodes:
            if node in self.test_set:
                node_colors[node] = 'blue'
            elif node in self.train_set:
                node_colors[node] = 'red'
            else:
                node_colors[node] = 'yellow'
        nx.set_node_attributes(self.graph, node_colors, 'color')

    def set_node_sizes(self):
        node_benign_sizes = {}
        node_malicious_sizes = {}
        for node in self.graph.nodes:
            msg_sum = self.graph.nodes[node]['msg_sum']
            if node in self.train_set:
                node_benign_sizes[node] = 10.0000
                node_malicious_sizes[node] = 10.0000
            else:
                node_benign_sizes[node] = round(abs(msg_sum[0]), 4) if msg_sum[0] != 0 else 0.0010
                node_malicious_sizes[node] = round(abs(msg_sum[1]), 4) if msg_sum[1] != 0 else 0.0010
        nx.set_node_attributes(self.graph, node_benign_sizes, 'benign_size')
        nx.set_node_attributes(self.graph, node_malicious_sizes, 'malicious_size')
        return node_benign_sizes, node_malicious_sizes


class GraphNode:
    def __init__(self, graph):
        self.graph = graph
        self.train = None
        self.test = None
        self.url_labels = None

    def get_graph(self):
        return self.graph

    def init_nodes(self, data, urls):
        self.train = data['train']
        self.test = data['test']
        self.url_labels = urls
        self.init_all_graph_nodes()
        self.init_train_node()
        # self.init_test_node_cost()
        # for node in self.graph.nodes:
        #     cost = self.graph.nodes[node]['cost']
        return self.graph

    def init_all_graph_nodes(self):
        for node in self.graph.nodes():
            self.graph.nodes[node]["label"] = 0.5
            self.graph.nodes[node]["predict_label"] = -1
            self.graph.nodes[node]["cost"] = [0.5, 0.5]
            self.graph.nodes[node]["msg_sum"] = [0, 0]
            self.graph.nodes[node]["msg_nbr"] = {}
            for nbr in list(self.graph.neighbors(node)):
                self.graph.nodes[node]["msg_nbr"][nbr] = [0, 0]

    def init_train_node(self):
        for node in self.train:
            label = self.url_labels[node]
            self.graph.nodes[node]['label'] = label
            if label == 1:
                self.graph.nodes[node]['cost'] = [0.99, 0.01]
            elif label == 0:
                self.graph.nodes[node]['cost'] = [0.01, 0.99]
            else:
                raise NameError('wrong label detected')
    def init_test_node_cost(self):
        with open('./basic_models/features/benign_probability.pickle', 'rb') as f:
            benign_probability = pickle.load(f)
        for node in self.test:
            proba = benign_probability[node]
            self.graph.nodes[node]['cost'] = [proba, round(1 - proba, 4)]
        # for node in self.graph.nodes:
        #     cost = self.graph.nodes[node]['cost']
    def infer_node_labels(self):
        self.assign_predicted_label_to_hidden_nodes()
        self.assign_predicted_label_to_train_nodes()

    def assign_predicted_label_to_hidden_nodes(self):
        same = 0
        for url in self.test:
            prior_probability = math.log(1 - self.graph.nodes[url]["label"])
            cost = [prior_probability + msg for msg in self.graph.nodes[url]['msg_sum']]
            if cost[0] == cost[1]:
                same += 1
            if cost[0] < cost[1]:
                self.graph.nodes[url]["predict_label"] = 0
            else:
                self.graph.nodes[url]["predict_label"] = 1
            self.graph.nodes[url]['cost'] = cost
        # print('TEST URLS Same Cost', same)

    def assign_predicted_label_to_train_nodes(self):
        for node in self.train:
            cost_benign = self.graph.nodes[node]['cost'][0]
            cost_phish = self.graph.nodes[node]['cost'][1]
            if cost_benign < cost_phish:
                self.graph.nodes[node]["predict_label"] = 0
            else:
                self.graph.nodes[node]["predict_label"] = 1

    def calculate_performance(self):
        TP_B_B, TN_M_M, FP_B_M, FN_M_B = self.compare_hidden_nodes_predicted_label_with_actual_label()
        accuracy = round((TN_M_M + TP_B_B) / len(self.test) if len(self.test) != 0 else float(0), 4)
        recall = round(TP_B_B / (TP_B_B + FN_M_B) if TP_B_B + FN_M_B != 0 else float(0), 4)
        precision = round(TP_B_B / (TP_B_B + FP_B_M) if TP_B_B + FP_B_M != 0 else float(0), 4)
        f1score = round(2 * (precision * recall) / (precision + recall) if precision + recall != 0 else float(0), 4)
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print(f'TP, TN, FP, FN = {TP_B_B}, {TN_M_M}, {FP_B_M}, {FN_M_B}')
        # print(f'Accuracy, Recall, Precision, F1 = {accuracy}, {recall}, '
        #       f'{precision}, {f1score} ')
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return accuracy, recall, precision, f1score

    def compare_hidden_nodes_predicted_label_with_actual_label(self):
        train_right, train_wrong, test_right, test_wrong = 0, 0, 0, 0
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
        for url in self.test:
            actual_label = self.url_labels[url]
            predict_label = self.graph.nodes[url]['predict_label']
            if actual_label == predict_label:
                test_right += 1
                if predict_label == 0:
                    true_positive += 1
                else:
                    true_negative += 1
            else:
                test_wrong += 1
                if predict_label == 0:
                    false_positive += 1
                else:
                    false_negative += 1
        return true_positive, true_negative, false_positive, false_negative


class Similarity:
    def __init__(self, graph, sim_type, file):
        self.type = sim_type
        self.graph = graph
        self.file = file
        self.emb = None
        self.min = sys.maxsize
        self.max = - sys.maxsize - 1

    def get_similarity_range(self):
        return self.min, self.max

    def init_edge_similarity(self):
        self.emb = self.file.get_data()
        for edge in self.graph.edges():
            v1, v2 = self.normalize(edge[0]), self.normalize(edge[1])
            sim = self.calculate_similarity(v1, v2)
            self.min = min(sim, self.min)
            self.max = max(sim, self.max)
            self.graph.edges[edge]['sim'] = sim

    def normalize(self, node):
        v = self.emb[node]
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def calculate_similarity(self, v1, v2):
        sim = 0
        if self.type == 'cos':
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        elif self.type == 'rbf':
            # m = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            distance = np.linalg.norm(v1 - v2)
            sim = np.exp((-1.0 / 2.0) * np.power(distance, 2.0))
        return sim


class Message:
    def __init__(self, graph, edge_type):
        self.prev_graph = copy.deepcopy(graph)
        self.graph = graph
        self.type = edge_type
        self.send = None
        self.receive = None
        self.ths1 = None
        self.ths2 = None

    def set_send_message_thresholds(self, t1, t2):
        self.ths1 = t1
        self.ths2 = t2

    def set_send_node_name(self, send):
        self.send = send

    def set_receive_node_name(self, receive):
        self.receive = receive

    def send_message(self, sender):
        self.set_send_node_name(sender)
        label = self.graph.nodes[sender]['label']
        if label == 1 or label == 0:
            self.send_message_from_known_node(label)
        else:
            self.send_message_from_unknown_node()

    def send_message_from_known_node(self, label):
        if label == 0:
            msg = [0, 1]
        else:
            msg = [1, 0]
        for nbr in self.graph.neighbors(self.send):
            if self.graph.nodes[nbr]["label"] == 0.5:
                self.set_receive_node_name(nbr)
                self.update_receive_node_message(msg)

    def send_message_from_unknown_node(self):
        for nbr in self.graph.neighbors(self.send):
            if self.graph.nodes[nbr]["label"] == 0.5:
                self.set_receive_node_name(nbr)
                msg = self.min_sum()
                self.update_receive_node_message(msg)

    def update_receive_node_message(self, msg):
        send_node, receive_node = self.graph.nodes[self.send], self.graph.nodes[self.receive]
        for i in range(2):
            receive_node['msg_sum'][i] = round(
                receive_node['msg_sum'][i] + (-receive_node['msg_nbr'][self.send][i]) + msg[i], 10)
            receive_node['msg_nbr'][self.send][i] = round(msg[i], 10)

    def min_sum(self):
        send_node = self.graph.nodes[self.send]
        msg = [0] * 2
        receive_prior_benign = self.graph.nodes[self.receive]['cost'][0]
        receive_prior_malicious = self.graph.nodes[self.receive]['cost'][1]
        sumOfSendNode = self.sums_except_receive_node(send_node)
        for assumeLabel in range(2):
            edges = self.calculate_edge_potential(assumeLabel)
            # TODO try remove log
            # msg_benign = round(math.log(1 - receive_prior_benign) + edges[0] + sumOfSendNode[0], 10)
            # msg_phishing = round(math.log(1 - receive_prior_malicious) + edges[1] + sumOfSendNode[1], 10)
            msg_benign = round(1 - receive_prior_benign + edges[0] + sumOfSendNode[0], 10)
            msg_phishing = round(1 - receive_prior_malicious + edges[1] + sumOfSendNode[1], 10)
            msg[assumeLabel] = min(msg_benign, msg_phishing)
        return msg

    def calculate_edge_potential(self, assumeLabel):
        edges = [0.0] * 2
        if self.type == 't1':
            edges = self.edge_potential_epsilon(assumeLabel)
        else:
            sim = self.graph[self.send][self.receive]['sim']
            if self.type == 'sim_only':
                edges = self.edge_potential_similarity_only(assumeLabel, sim)
            elif self.type == 'sim':
                edges = self.edge_potential_similarity(assumeLabel, sim)
            elif self.type == 'sim_max':
                edges = self.edge_potential_similarity_max(assumeLabel, sim)
        return [round(e, 10) for e in edges]

    def edge_potential_epsilon(self, assumeLabel):
        e = 0.0001
        if assumeLabel == 0:
            edges = [0.5 + e, 0.5 - e]
        else:
            edges = [0.5 - e, 0.5 + e]
        return edges
    def edge_potential_similarity_only(self, assumeLabel, sim):
        if assumeLabel == 0:
            edges = [1 - sim, sim]
        else:
            edges = [sim, 1 - sim]
        return edges

    def edge_potential_similarity(self, assumeLabel, sim):
        if assumeLabel == 0:
            edges = [min(self.ths1, 1 - sim), max(self.ths2, sim)]
        else:
            edges = [max(self.ths2, sim), min(self.ths1, 1 - sim)]
        return edges

    def edge_potential_similarity_max(self, assumeLabel, sim):
        if assumeLabel == 0:
            edges = [max(self.ths1, 1 - sim), min(self.ths2, sim)]
        else:
            edges = [min(self.ths2, sim), max(self.ths1, 1 - sim)]
        return edges

    def sums_except_receive_node(self, send_node):
        return [round(a - b, 10) for a, b in zip(send_node['msg_sum'], send_node['msg_nbr'][self.receive])]

    # TODO calculate accuracy/iteration
    def calculate_accuracy(self, testSet, urls):
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
        for node in testSet:
            label = urls[node]
            prior_probability = math.log(1 - self.graph.nodes[node]["label"])
            cost = [prior_probability + msg for msg in self.graph.nodes[node]['msg_sum']]
            if cost[0] < cost[1]:
                predict_label = 0
            else:
                predict_label = 1
            if predict_label == label:
                if predict_label == 0:
                    true_positive += 1
                else:
                    true_negative += 1
            else:
                if predict_label == 0:
                    false_positive += 1
                else:
                    false_negative += 1
        accuracy = round((true_negative + true_positive) / len(testSet) if len(testSet) != 0 else float(0), 4)
        recall = round(true_positive / (true_positive + false_negative)
                       if true_positive + false_negative != 0 else 0.0, 4)
        precision = round(true_positive / (false_positive + false_positive)
                          if true_positive + false_positive != 0 else 0.0, 4)
        f1score = round(2 * (precision * recall) / (precision + recall) if precision + recall != 0 else float(0), 4)
        return accuracy, recall, precision, f1score

    def check_if_graph_converged(self, nodes):
        total_hidden_nodes, converged_nodes = 0, 0
        for node in nodes:
            total_hidden_nodes += 1
            new_msg = self.graph.nodes[node]['msg_sum']
            old_msg = self.prev_graph.nodes[node]['msg_sum']
            k = 0
            for i in range(2):
                d = abs(new_msg[i] - old_msg[i])
                if d < 0.00001:
                    k += 1
            if k == 2:
                # print(f'{converged_nodes}, {node},', self.graph.nodes[node]['msg_nbr'])
                converged_nodes += 1
        return converged_nodes


class File:
    def __init__(self, name):
        self.name = name
        self.data = None

    def get_data(self):
        self.read_file_from_file()
        return self.data

    def store_data(self, data):
        self.data = data
        self.write_data_to_file()

    def read_file_from_file(self):
        with open(self.name, 'rb') as f:
            self.data = pickle.load(f)

    def write_data_to_file(self):
        with open(self.name, 'wb') as f:
            pickle.dump(self.data, f)


class Folds:
    def __init__(self, num):
        self.folds = num
        self.kf = None
        self.data = []

    def get_data(self):
        return self.data

    def init_KFold(self, data):
        self.kf = KFold(n_splits=self.folds, shuffle=True)
        self.split_data_to_KFolds(data)

    def split_data_to_KFolds(self, data):
        for i, (train, test) in enumerate(self.kf.split(data)):
            training_set = set(np.array(data)[train])
            test_set = set(np.array(data)[test])
            self.data.append({'train': training_set, 'test': test_set})
            print(f'Fold {i} init, train : test = {len(training_set)} : {len(test_set)}')

    def print_dataset_ratio(self, urls):
        for i, d in enumerate(self.data):
            train_set, test_set = d['train'], d['test']
            # train_set, test_set = self.data[-1]['train'], self.data[-1]['test']
            train_benign, train_malicious, test_benign, test_malicious = 0, 0, 0, 0
            for url in train_set:
                label = urls[url]
                if label == 0:
                    train_benign += 1
                else:
                    train_malicious += 1
            for url in test_set:
                label = urls[url]
                if label == 0:
                    test_benign += 1
                else:
                    test_malicious += 1
            print(f'Fold {i}')
            print(f'Train benign : malicious = {train_benign} : {train_malicious}')
            print(f'Test benign : malicious = {test_benign} : {test_malicious}')
