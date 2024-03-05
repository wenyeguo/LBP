import copy
import math
import pickle
import sys
import numpy as np
from sklearn.model_selection import KFold


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

    def init_all_graph_nodes(self):
        for node in self.graph.nodes():
            self.graph.nodes[node]["label"] = 0.5
            self.graph.nodes[node]["predict_label"] = -1
            self.graph.nodes[node]["cost"] = [0.5, 0.5]
            self.graph.nodes[node]["msg_sum"] = [0, 0]
            self.graph.nodes[node]["msg_nbr"] = {}
            neighbors = [n for n in list(self.graph.neighbors(node))]
            for nbr in neighbors:
                self.graph.nodes[node]["msg_nbr"][nbr] = [0, 0]

    def init_train_node(self):
        for node in self.train:
            label = self.url_labels[node]
            self.graph.nodes[node]['label'] = label
            if label == 1:
                self.graph.nodes[node]['cost'] = [0.99, 0.01]
            elif label == 0:
                self.graph.nodes[node]['cost'] = [0.01, 0.99]

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
        print('TEST URLS Same Cost', same)

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
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'TP, TN, FP, FN = {TP_B_B}, {TN_M_M}, {FP_B_M}, {FN_M_B}')
        print(f'Accuracy, Recall, Precision, F1 = {accuracy}, {recall}, '
              f'{precision}, {f1score} ')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
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
            v1, v2 = self.emb[edge[0]], self.emb[edge[1]]
            sim = self.calculate_similarity(v1, v2)
            self.min = min(sim, self.min)
            self.max = max(sim, self.max)
            self.graph.edges[edge]['sim'] = sim

    def normalize_vector(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm == 0 else v

    def calculate_similarity(self, v1, v2):
        v1 = self.normalize_vector(v1)
        v2 = self.normalize_vector(v2)
        sim = 0
        if self.type == 'cos':
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        elif self.type == 'rbf':
            distance = np.linalg.norm(v1 - v2)
            sim = np.exp(-distance ** 2 / (2 * 1 ** 2))
        return sim


class Message:
    def __init__(self, graph):
        self.prev_graph = copy.deepcopy(graph)
        self.graph = graph
        self.type = None
        self.send = None
        self.receive = None
        self.ths1 = None
        self.ths2 = None

    def set_send_message_type(self, t):
        self.type = t

    def set_send_message_thresholds(self, t1, t2):
        self.ths1 = t1
        self.ths2 = t2

    def set_send_node_name(self, send):
        self.send = send

    def set_receive_node_name(self, receive):
        self.receive = receive

    def send_message(self, send):
        self.set_send_node_name(send)
        label = self.graph.nodes[send]['label']
        if label == 1 or label == 0:
            self.send_message_from_known_node(label)
        else:
            self.send_message_from_unknown_node(label)

    def send_message_from_known_node(self, label):
        msg = [1, 0] if label == 0 else [0, 1]
        for nbr in self.graph.neighbors(self.send):
            if self.graph.nodes[nbr]["label"] == 0.5:
                self.set_receive_node_name(nbr)
                self.update_receive_node_message(msg)

    def send_message_from_unknown_node(self, label):
        for nbr in self.graph.neighbors(self.send):
            if self.graph.nodes[nbr]["label"] == 0.5:
                self.set_receive_node_name(nbr)
                msg, sums, edge, m = self.min_sum(label)
                self.update_receive_node_message(msg)

                # prev_send_node_msg_sum = self.prev_graph.nodes[self.send]['msg_sum']
                # current_send_node_msg_sum = self.graph.nodes[self.send]['msg_sum']
                # msg, sums, edge, m = self.min_sum(label)
                # self.update_receive_node_message(msg)
                # if prev_send_node_msg_sum != current_send_node_msg_sum:
                #     print()
                #     print('prev send_node', self.prev_graph.nodes[self.send]['msg_sum'])
                #     print('send_node', self.graph.nodes[self.send]['msg_sum'])
                #     print('send_node', self.graph.nodes[self.send]['msg_nbr'])
                #
                # # print(msg)
                #
                # if msg != self.graph.nodes[nbr]['msg_nbr'][self.send]:
                #     self.update_receive_node_message(msg)

    # 1. prev sum, nbrs 2. update receive node 3. check prev and cur
    def update_receive_node_message(self, msg):
        send_node, receive_node = self.graph.nodes[self.send], self.graph.nodes[self.receive]
        for i in range(2):
            receive_node['msg_sum'][i] = round(
                receive_node['msg_sum'][i] + (-receive_node['msg_nbr'][self.send][i]) + msg[i], 10)
            receive_node['msg_nbr'][self.send][i] = round(msg[i], 10)

    def if_send_node_message_not_change(self, prev):
        return prev == self.graph.nodes[self.receive]['msg_nbr'][self.send]

    def min_sum(self, label):
        send_node, receive_node = self.graph.nodes[self.send], self.graph.nodes[self.receive]
        msg = [0] * 2
        sums = self.sums_except_receive_node(send_node, receive_node)
        for i in range(2):
            edges = self.calculate_edge_potential(i)
            msg_benign = round(math.log(1 - label) + edges[0] + sums[0], 10)
            msg_phishing = round(math.log(1 - label) + edges[1] + sums[1], 10)
            msg[i] = min(msg_benign, msg_phishing)
        return msg

    def calculate_edge_potential(self, i):
        e = 0.0001
        edges = [0] * 2
        if self.type is None:
            edges[0] = 0.5 + e if i == 0 else 0.5 - e
            edges[1] = 0.5 + e if i == 1 else 0.5 - e
        elif self.type == 'sim_only':
            sim = self.graph[self.send][self.receive]['sim']
            edges[0] = 1 - sim if i == 0 else sim
            edges[1] = 1 - sim if i == 1 else sim
        elif self.type == 'sim':
            sim = self.graph[self.send][self.receive]['sim']
            # edges[0] = min(self.ths1, 1 - sim) if i == 0 else max(self.ths2, sim)
            # edges[1] = min(self.ths1, 1 - sim) if i == 1 else max(self.ths2, sim)
            edges[0] = max(self.ths1, 1 - sim) if i == 0 else min(self.ths2, sim)
            edges[1] = max(self.ths1, 1 - sim) if i == 1 else min(self.ths2, sim)
        return [round(e, 10) for e in edges]

    def sums_except_receive_node(self, send_node, receive_node):
        return [round(a - b, 10) for a, b in zip(send_node['msg_sum'], send_node['msg_nbr'][self.receive])]

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
            self.data = pickle.load(f)


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



