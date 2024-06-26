import copy
import networkx as nx
from module.predictModule import assign_node_predict_label


class Message:
    def __init__(self, graph, edge_type, similarity):
        self.prev_graph = copy.deepcopy(graph)
        self.graph = graph
        self.edgePotentialType = edge_type
        self.sender = None
        self.receiver = None
        self.ths1 = None
        self.ths2 = None
        self.nodesMsgDic = {}
        self.similarity = similarity
        self.receive_node_information = None
        self.init_nodes_message_dictionary()
        self.messageType = None

    def set_thresholds(self, t1, t2):
        if self.edgePotentialType == 'sim' or self.edgePotentialType == 'sim_max':
            self.ths1 = t1
            self.ths2 = t2

    def set_message_type(self, msgType):
        self.messageType = msgType

    def set_send_node_name(self, sender):
        self.sender = sender

    def set_receive_node_name(self, receiver):
        self.receiver = receiver

    def init_nodes_message_dictionary(self):
        for node in self.graph.nodes:
            msg_sum_copy = self.graph.nodes[node]['msg_sum'].copy()
            self.nodesMsgDic[node] = {'msg_sum': msg_sum_copy,
                                      'msg_nbr': {}}
            for nbr in self.graph.nodes[node]['msg_nbr']:
                msg_nbr_copy = self.graph.nodes[node]['msg_nbr'][nbr].copy()
                self.nodesMsgDic[node]['msg_nbr'][nbr] = msg_nbr_copy

    def receive_message(self, receiver):
        self.set_receive_node_name(receiver)
        self.init_receive_node_info()
        for node in self.graph.neighbors(receiver):
            self.set_send_node_name(node)
            msg = self.min_sum()
            self.update_receiver_node_info(msg)
        return {self.receiver: self.receive_node_information}

    def init_receive_node_info(self):
        msg_sum_copy = self.graph.nodes[self.receiver]['msg_sum'].copy()
        self.receive_node_information = {'msg_sum': msg_sum_copy, 'msg_nbr': {}}
        for nbr in self.graph.nodes[self.receiver]['msg_nbr']:
            msg_nbr_copy = self.graph.nodes[self.receiver]['msg_nbr'][nbr].copy()
            self.receive_node_information['msg_nbr'][nbr] = msg_nbr_copy

    def update_receiver_node_info(self, msg):
        for i in range(2):
            self.receive_node_information['msg_sum'][i] = (
                round(self.receive_node_information['msg_sum'][i]
                      + (-self.receive_node_information['msg_nbr'][self.sender][i])
                      + msg[i],
                      10))
            self.receive_node_information['msg_nbr'][self.sender][i] = round(msg[i], 10)

    def update_nodes_message_dictionary(self, msg):
        for i in range(2):
            self.nodesMsgDic[self.receiver]['msg_sum'][i] = (
                round(self.nodesMsgDic[self.receiver]['msg_sum'][i]
                      + (-self.nodesMsgDic[self.receiver]['msg_nbr'][self.sender][i])
                      + msg[i],
                      10))
            self.nodesMsgDic[self.receiver]['msg_nbr'][self.sender][i] = round(msg[i], 10)

    def update_receive_node_message(self, msg):
        send_node, receive_node = self.graph.nodes[self.sender], self.graph.nodes[self.receiver]
        for i in range(2):
            receive_node['msg_sum'][i] = round(
                receive_node['msg_sum'][i] + (-receive_node['msg_nbr'][self.sender][i]) + msg[i], 10)
            receive_node['msg_nbr'][self.sender][i] = round(msg[i], 10)

    def update_graph_nodes_message(self, nodesMsgDic):
        for nodeItem in nodesMsgDic:
            for node, nodeInfo in nodeItem.items():
                updated_msg_sum_copy = nodeInfo['msg_sum'].copy()
                self.graph.nodes[node]['msg_sum'] = updated_msg_sum_copy
                for nbr in nodeInfo['msg_nbr']:
                    updated_msg_nbr_copy = nodeInfo['msg_nbr'][nbr].copy()
                    self.graph.nodes[node]['msg_nbr'][nbr] = updated_msg_nbr_copy

    def min_sum(self):
        send_prior_benign = self.graph.nodes[self.sender]['prior_probability'][0]
        send_prior_malicious = self.graph.nodes[self.sender]['prior_probability'][1]
        msg = [0] * 2
        sumOfSendNode = self.sums_of_neighbors()

        for assumeLabel in range(2):
            edges = self.calculate_edge_potential(assumeLabel)
            msg_benign = round(1 - send_prior_benign + edges[0] + sumOfSendNode[0], 10)
            msg_phishing = round(1 - send_prior_malicious + edges[1] + sumOfSendNode[1], 10)
            msg[assumeLabel] = min(msg_benign, msg_phishing)
        return msg

    def sums_of_neighbors(self):
        sumOfSendNode = [0, 0]
        if self.messageType == 'normal':
            if self.graph.nodes[self.sender]['label'] == 0.5:
                sumOfSendNode = self.sums_except_receive_node()
        else:
            if self.graph.nodes[self.sender]['label'] == -1:
                sumOfSendNode = self.sums_except_receive_node()
        return sumOfSendNode

    def sums_except_receive_node(self):
        send_node = self.graph.nodes[self.sender]
        sums = [0] * 2
        sums[0] = send_node['msg_sum'][0] - send_node['msg_nbr'][self.receiver][0]
        sums[1] = send_node['msg_sum'][1] - send_node['msg_nbr'][self.receiver][1]
        return [round(s, 10) for s in sums]

    def calculate_edge_potential(self, assumeLabel):
        if self.edgePotentialType == 't1':
            edges = self.edge_potential_epsilon(assumeLabel)
        else:
            edgeTuple = tuple(sorted([self.sender, self.receiver]))
            similarity = self.similarity[edgeTuple]
            edges = self.calculate_edge_potentials_with_specific_type(assumeLabel, similarity)
        edges = [round(e, 10) for e in edges]
        return edges

    def calculate_edge_potentials_with_specific_type(self, assumeLabel, similarity):
        edges = [0.0] * 2
        if self.edgePotentialType == 'sim_only':
            edges = self.edge_potential_similarity_only(assumeLabel, similarity)
        elif self.edgePotentialType == 'sim':
            edges = self.edge_potential_similarity(assumeLabel, similarity)
        elif self.edgePotentialType == 'sim_max':
            edges = self.edge_potential_similarity_max(assumeLabel, similarity)
        return edges

    def edge_potential_epsilon(self, assumeLabel):
        e = 0.0001
        if assumeLabel == 0:
            edges = [0.5 - e, 0.5 + e]
        else:
            edges = [0.5 + e, 0.5 - e]
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

    # predict hiddenNodes label, if one of its neighbors is known
    def assignNodeLabels(self, hiddenNodes, classify_threshold):
        connected_components = list(nx.connected_components(self.graph))
        for nodes in connected_components:
            status = False
            for node in nodes:
                if node not in hiddenNodes:
                    status = True
                    break
            if status:
                for node in nodes:
                    if node in hiddenNodes:
                        predict_label = assign_node_predict_label(self.graph.nodes[node], classify_threshold)
                        self.graph.nodes[node]['predict_label'] = predict_label
                        if predict_label == 0:
                            self.graph.nodes[node]['prior_probability'] = [1, 0]
                        else:
                            self.graph.nodes[node]['prior_probability'] = [0, 1]

    def count_converged_nodes(self, nodes):
        converged_nodes = 0
        for node in nodes:
            new_msg = self.graph.nodes[node]['msg_sum']
            old_msg = self.prev_graph.nodes[node]['msg_sum']
            k = 0
            for i in range(2):
                d = abs(new_msg[i] - old_msg[i])
                if d < 0.001:
                    k += 1
            if k == 2:
                converged_nodes += 1
        return converged_nodes
