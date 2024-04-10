import copy
import networkx as nx


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
        self.init_nodes_message_dictionary()

    def set_thresholds(self, t1, t2):
        if self.edgePotentialType == 'sim' or self.edgePotentialType == 'sim_max':
            self.ths1 = t1
            self.ths2 = t2

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

    def init_nodes_message_dictionary_cycle(self, data):
        for node in data:
            msg_sum_copy = self.graph.nodes[node]['msg_sum'].copy()
            self.nodesMsgDic[node] = {'msg_sum': msg_sum_copy,
                                      'msg_nbr': {}}
            for nbr in self.graph.nodes[node]['msg_nbr']:
                msg_nbr_copy = self.graph.nodes[node]['msg_nbr'][nbr].copy()
                self.nodesMsgDic[node]['msg_nbr'][nbr] = msg_nbr_copy

    def receive_message(self, receiver):
        self.set_receive_node_name(receiver)
        for node in self.graph.neighbors(receiver):
            self.set_send_node_name(node)
            msg = self.min_sum('cycle')
            self.update_nodes_message_dictionary(msg)

    def send_message(self, sender):
        self.set_send_node_name(sender)
        for nbr in self.graph.neighbors(self.sender):
            if self.graph.nodes[nbr]["label"] == 0.5:
                self.set_receive_node_name(nbr)
                msg = self.min_sum('normal')
                self.update_nodes_message_dictionary(msg)

    def send_message_to_cycle_neighbors(self):
        for nbr in self.graph.neighbors(self.sender):
            if self.graph.nodes[nbr]["predict_label"] == -1:
                self.set_receive_node_name(nbr)
                msg = self.min_sum('normal')
                self.update_nodes_message_dictionary(msg)

    def send_message_to_neighbors(self):
        for nbr in self.graph.neighbors(self.sender):
            if self.graph.nodes[nbr]["label"] == 0.5:
                self.set_receive_node_name(nbr)
                msg = self.min_sum('normal')
                self.update_nodes_message_dictionary(msg)

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

    def update_graph_nodes_message(self):
        for node in self.nodesMsgDic:
            updated_msg_sum_copy = self.nodesMsgDic[node]['msg_sum'].copy()
            self.graph.nodes[node]['msg_sum'] = updated_msg_sum_copy
            for nbr in self.nodesMsgDic[node]['msg_nbr']:
                updated_msg_nbr_copy = self.nodesMsgDic[node]['msg_nbr'][nbr].copy()
                self.graph.nodes[node]['msg_nbr'][nbr] = updated_msg_nbr_copy

    def min_sum(self, messageType):
        send_prior_benign = self.graph.nodes[self.sender]['prior_probability'][0]
        send_prior_malicious = self.graph.nodes[self.sender]['prior_probability'][1]
        msg = [0] * 2
        sumOfSendNode = self.sums_of_neighbors(messageType)

        for assumeLabel in range(2):
            edges = self.calculate_edge_potential(assumeLabel)
            msg_benign = round(1 - send_prior_benign + edges[0] + sumOfSendNode[0], 10)
            msg_phishing = round(1 - send_prior_malicious + edges[1] + sumOfSendNode[1], 10)
            msg[assumeLabel] = min(msg_benign, msg_phishing)
        return msg

    def sums_of_neighbors(self, messageType):
        sumOfSendNode = [0, 0]
        if messageType == 'normal':
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
            edges = self.switch_edge_potential_type(assumeLabel, similarity)
        edges = [round(e, 10) for e in edges]
        return edges

    def switch_edge_potential_type(self, assumeLabel, similarity):
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
            # edges = [0.5 + e, 0.5 - e]
            edges = [0.5 - e, 0.5 + e]
        else:
            # edges = [0.5 - e, 0.5 + e]
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

    # assign label to nodes which has known neighbor
    # TODO assign node label if it receive msg from more than half neighbors
    # if node has knownlabel, predict hiddenNodes label
    def assignNodeLabels(self, hiddenNodes):
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
                        if self.receivedFromMajorityNeighbors(node, -1):
                            self.predict_nodeLabel(node)

    # predict node label
    def predict_nodeLabel(self, node):
        cost_benign = 1 - self.graph.nodes[node]['prior_probability'][0] + self.graph.nodes[node]['msg_sum'][0]
        cost_phish = 1 - self.graph.nodes[node]['prior_probability'][1] + self.graph.nodes[node]['msg_sum'][1]
        if cost_benign < cost_phish:
            self.graph.nodes[node]["predict_label"] = 0
            self.graph.nodes[node]['prior_probability'] = [1, 0]
        else:
            self.graph.nodes[node]["predict_label"] = 1
            self.graph.nodes[node]['prior_probability'] = [0, 1]

    def convergedNodes(self, nodes):
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

    def receivedFromMajorityNeighbors(self, node, percent):
        count = 0
        totalNeighbors = len(self.graph.nodes[node]['msg_nbr'])
        for neighbor, message in self.graph.nodes[node]['msg_nbr'].items():
            if message != [0, 0]:
                count += 1
        if count / totalNeighbors * 100 > percent:
            return True
        return False
