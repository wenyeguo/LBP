import pickle
import networkx as nx


class GraphNode:
    def __init__(self, graph):
        self.graph = graph
        self.train = None
        self.test = None
        self.url_labels = None
        self.deletedEdges = {}
        self.cycles = []
        self.similarity = []

    def get_graph(self):
        return self.graph

    def init_nodes(self, data, urls):
        self.train = data['train']
        self.test = data['test']
        self.url_labels = urls
        self.init_all_graph_nodes()
        self.init_train_node()
        return self.graph

    def identify_cycles_in_graph(self):
        cycles = list(nx.simple_cycles(self.graph))
        print()

    # 'prior_probability' == probability of node with benign label and malicious label
    def init_all_graph_nodes(self):
        nodes_with_known_neighbors = 0
        for node in self.graph.nodes():
            self.graph.nodes[node]["label"] = 0.5
            self.graph.nodes[node]["predict_label"] = -1
            self.graph.nodes[node]["prior_probability"] = [0.5, 0.5]
            self.graph.nodes[node]["msg_sum"] = [0, 0]
            self.graph.nodes[node]["msg_nbr"] = {}
            for nbr in list(self.graph.neighbors(node)):
                self.graph.nodes[node]["msg_nbr"][nbr] = [0, 0]

    def check_if_all_neighbors_known(self, node):
        for nbr, msg in self.graph.nodes[node]['msg_nbr'].items():
            if nbr not in self.test:
                return False
        return True

    def init_train_node(self):
        for node in self.train:
            label = self.url_labels[node]
            self.graph.nodes[node]['label'] = label
            if label == 1:
                self.graph.nodes[node]['prior_probability'] = [0, 1]
            elif label == 0:
                self.graph.nodes[node]['prior_probability'] = [1, 0]
            else:
                raise NameError('wrong label detected')

    def init_test_node_probability(self):
        with open('./baseline/benign_probability.pickle', 'rb') as f:
            benign_probability = pickle.load(f)
        for node in self.test:
            proba = benign_probability[node]
            self.graph.nodes[node]['prior_probability'] = [round(proba, 4), round(1 - proba, 4)]
        # for node in self.graph.nodes:
        #     cost = self.graph.nodes[node]['prior_probability']

    def assign_predicted_label(self):
        for node in self.graph.nodes:
            if self.graph.nodes[node]['predict_label'] == -1:
                cost_benign = (1 - self.graph.nodes[node]['prior_probability'][0]) + self.graph.nodes[node]['msg_sum'][
                    0]
                cost_phish = (1 - self.graph.nodes[node]['prior_probability'][1]) + self.graph.nodes[node]['msg_sum'][1]
                if cost_benign < cost_phish:
                    self.graph.nodes[node]["predict_label"] = 0
                else:
                    self.graph.nodes[node]["predict_label"] = 1

    def assign_predicted_label_to_test_urls(self):
        for url in self.test:
            cost_benign = (1 - self.graph.nodes[url]['prior_probability'][0]) + self.graph.nodes[url]['msg_sum'][0]
            cost_phish = (1 - self.graph.nodes[url]['prior_probability'][1]) + self.graph.nodes[url]['msg_sum'][1]
            if cost_benign < cost_phish:
                self.graph.nodes[url]["predict_label"] = 0
            else:
                self.graph.nodes[url]["predict_label"] = 1

    def calculate_performance(self):
        TP_B_B, TN_M_M, FP_B_M, FN_M_B = self.compare_hidden_nodes_predicted_label_with_actual_label()
        accuracy = round((TN_M_M + TP_B_B) / len(self.test) if len(self.test) != 0 else float(0), 4)
        recall = round(TP_B_B / (TP_B_B + FN_M_B) if TP_B_B + FN_M_B != 0 else float(0), 4)
        precision = round(TP_B_B / (TP_B_B + FP_B_M) if TP_B_B + FP_B_M != 0 else float(0), 4)
        f1score = round(2 * (precision * recall) / (precision + recall) if precision + recall != 0 else float(0), 4)
        return accuracy, recall, precision, f1score

    def compare_hidden_nodes_predicted_label_with_actual_label(self):
        test_right, test_wrong = 0, 0
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

    def remove_cycles(self):
        hiddenNodes = [e for e in list(self.graph.nodes()) if self.graph.nodes[e]['label'] == 0.5]
        hiddenGraph = self.create_subgraph_only_contain_hidden_nodes(hiddenNodes)
        edgesDeleted = {}
        while True:
            graphCycles = nx.cycle_basis(hiddenGraph)
            if graphCycles:
                hiddenGraph, edgesDeleted = self.delete_cycles(hiddenGraph, graphCycles, edgesDeleted)
            else:
                break
        self.remove_cycles_from_original_graph(edgesDeleted)
        return self.graph

    def remove_cycles_from_original_graph(self, edgesDeleted):
        for cycle, edges in edgesDeleted.items():
            for edge in edges:
                self.graph.remove_edge(edge[0], edge[1])
        return self.graph

    def create_subgraph_only_contain_hidden_nodes(self, nodes):
        return nx.Graph(self.graph.subgraph(nodes))

    def delete_cycles(self, graph, graphCycles, edgesDeleted):
        for idx, cycle in enumerate(graphCycles):
            tupleCycle = tuple(sorted(cycle))
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if graph.has_edge(u, v):
                    edgesDeleted[tupleCycle] = edgesDeleted.get(tupleCycle, [])
                    edgesDeleted[tupleCycle].append([u, v])
                    self.deletedEdges[tupleCycle] = self.deletedEdges.get(tupleCycle, [])
                    self.deletedEdges[tupleCycle].append([u, v])
                    graph.remove_edge(u, v)
        return graph, edgesDeleted

    def has_deleted_cycles(self):
        return len(self.deletedEdges) != 0

    def add_deleted_cycles(self):
        # 1. add all known nodes
        addKnownCycles = self.add_cycle_without_unknown_nodes()
        # print(f'Add {addKnownCycles} cycles with only known nodes')

        addWithOneNodeCycle = self.add_cycle_with_at_least_one_known_node()
        # print(f'Add {addWithOneNodeCycle} cycles with at least one known node')
        # TODO => if hidden nodes graph has cycle, if has delete cycle
        if addKnownCycles + addWithOneNodeCycle > 0:
            self.remove_unknown_cycles()

    def remove_unknown_cycles(self):
        hiddenNodes = [e for e in list(self.graph.nodes()) if self.graph.nodes[e]['predict_label'] == -1]
        # TODO not contain all nodes, why??
        edgesDeleted = {}
        graph = self.create_subgraph_only_contain_hidden_nodes(hiddenNodes)
        while True:
            currentCycles = nx.cycle_basis(graph)
            if currentCycles:
                graph, edgesDeleted = self.delete_cycles(graph, currentCycles, edgesDeleted)
            else:
                break
        if edgesDeleted:
            self.remove_cycles_from_original_graph(edgesDeleted)
        return self.graph

    def add_cycle_with_at_least_one_known_node(self):
        count = 0
        cycleToDelete = []
        for cycle, edges in self.deletedEdges.items():
            if not self.allUnknownNodes(cycle):
                count += 1
                for edge in edges:
                    self.graph.add_edge(edge[0], edge[1])
                cycleToDelete.append(cycle)
        for cycle in cycleToDelete:
            del self.deletedEdges[cycle]
        return count

    def has_unknown_cycles(self, idx):
        hiddenNodes = [e for e in list(self.graph.nodes()) if self.graph.nodes[e]['predict_label'] == -1]
        c = nx.cycle_basis(self.graph)
        cycle = Cycle(self.graph)
        currentCycles = cycle.get_cycles(hiddenNodes)
        if currentCycles:
            for edge in self.deletedEdges[idx]:
                self.graph.remove_edge(edge[0], edge[1])
            return True
        return False

    def add_cycle_without_unknown_nodes(self):
        addCycles = 0
        # TODO dic changed during iteration, cycleToDelete = []
        cycleToDelete = []
        for cycle, edges in self.deletedEdges.items():
            if self.allKnownNodes(cycle):
                addCycles += 1
                for edge in edges:
                    self.graph.add_edge(edge[0], edge[1])
                cycleToDelete.append(cycle)
        for cycle in cycleToDelete:
            del self.deletedEdges[cycle]
        return addCycles

    def add_one_cycle_with_unknown_nodes(self):
        addedUnknownNodes = []
        # add 1 cycle (the 1st has known label cycle) each time
        for idx, cycle in enumerate(self.cycles):
            if self.cycles[idx]:
                if not self.allUnknownNodes(cycle):
                    # add edges back and remove cycle from list
                    for node in cycle:
                        if self.graph.nodes[node]['predict_label'] == -1:
                            addedUnknownNodes.append(node)
                    for edge in self.deletedEdges[idx]:
                        self.graph.add_edge(edge[0], edge[1])
                    self.cycles[idx] = []
                    break
        return addedUnknownNodes

    def allKnownNodes(self, cycle):
        for node in cycle:
            if self.graph.nodes[node]['predict_label'] == -1:
                return False
        return True

    def allUnknownNodes(self, cycle):
        for node in cycle:
            if self.graph.nodes[node]['predict_label'] != -1:
                return False
        return True
