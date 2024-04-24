import pickle
import networkx as nx
import pandas as pd
from module.predictModule import assign_node_predict_label


class GraphNode:
    def __init__(self, graph, dataSuffix):
        self.graph = graph
        self.totalEdges = graph.number_of_edges()
        self.dataSuffix = dataSuffix
        self.train = None
        self.test = None
        self.url_labels = None
        self.classify_threshold = None
        self.deletedEdges = {}
        self.cycles = []
        self.similarity = []

    def get_graph(self):
        return self.graph

    def set_classify_threshold(self, threshold):
        self.classify_threshold = threshold

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

    def init_train_node(self):
        for node in self.train:
            label = self.url_labels[node]
            self.graph.nodes[node]['label'] = label
            self.graph.nodes[node]['predict_label'] = label
            if label == 1:
                self.graph.nodes[node]['prior_probability'] = [0, 1]
            elif label == 0:
                self.graph.nodes[node]['prior_probability'] = [1, 0]
            else:
                raise NameError('wrong label detected')

    def add_test_node_probability(self):
        probability_df = pd.read_csv(
            f"./data/probability/benign_probability_RandomForestClassifier_{self.dataSuffix}.csv")
        # probability_df = pd.read_csv(f"./data/probability/benign_probability_RandomForestClassifier_final.csv")

        benign_probability = {}
        for index, row in probability_df.iterrows():
            url = row['url']
            proba_benign = row['Probability (benign)']
            proba_malicious = row['Probability (malicious)']
            benign_probability[url] = [proba_benign, proba_malicious]
        for node in self.test:
            self.graph.nodes[node]['prior_probability'] = benign_probability[node]

    def assign_predicted_label(self):
        for node in self.graph.nodes:
            if self.graph.nodes[node]['predict_label'] == -1:
                predict_label = assign_node_predict_label(self.graph.nodes[node], self.classify_threshold)
                self.graph.nodes[node]['predict_label'] = predict_label
                if predict_label == 0:
                    self.graph.nodes[node]['prior_probability'] = [1, 0]
                else:
                    self.graph.nodes[node]['prior_probability'] = [0, 1]

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
        totalDeletedEdges = 0
        for key, edges in self.deletedEdges.items():
            totalDeletedEdges += len(edges)
        print(f'current edges: {self.graph.number_of_edges()}, deleted edges: {totalDeletedEdges}')
        if self.graph.number_of_edges() + totalDeletedEdges != self.totalEdges:
            print("THIS WRONG")
        return len(self.deletedEdges) != 0

    def add_deleted_cycles(self):
        # 1. add cycles with all known nodes
        addKnownCycles = self.add_cycle_without_unknown_nodes()
        # 2. add cycles with at least one known nodes
        addWithOneKnownNodeCycle = self.add_cycle_with_at_least_one_known_node()
        # 3. remove cycles that only contain unknown nodes
        if addKnownCycles + addWithOneKnownNodeCycle > 0:
            self.remove_unknown_cycles()

    def remove_unknown_cycles(self):
        hiddenNodes = [e for e in list(self.graph.nodes()) if self.graph.nodes[e]['predict_label'] == -1]
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

    def add_cycle_without_unknown_nodes(self):
        addCycles = 0
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
