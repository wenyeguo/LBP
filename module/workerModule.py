from module.graphNodeModule import GraphNode
from module.messageModule import Message
from module.metricModule import Metric


class Worker:
    def __init__(self, t):
        self.workerType = t
        self.dataSuffix = None
        self.similarity = None
        self.urls = None
        self.graph = None
        self.data = None
        self.processId = None
        self.threshold1 = None
        self.threshold2 = None
        self.edgePotentialType = None
        self.classify_threshold = None
        self.add_test_probability = None

    def run(self, args):
        (self.dataSuffix, self.add_test_probability, self.edgePotentialType, self.classify_threshold, self.threshold1, self.threshold2,
         self.graph, self.urls, self.similarity,
         self.data, self.processId) = args
        if self.workerType == 'cycle':
            self.worker_cycle()
        elif self.workerType == 'normal':
            self.worker_normal()
        metrics = Metric(self.graph, self.data, self.urls)
        return metrics.get_ROC_data()

    def worker_normal(self):
        graph_node = GraphNode(self.graph, self.dataSuffix)
        graph_node.set_classify_threshold(self.classify_threshold)
        graph_node.init_nodes(self.data, self.urls)
        if self.add_test_probability:
            graph_node.add_test_node_probability()

        times = 6
        self.messagePassing('normal', times)

    def worker_cycle(self):
        """delete cycles, passing message, predict labels"""
        # init graph nodes, add metadata (label, predict_label ...)
        graph_node = GraphNode(self.graph, self.dataSuffix)
        graph_node.set_classify_threshold(self.classify_threshold)
        graph_node.init_nodes(self.data, self.urls)
        if self.add_test_probability:
            graph_node.add_test_node_probability()

        self.graph = graph_node.remove_cycles()
        self.messagePassing('normal', -1)

        withoutPredictionNodesNums = self.withoutPredictionNodes()
        print(f'Process {self.processId}, after remove all cycles, not predicted nodes', withoutPredictionNodesNums)

        add_all_cycles_back = self.add_cycle_to_worker(graph_node)
        if not add_all_cycles_back:
            self.print_convergence_result()
        graph_node.assign_predicted_label()

    def print_convergence_result(self):
        unpredictedHiddenNodes, unpredictedTestUrls = 0, 0
        total_hidden_nodes = self.graph.number_of_nodes() - len(self.data["train"])
        for node in self.graph.nodes:
            if self.graph.nodes[node]['predict_label'] == -1:
                if node in self.data['test']:
                    unpredictedTestUrls += 1
                unpredictedHiddenNodes += 1
        percentTestURL = unpredictedTestUrls / len(self.data["test"]) * 100
        percentHiddenNodes = unpredictedHiddenNodes / total_hidden_nodes * 100
        print(
            f'Process {self.processId}: \n '
            f'% of Unpredicted Test URLs: {round(percentTestURL, 2)}\n '
            f'% of Unpredicted Hidden Nodes: {round(percentHiddenNodes, 2)}')

    def withoutPredictionNodes(self):
        count = 0
        for node in self.graph.nodes:
            if self.graph.nodes[node]['predict_label'] == -1:
                count += 1
        return count

    def add_cycle_to_worker(self, graph_node):
        # remain unpredicted labels are node whose neighbors are all unknown currently
        while graph_node.has_deleted_cycles():
            prevEdges = self.graph.number_of_edges()
            graph_node.add_deleted_cycles()
            currentEdges = self.graph.number_of_edges()
            print('----------------------------------------------------------------')
            print(f'Process {self.processId}, add {currentEdges - prevEdges} edges')
            print('----------------------------------------------------------------')

            if currentEdges > prevEdges:
                self.messagePassing('cycle', -1)
            else:
                print('----------------------------------------------------------------')
                print(f'Process {self.processId} stop adding cycles, since no known cycle exists')
                print('----------------------------------------------------------------')

                return False
        print('----------------------------------------------------------------')
        print(f'Process {self.processId}, Added All Edges Back, Current Edges {self.graph.number_of_edges()}')
        print('----------------------------------------------------------------')

        return True

    def messagePassing(self, passType, iterationTimes):
        iterations = 0
        if passType == 'normal':
            unknownNodes = [e for e in list(self.graph.nodes()) if self.graph.nodes[e]['label'] == 0.5]
        else:
            unknownNodes = [e for e in list(self.graph.nodes()) if self.graph.nodes[e]['predict_label'] == -1]

        while True:
            message = Message(self.graph, self.edgePotentialType, self.similarity)
            message.set_thresholds(self.threshold1, self.threshold2)
            message.set_message_type(passType)
            outputs = [message.receive_message(node) for node in unknownNodes]
            message.update_graph_nodes_message(outputs)
            current_converged_nodes = message.count_converged_nodes(unknownNodes)

            print(f'Process {self.processId} Iteration {iterations}: '
                  f'total hidden nodes {len(unknownNodes)}, converged nodes {current_converged_nodes}')

            if current_converged_nodes == len(unknownNodes):
                message.assignNodeLabels(unknownNodes, self.classify_threshold)
                break
            elif iterationTimes != -1 and iterations == iterationTimes:
                message.assignNodeLabels(unknownNodes, self.classify_threshold)
                break

            iterations += 1
