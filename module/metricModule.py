class Metric:
    def __init__(self, graph, data, urls):
        self.graph = graph
        self.data = data
        self.urls = urls
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1Score = 0

    def set_accuracy(self):
        self.accuracy = round((self.true_positive + self.true_negative) / len(self.data['test'])
                              if len(self.data['test']) != 0 else float(0), 4)

    def set_precision(self):
        self.precision = round(self.true_positive / (self.true_positive + self.false_positive)
                               if (self.true_positive + self.false_positive) != 0 else float(0), 4)

    def set_recall(self):
        self.recall = round(self.true_positive / (self.true_positive + self.false_negative)
                            if (self.true_positive + self.false_negative) != 0 else float(0), 4)

    def set_f1Score(self):
        self.f1Score = round(2 * (self.precision * self.recall) / (self.precision + self.recall)
                             if self.precision + self.recall != 0 else float(0), 4)

    def get_metrics(self):
        self.count_predict_result()
        self.set_accuracy()
        self.set_recall()
        self.set_precision()
        self.set_f1Score()
        return {'accuracy': self.accuracy, 'recall': self.recall, 'precision': self.precision, 'f1': self.f1Score}

    def count_predict_result(self):
        test_right, test_wrong = 0, 0
        for url in self.data['test']:
            actual_label = self.urls[url]
            predict_label = self.graph.nodes[url]['predict_label']
            if actual_label == predict_label:
                test_right += 1
                if predict_label == 0:
                    self.true_positive += 1
                else:
                    self.true_negative += 1
            else:
                test_wrong += 1
                if predict_label == 0:
                    self.false_positive += 1
                else:
                    self.false_negative += 1
