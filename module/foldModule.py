import numpy as np
from sklearn.model_selection import KFold


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
