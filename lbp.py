import sys
import time
import multiprocessing
from module.fileModule import File
from module.foldModule import Folds
from module.workerModule import Worker

N_FOLDS = 5
DATA_SUFFIX = "final"
EMBEDDING_PREFIX = "word2vec"
TYPE_SIMILARITY = "rbf"
TYPE_EDGE_POTENTIAL = "sim_only"
DELETE_CYCLE = True
ADD_PRIOR_PROBABILITY = True
CLASSIFY_THRESHOLD = 0.5
SIMILARITY_THRESHOLD1 = 0.6
SIMILARITY_THRESHOLD2 = 0.4


def main():
    global DATA_SUFFIX, EMBEDDING_PREFIX, TYPE_SIMILARITY, TYPE_EDGE_POTENTIAL, SIMILARITY_THRESHOLD1, \
        CLASSIFY_THRESHOLD, SIMILARITY_THRESHOLD2, DELETE_CYCLE, ADD_PRIOR_PROBABILITY
    if len(sys.argv) > 7:
        DATA_SUFFIX = sys.argv[1]
        EMBEDDING_PREFIX = sys.argv[2]
        TYPE_SIMILARITY = sys.argv[3]
        TYPE_EDGE_POTENTIAL = sys.argv[4]
        DELETE_CYCLE = sys.argv[5]
        ADD_PRIOR_PROBABILITY = sys.argv[6]
        CLASSIFY_THRESHOLD = float(sys.argv[7])
        if len(sys.argv) == 9:
            SIMILARITY_THRESHOLD1 = float(sys.argv[8])
            SIMILARITY_THRESHOLD2 = float(sys.argv[9])
    print(DATA_SUFFIX, EMBEDDING_PREFIX, TYPE_SIMILARITY, TYPE_EDGE_POTENTIAL, DELETE_CYCLE, ADD_PRIOR_PROBABILITY,
          CLASSIFY_THRESHOLD, SIMILARITY_THRESHOLD1, SIMILARITY_THRESHOLD2)

    crossValidation()
    # find_edge_potential_threshold()
    # find_classify_threshold()


def find_edge_potential_threshold():
    global SIMILARITY_THRESHOLD1, SIMILARITY_THRESHOLD2
    t1 = [n / 10 for n in range(11)]
    t2 = [n / 10 for n in range(11)]
    results = {}
    for i in t1:
        for j in t2:
            SIMILARITY_THRESHOLD1, SIMILARITY_THRESHOLD2 = i, j
            rs = crossValidation()
            key = f'{i}:{j}'
            results[key] = rs
    fileT = File(f'graph_{DATA_SUFFIX}_{TYPE_SIMILARITY}_{TYPE_EDGE_POTENTIAL}_result.csv')
    fileT.store_threshold_result_into_csv_file(results)


def find_classify_threshold():
    global CLASSIFY_THRESHOLD
    performance = [[] for _ in range(11)]
    for i in range(11):
        CLASSIFY_THRESHOLD = i / 10
        rs = crossValidation()
        performance[i] = rs
    print(f'DIFFERENT THRESHOLD RESULT: accuracy, precision, recall, F1, TPR, FPR')
    for p in performance:
        print(p)


def load_data():
    graph_file = File(f'./data/graphs/graph_{DATA_SUFFIX}.pickle')
    url_file = File(f'./data/urls/url_{DATA_SUFFIX}.pickle')
    simFile = File(f'./data/similarity/word2vec_{TYPE_SIMILARITY}_similarity.pickle')
    G = graph_file.get_data()
    url_labels = url_file.get_data()
    similarities = simFile.get_data()
    print("Nodes:", G.number_of_nodes(), ", Edges:", G.number_of_edges())
    return G, url_labels, similarities


def crossValidation():
    # return test_with_specific_datasets(graph, url_labels, similarity)
    graph, url_labels, similarity = load_data()
    kf = Folds(N_FOLDS)
    kf.init_KFold(list(url_labels.keys()))
    kf.print_dataset_ratio(url_labels)
    datasets = kf.get_data()
    return train_model(graph, url_labels, similarity, datasets)


def test_with_specific_datasets(graph, url_labels, similarity):
    dataFile = File(f'./dataList/{len(url_labels)}_data.pickle')
    datasets = dataFile.get_data()
    return train_model(graph, url_labels, similarity, datasets)

    # test using 1 process
    # w = Worker('cycle')
    # return w.run((TYPE_EDGE_POTENTIAL, CLASSIFY_THRESHOLD, SIMILARITY_THRESHOLD1, SIMILARITY_THRESHOLD2, graph, url_labels, similarity,
    #               datasets[0], 0))


def train_model(graph, url_labels, similarity, datasets):
    print('Passing Message ... ')
    start_time = time.perf_counter()
    with multiprocessing.Pool(processes=N_FOLDS) as pool:
        try:
            if DELETE_CYCLE:
                outputs = pool.map(Worker('cycle').run, [(
                    ADD_PRIOR_PROBABILITY, TYPE_EDGE_POTENTIAL, CLASSIFY_THRESHOLD,
                    SIMILARITY_THRESHOLD1, SIMILARITY_THRESHOLD2, graph,
                    url_labels, similarity, d, datasets.index(d)) for d in
                    datasets])
            else:
                outputs = pool.map(Worker('normal').run, [(
                    ADD_PRIOR_PROBABILITY, TYPE_EDGE_POTENTIAL, CLASSIFY_THRESHOLD,
                    SIMILARITY_THRESHOLD1, SIMILARITY_THRESHOLD2, graph,
                    url_labels, similarity, d, datasets.index(d)) for d in
                    datasets])
        finally:
            pool.close()
            pool.join()
    end_time = time.perf_counter()
    print("Total train time {:.4f}".format(end_time - start_time), 'sec')

    return calculate_average_performance(outputs)


def calculate_average_performance(outputs):
    accuracy_sum, precision_sum, recall_sum, f1score_sum = 0, 0, 0, 0
    TPR_sum, FPR_sum = 0, 0
    for o in outputs:
        accuracy_sum += o['accuracy']
        precision_sum += o['precision']
        recall_sum += o['recall']
        f1score_sum += o['f1']
        TPR_sum += o['TPR']
        FPR_sum += o['FPR']
    accuracy = round(accuracy_sum / N_FOLDS, 4)
    precision = round(precision_sum / N_FOLDS, 4)
    recall = round(recall_sum / N_FOLDS, 4)
    f1 = round(f1score_sum / N_FOLDS, 4)
    tpr = round(TPR_sum / N_FOLDS, 4)
    fpr = round(FPR_sum / N_FOLDS, 4)

    print("Done.")
    print(
        f'Graph {DATA_SUFFIX}: \n'
        f'Type Embedding - {EMBEDDING_PREFIX} \n'
        f'Type Edge Potential - {TYPE_EDGE_POTENTIAL}\n'
        f'Type Similarity - {TYPE_SIMILARITY}\n'
        f'Classify Threshold - {CLASSIFY_THRESHOLD}\n'
        f'Delete Cycles - {DELETE_CYCLE}\n'
        f'With Prior Probability - {ADD_PRIOR_PROBABILITY}'
    )

    if SIMILARITY_THRESHOLD1 and SIMILARITY_THRESHOLD2:
        print(f'Threshold1 - {SIMILARITY_THRESHOLD1}, Threshold2 - {SIMILARITY_THRESHOLD2}')
    print(f'Averaged accuracy, precision, recall, F1, TPR, FPR : {accuracy}, {precision}, {recall}, {f1}, {tpr}, {fpr}')
    return [accuracy, precision, recall, f1, tpr, fpr]


if __name__ == '__main__':
    main()
