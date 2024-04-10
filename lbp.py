import sys
import time
import multiprocessing
from module.fileModule import File
from module.foldModule import Folds
from module.workerModule import Worker

N_FOLDS = 5
DATA_SUFFIX = "final"
EMBEDDING_PREFIX = "word2vec"
TYPE_SIMILARITY = "cos"
TYPE_EDGE_POTENTIAL = "t1"
DELETE_CYCLE = True
SIMILARITY_THRESHOLD1 = 0.5
SIMILARITY_THRESHOLD2 = 0.5


def main():
    global DATA_SUFFIX, EMBEDDING_PREFIX, TYPE_SIMILARITY, TYPE_EDGE_POTENTIAL, SIMILARITY_THRESHOLD1, SIMILARITY_THRESHOLD2, DELETE_CYCLE
    if len(sys.argv) > 5:
        DATA_SUFFIX = sys.argv[1]
        EMBEDDING_PREFIX = sys.argv[2]
        TYPE_SIMILARITY = sys.argv[3]
        TYPE_EDGE_POTENTIAL = sys.argv[4]
        DELETE_CYCLE = sys.argv[5]
        if len(sys.argv) == 8:
            SIMILARITY_THRESHOLD1 = float(sys.argv[6])
            SIMILARITY_THRESHOLD2 = float(sys.argv[7])
    print(DATA_SUFFIX, EMBEDDING_PREFIX, TYPE_SIMILARITY, TYPE_EDGE_POTENTIAL, SIMILARITY_THRESHOLD1,
          SIMILARITY_THRESHOLD2, DELETE_CYCLE)

    graph, url_labels, similarities = load_data()

    return crossValidation(graph, url_labels, similarities)


def load_data():
    graph_file = File(f'./data/graphs/graph_{DATA_SUFFIX}.pickle')
    url_file = File(f'./data/urls/url_{DATA_SUFFIX}.pickle')
    simFile = File(f'./data/similarity/word2vec_{TYPE_SIMILARITY}_similarity.pickle')
    G = graph_file.get_data()
    url_labels = url_file.get_data()
    similarities = simFile.get_data()
    print("Nodes:", G.number_of_nodes(), ", Edges:", G.number_of_edges())
    return G, url_labels, similarities


def crossValidation(graph, url_labels, similarity):

    # test_with_specific_datasets(graph, url_labels, similarity)

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
    # return w.run((TYPE_EDGE_POTENTIAL, SIMILARITY_THRESHOLD1, SIMILARITY_THRESHOLD2, graph, url_labels, similarity,
    #               datasets[0], 0))


def train_model(graph, url_labels, similarity, datasets):
    print('Passing Message ... ')
    start_time = time.perf_counter()
    with multiprocessing.Pool(processes=N_FOLDS) as pool:
        try:
            if DELETE_CYCLE:
                outputs = pool.map(Worker('cycle').run, [(TYPE_EDGE_POTENTIAL, SIMILARITY_THRESHOLD1,
                                                          SIMILARITY_THRESHOLD2, graph, url_labels, similarity, d,
                                                          datasets.index(d)) for d in datasets])
            else:
                outputs = pool.map(Worker('normal').run, [(TYPE_EDGE_POTENTIAL, SIMILARITY_THRESHOLD1,
                                                          SIMILARITY_THRESHOLD2, graph, url_labels, similarity, d,
                                                          datasets.index(d)) for d in datasets])
        finally:
            pool.close()
            pool.join()
    end_time = time.perf_counter()
    print("Total train time {:.4f}".format(end_time - start_time), 'sec')

    return calculate_average_performance(outputs)


def calculate_average_performance(outputs):
    accuracy_sum, precision_sum, recall_sum, f1score_sum = 0, 0, 0, 0
    for o in outputs:
        accuracy_sum += o['accuracy']
        precision_sum += o['precision']
        recall_sum += o['recall']
        f1score_sum += o['f1']
    accuracy = round(accuracy_sum / N_FOLDS, 4)
    precision = round(precision_sum / N_FOLDS, 4)
    recall = round(recall_sum / N_FOLDS, 4)
    f1 = round(f1score_sum / N_FOLDS, 4)

    print("Done.")
    print(
        f'Graph {DATA_SUFFIX}: \n'
        f'Type Embedding - {EMBEDDING_PREFIX} \n'
        f'Type Edge Potential - {TYPE_EDGE_POTENTIAL}\n'
        f'Type Similarity - {TYPE_SIMILARITY}\n')
    if SIMILARITY_THRESHOLD1 and SIMILARITY_THRESHOLD2:
        print(f'Threshold1 - {SIMILARITY_THRESHOLD1}, Threshold2 - {SIMILARITY_THRESHOLD2}')
    print(f'Averaged accuracy, precision, recall, F1 : {accuracy}, {precision}, {recall}, {f1}')
    # return [accuracy, precision, recall, f1]


if __name__ == '__main__':
    main()
