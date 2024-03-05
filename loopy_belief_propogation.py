import csv
import multiprocessing
import time
from fileinput import filename
from tqdm import tqdm
from util import Similarity, Message, GraphNode, File, Folds

N_FOLDS = 5


def step(edge_type, t1, t2, graph, training_set, idx):
    status = True
    iteration_num = 1
    nodes_list = [e for e in list(graph.nodes()) if graph.nodes[e]['label'] == 0.5]
    prev_converged_nodes = -1
    while status:
        message = Message(graph)
        if edge_type == 'sim':
            message.set_send_message_type(edge_type)
            message.set_send_message_thresholds(t1, t2)
        nodes_list = nodes_list[::-1]
        # random.shuffle(nodes_list)
        with (tqdm(total=len(nodes_list) + len(training_set), desc=f'Process {idx} Iteration {iteration_num}')
              as progress_bar):
            # if iteration != 1 not go this
            for node in training_set:
                progress_bar.update(1)
                message.send_message(node)

            # send msg from known urls nbr to hidden node
            for node in nodes_list:
                progress_bar.update(1)
                message.send_message(node)
            progress_bar.close()

            current_converged_nodes = message.check_if_graph_converged(nodes_list)
            print(f'Process {idx} Iteration {iteration_num}')
            print(f'converged node previous : current = {prev_converged_nodes} : {current_converged_nodes}')
            if current_converged_nodes == len(nodes_list):
                status = False
                print('Terminate since converged')
            prev_converged_nodes = current_converged_nodes

            iteration_num += 1
            if iteration_num > 5:
                status = False


def worker(args):
    edge_type, t1, t2, G, urls, data, idx = args
    training_set = data['train']
    graph_node = GraphNode(G)
    graph_node.init_nodes(data, urls)

    step(edge_type, t1, t2, G, training_set, idx)

    graph_node.infer_node_labels()
    accuracy, recall, precision, f1 = graph_node.calculate_performance()
    return {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1}


def read_data_from_files(suffix, emb_prefix):
    # load graph
    graph_file = File(f'graph_{suffix}.pickle')
    G = graph_file.get_data()
    # load url_labels, urls
    url_file = File(f'urls_{suffix}.pickle')
    url_labels = url_file.get_data()
    urls = list(url_labels.keys())
    # assign edge similarity according emb file
    emb_file = File(f'{emb_prefix}_embeddings.pickle')
    return G, url_labels, urls, emb_file


def calculate_average_performance(outputs):
    accuracy_sum, precision_sum, recall_sum, f1score_sum = 0, 0, 0, 0
    for o in outputs:
        accuracy_sum += o['accuracy']
        precision_sum += o['precision']
        recall_sum += o['recall']
        f1score_sum += o['f1']
    accuracy, precision, recall, f1 = round(accuracy_sum / N_FOLDS, 4), round(precision_sum / N_FOLDS,4), round(recall_sum / N_FOLDS, 4), round(f1score_sum / N_FOLDS, 4)

    print("Done.")
    print(f'Averaged accuracy, precision, recall, F1 : '
          f'{accuracy}, {precision}, {recall}, {f1}')
    return [accuracy, precision, recall, f1]


def main(t1, t2, sim_type, emb_prefix, edge_type, suffix):
    print(edge_type, t1, t2, sim_type, emb_prefix, suffix)

    G, url_labels, urls, emb_file = read_data_from_files(suffix, emb_prefix)
    print("Nodes:", G.number_of_nodes(), ", Edges:", G.number_of_edges())
    sim = Similarity(G, sim_type, emb_file)
    sim.init_edge_similarity()
    min_sim, max_sim = sim.get_similarity_range()
    print(f'similarity range: [{min_sim}, {max_sim}]')

    kf = Folds(N_FOLDS)
    kf.init_KFold(urls)
    data_list = kf.get_data()

    worker([edge_type, t1, t2, G, url_labels, data_list[0], 0])
    print()
    print('Passing Message ... ')
    start_time = time.perf_counter()
    with multiprocessing.Pool(processes=N_FOLDS) as pool:
        try:
            outputs = pool.map(worker, [(edge_type, t1, t2, G, url_labels, d, data_list.index(d)) for d in data_list])
        finally:
            pool.close()
            pool.join()
    end_time = time.perf_counter()
    print("Total train time {:.4f}".format(end_time - start_time), 'sec')

    return calculate_average_performance(outputs)


def store_result_dic_to_csv_file(filename, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ths+','ths-', 'accuracy', 'recursion', 'precision', 'F1'])
        for key, val in data.items():
            t1, t2 = key.split('-')
            accuracy, precision, recall, f1 = val[0], val[1], val[2], val[3]
            writer.writerow([t1, t2, accuracy, precision, recall, f1])



if __name__ == '__main__':
    s = ['501']
    for suffix in s:
        rs_t1 = main(1, 0.1, 'rbf', 'word2vec', 't1', suffix)
        rs_sim_only = main(1, 0.1, 'rbf', 'word2vec', 'sim_only', suffix)
        print('graph', suffix)
        print('t1 result', rs_t1)
        print('rs_sim_only', rs_sim_only)
    # edge_prefix = 'word2vec'
    # suffix = '24816'
    # result_dic = {}
    # for i in range(8, -1, -1):
    #     for j in range(2, 11):
    #         threshold1, threshold2 = (i / 10), (j / 10)
    #         rs = main(threshold1, threshold2, 'rbf', 'word2vec', edge_prefix, suffix)
    #         key = f'{threshold1}-{threshold2}'
    #         result_dic[key] = rs
    # store_result_dic_to_csv_file(f'graph_{suffix}_{edge_prefix}_result.csv', result_dic)
    #
