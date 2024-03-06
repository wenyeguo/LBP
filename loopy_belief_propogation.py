import csv
import multiprocessing
import time
from fileinput import filename
from tqdm import tqdm
from utils import Similarity, Message, GraphNode, File, Folds

N_FOLDS = 5

def read_data_from_files(suffix, emb_prefix):
    graph_file = File(f'./data/graphs/graph_{suffix}.pickle')
    url_file = File(f'./data/urls/url_{suffix}.pickle')
    emb_file = File(f'./data/{emb_prefix}_embeddings.pickle')
    G = graph_file.get_data()
    url_labels = url_file.get_data()
    urls = list(url_labels.keys())
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


def store_result_dic_to_csv_file(filename, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ths+', 'ths-', 'accuracy', 'recursion', 'precision', 'F1'])
        for key, val in data.items():
            t1, t2 = key.split(':')
            accuracy, precision, recall, f1 = val[0], val[1], val[2], val[3]
            writer.writerow([t1, t2, accuracy, precision, recall, f1])


def find_suitable_similarity_thresholds(edge_prefix, sim_type, suffix):
    # # t1 = [1.0, 1.1, 1.2, 1.3, 1.4]
    # # t2 = [0.0, -0.1, -0.2, -0.3, -0.4]
    t1 = [n / 10 for n in range(11)]
    t2 = [n / 10 for n in range(11)]
    result_dic = {}
    for i in t1:
        for j in t2:
            threshold1, threshold2 = i, j
            rs = main(threshold1, threshold2, sim_type, edge_prefix, 'sim_max', suffix)
            key = f'{threshold1}:{threshold2}'
            result_dic[key] = rs
    store_result_dic_to_csv_file(f'graph_{suffix}_{sim_type}_{edge_prefix}_result.csv', result_dic)


def step(edge_type, t1, t2, graph, data, idx):
    status = True
    iteration_num = 0
    training_set = data['train']
    nodes_list = [e for e in list(graph.nodes()) if graph.nodes[e]['label'] == 0.5]
    prev_converged_nodes = -1
    converged_nodes_num_list = []
    while status:
        message = Message(graph, edge_type)
        if edge_type == 'sim':
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
            converged_nodes_num_list.append(current_converged_nodes)
            # print(f'Process {idx} Iteration {iteration_num}')
            # print(f'converged node previous : current = {prev_converged_nodes} : {current_converged_nodes}')
            prev_converged_nodes = current_converged_nodes

            if current_converged_nodes == len(nodes_list):
                status = False
            elif iteration_num > 100:
                status = False
            iteration_num += 1
    print(f'Process {idx} Iteration {iteration_num}')
    print(f'Converged nodes List: {converged_nodes_num_list}')


def worker(args):
    edge_type, t1, t2, G, urls, data, idx = args
    graph_node = GraphNode(G)
    graph_node.init_nodes(data, urls)

    step(edge_type, t1, t2, G, data, idx)

    graph_node.infer_node_labels()
    accuracy, recall, precision, f1 = graph_node.calculate_performance()
    return {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1}


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
    kf.print_dataset_ratio(url_labels)
    data_list = kf.get_data()

    # worker([edge_type, t1, t2, G, url_labels, data_list[0], 0])
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


if __name__ == '__main__':
    edge_prefix = 'word2vec'
    sim_type = 'rbf'
    suffix = '5000'
    # s = [100, 200, 500, 1000, 5000, 10000, 20000, 24826, 'final']
    s = ['final']
    for suffix in s:
        # rs_t1 = main(1, 0, 'cos', edge_prefix, 't1', suffix)
        rs_cos_sim_only = main(1, 0, 'cos', edge_prefix, 'sim_only', suffix)
        # rs_rbf_sim_only = main(1, 0, 'rbf', edge_prefix, 'sim_only', suffix)
        print('graph', suffix)
        # print('0.5 + e', rs_t1)
        print('cos sim_only', rs_cos_sim_only)
        # print('rbf sim_only', rs_rbf_sim_only)




