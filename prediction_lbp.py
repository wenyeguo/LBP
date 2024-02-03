import copy
import pickle
import time
from data_processing import new_class
import math
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import multiprocessing

N_FOLDS = 5
NUM_PROCESS = N_FOLDS


def init_graph_nodes(g):
    for node in g.nodes():
        g.nodes[node]["label"] = 0.5
        g.nodes[node]["predict_label"] = -1
        g.nodes[node]["cost"] = [0.5, 0.5]
        g.nodes[node]["msg_sum"] = [0, 0]
        g.nodes[node]["msg_nbr"] = {}
        nbrs = [n for n in g.neighbors(node)]
        for nbr in nbrs:
            g.nodes[node]["msg_nbr"][nbr] = [0, 0]


def update_node_msg(fromnode, tonode, g, msg):
    temp_g = copy.deepcopy(g)
    to_entity = g.nodes[tonode]
    for i in range(2):
        to_entity['msg_sum'][i] = to_entity['msg_sum'][i] - to_entity['msg_nbr'][fromnode][i] + msg[i]
        to_entity['msg_nbr'][fromnode][i] = msg[i]


def min_sum(g, fromnode, tonode):
    e = 0.001
    msg = [0] * 2

    for i in range(2):
        msg_benign = 0
        msg_phishing = 0
        msg_benign += math.log((1 - 0.5))
        msg_phishing += math.log((1 - 0.5))

        edge_b = 0.5 + e if i == 0 else 0.5 - e
        edge_m = 0.5 - e if i == 0 else 0.5 + e
        msg_benign += edge_b
        msg_phishing += edge_m

        nbr_msg_sum = [0] * 2
        for j in range(2):
            fromnode_msg_sum = g.nodes[fromnode]["msg_sum"][j]
            fromnode_msg_tonode = g.nodes[fromnode]['msg_nbr'][tonode][j]
            nbr_msg_sum[j] = fromnode_msg_sum - fromnode_msg_tonode
        msg_benign += nbr_msg_sum[0]
        msg_phishing += nbr_msg_sum[1]

        msg[i] = min(msg_benign, msg_phishing)
    return msg



def send_msg_from_node_with_label(g, fromnode, tonode):
    msg = []
    from_label = g.nodes[fromnode]['label']
    if from_label == 0:
        msg = [1, 0]
    elif from_label == 1:
        msg = [0, 1]
    if msg:
        update_node_msg(fromnode, tonode, g, msg)
    else:
        print("No msg update!!!")


def send_msg_from_node_no_label(g, fromnode, tonode):
    msg = min_sum(g, fromnode, tonode)
    update_node_msg(fromnode, tonode, g, msg)


def read_file(graph, filename):
    with open(graph, 'rb') as f:
        G = pickle.load(f)
    print("Nodes:", G.number_of_nodes(), ", Edges:", G.number_of_edges())
    print()

    file = filename
    urls = new_class.load_data(file)
    data = []
    for key in urls.keys():
        data.append(key)
    return G, urls, data


def init_train_nodes(graph, training_set, urls):
    train_b = 0
    train_m = 0
    for node in graph.nodes:
        if node in training_set:
            # print('Train', node)
            graph.nodes[node]['label'] = urls[node]
            if urls[node] == 1:
                graph.nodes[node]['cost'] = [0.99, 0.01]
                train_m += 1
            elif urls[node] == 0:
                graph.nodes[node]['cost'] = [0.01, 0.99]
                train_b += 1
    # print(f"Train benign : malicious = {train_b} : {train_m}")


def predict_label(graph, training_set):
    for node in graph.nodes:
        if node not in training_set:
            cost = [0] * 2
            prior_with_label = math.log(1 - graph.nodes[node]["label"])
            for i in range(2):
                msg_sum = graph.nodes[node]['msg_sum'][i]
                cost[i] += prior_with_label + msg_sum
            graph.nodes[node]['cost'] = cost

    for node in graph.nodes:
        cost_benign = graph.nodes[node]['cost'][0]
        cost_phish = graph.nodes[node]['cost'][1]
        if cost_benign < cost_phish:
            graph.nodes[node]["predict_label"] = 0
        else:
            graph.nodes[node]["predict_label"] = 1
    return graph


def cal_accuracy(graph, training_set, test_set, urls):
    train_right = 0
    train_wrong = 0
    test_right = 0
    test_wrong = 0
    for node in graph.nodes:
        if node in training_set:
            if graph.nodes[node]["label"] == graph.nodes[node]["predict_label"]:
                train_right += 1
            else:
                train_wrong += 1
        if node in test_set:
            label = urls[node]
            if label == graph.nodes[node]["predict_label"]:
                test_right += 1
            else:
                test_wrong += 1

    benign_test = []
    phish_test = []
    for node in graph.nodes:
        if node in test_set:
            if urls[node] == 0:
                benign_test.append(node)
            else:
                phish_test.append(node)

    # # calculate accuracy, precision, recall, f1
    TP_B_B = 0
    TN_M_M = 0
    FP_B_M = 0
    FN_M_B = 0
    for node in graph.nodes:
        if node in test_set:
            if node in benign_test:
                if graph.nodes[node]['predict_label'] == 0:
                    TP_B_B += 1
                else:
                    FN_M_B += 1
            if node in phish_test:
                if graph.nodes[node]['predict_label'] == 0:
                    FP_B_M += 1
                else:
                    TN_M_M += 1

    accuracy_final = 0
    if len(test_set) == TP_B_B + FN_M_B + TN_M_M + FP_B_M:
        accuracy_final = (TN_M_M + TP_B_B) / len(test_set)

    if TP_B_B + FN_M_B == 0:
        recall = 0
    else:
        recall = TP_B_B / (TP_B_B + FN_M_B)

    if TP_B_B + FP_B_M == 0:
        precision = 0
    else:
        precision = TP_B_B / (TP_B_B + FP_B_M)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'Accuracy: {accuracy_final}')
    print(f'Recall', "{:.4f}".format(recall))
    print(f'Precision', "{:.4f}".format(precision))
    print('F1 score', "{:.4f}".format(f1))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    return accuracy_final, recall, precision, f1


def worker(args):
    G, urls, data, idx = args
    training_set = data['train']
    test_set = data['test']
    init_train_nodes(G, training_set, urls)
    nodes_list = [e for e in list(G.nodes()) if G.nodes[e]['label'] == 0.5]
    iteration_num = 1
    step = True

    while step:
        text_desc = "#Iterate" + "{}".format(iteration_num)
        progress_bar = tqdm(total=len(nodes_list) + len(training_set), desc=text_desc)
        for node in training_set:
            progress_bar.update(1)
            for nbr in G.neighbors(node):
                if G.nodes[nbr]["label"] == 0.5:
                    send_msg_from_node_with_label(G, node, nbr)
        for node in nodes_list:
            progress_bar.update(1)
            for nbr in G.neighbors(node):
                if G.nodes[nbr]["label"] == 0.5:
                    send_msg_from_node_no_label(G, node, nbr)
        progress_bar.close()

        if iteration_num >= 5:
            step = False
        else:
            iteration_num += 1

    G = predict_label(G, training_set)
    accuracy, recall, precision, f1 = cal_accuracy(G, training_set, test_set, urls)

    return {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1}


def main(args):
    graph_file = './features/graph_' + args + '.gpickle'
    url_file = './features/data_' + args + '/urls'
    G, urls, data = read_file(graph_file, url_file)
    kf = KFold(n_splits=N_FOLDS, shuffle=True)

    precision_sum = 0
    recall_sum = 0
    f1score_sum = 0
    accuracy_sum = 0

    data_list = []
    for i, (train, test) in enumerate(kf.split(data)):
        init_graph_nodes(G)
        data_pair = {}
        training_set = set(np.array(data)[train])
        test_set = set(np.array(data)[test])
        data_pair['train'] = training_set
        data_pair['test'] = test_set
        print(f'Fold {i} init, train : test = {len(training_set)} : {len(test_set)}')
        data_list.append(data_pair)
    print()
    print('Passing Message ... ')
    start_time = time.perf_counter()
    with multiprocessing.Pool(processes=N_FOLDS) as pool:
        outputs = pool.map(worker, [(G, urls, d, data_list.index(d)) for d in data_list])
    end_time = time.perf_counter()
    print("Total train time {:.4f}".format(end_time - start_time), 'sec')
    for o in outputs:
        precision_sum += o['precision']
        recall_sum += o['recall']
        f1score_sum += o['f1']
        accuracy_sum += o['accuracy']

    print()
    print("Done.")
    print()

    print("Averaged accuracy: {:.4f}".format(accuracy_sum / N_FOLDS))
    print("Averaged precision: {:.4f}".format(precision_sum / N_FOLDS))
    print("Averaged recall: {:.4f}".format(recall_sum / N_FOLDS))
    print("Averaged F1 score: {:.4f}".format(f1score_sum / N_FOLDS))


if __name__ == '__main__':
    arg = 'final'
    main(arg)
