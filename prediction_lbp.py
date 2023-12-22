import copy
import pickle
from new_class import load_data
import networkx as nx
import math
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm

N_FOLDS = 5


# init graph nodes, nodes = entities (url, words, IP, nameserver)
# [benign, phishing]
def check_difference(l1, l2):
    c = 0
    for i in range(2):
        dif = abs(l1[i] - l2[i])
        if dif < 0.0001 or dif < l1[i] / 100000000:
            c += 1
        else:
            print(f'l1 {l1}')
            print(f'l2 {l2}')
            print(f'{dif}')
    if c == 2:
        return True
    else:
        return False


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


# update tonode['msg_sum'] -= msg_nbr[fromnode] + new_msg
# update tonode['msg_nbr'][fromnode] = new_msg
def update_node_msg(fromnode, tonode, g, msg):
    temp_g = copy.deepcopy(g)
    to_entity = g.nodes[tonode]
    for i in range(2):
        to_entity['msg_sum'][i] = to_entity['msg_sum'][i] - to_entity['msg_nbr'][fromnode][i] + msg[i]
        to_entity['msg_nbr'][fromnode][i] = msg[i]

    # check if msg_sum == sum of all msg_nbr
    right = check_msg_sum(g, tonode)
    if not right:
        print("!!! WRONG NODE MSG UPDATE !!!")
        print(f'NODE {tonode}')
        print('before', temp_g.nodes[tonode])
        print('after', g.nodes[tonode])


def check_msg_sum(g, node):
    entity = g.nodes[node]
    nbr_sum = [0] * 2
    for nbr in g.neighbors(node):
        if nbr not in entity['msg_nbr']:
            print(f'{nbr} is not nbr of {node}')

    for nbr in entity['msg_nbr']:
        for i in range(2):
            nbr_sum[i] += entity['msg_nbr'][nbr][i]
    return check_difference(entity['msg_sum'], nbr_sum)


# if fromnode 0, tonode min(0, 1) to
    # 1. msg[0]  ==> log(1-0.5) + x0y0 (0.5+e) + from[msg_sum][0] - from[msg-nbr][tonode][0]
    #    to 1 ==> log(1-0.5) +x0y1 (0.5-e) + from[msg_sum][0] - from[msg-nbr][tonode][0]
    # 2. msg[1] fromnode 1, tonode
    #   to0 -> log(1-0.5) + x1y0 (0.5-e) + from[msg_sum][1] - from[msg-nbr][tonode][1]
    #   to1 -> log(1-0.5) + x1y1 (0.5+e) + from[msg_sum][1] - from[msg-nbr][tonode][1]
    # print(f'MIN SUM fromnode ------ {fromnode}')
    # print(f'{g.nodes[fromnode]["msg_sum"]}')
def min_sum(g, fromnode, tonode):
    e = 0.001
    msg = [0] * 2
    right = check_msg_sum(g, fromnode)
    if not right:
        print(f'!!!!!!!!!!!!!!!!WRONG msg sum of from node!!!!!!!!!!!!!!!!')

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


# should have label ==> train set
# all hidden urls: prior of the url has label l == 0.5 
# phi edge potential (x has label l, y has label l'): 1. Polonium (0.5 +/- 0.001) 2. improved compatibility matrix(similarity) (min/max(ths, sim(x, y)))
# phi edge potential joint-probability (nbr x,y relation based on labels)


# # known label {0:[1,0], 1:[0,1]}
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
    # min-sum algorithm ==> prior probability=="label", joint probability x&y==edge potential, sum of nbr msgs
    msg = min_sum(g, fromnode, tonode)
    update_node_msg(fromnode, tonode, g, msg)


def fun_dif(l):
    n = len(l)
    k = 0
    for i in range(n - 1):
        dic1 = l[i]
        dic2 = l[i + 1]
        for key in dic1:
            old = dic1[key]
            new = dic2[key]
            if new > old:
                k += 1
    print("difference increasing", k)


#
graph = './features/graph_1000.gpickle'
# graph = input("Please enter graph filename: ")
with open(graph, 'rb') as f:
    G = pickle.load(f)
print("Nodes:", G.number_of_nodes(), ", Edges:", G.number_of_edges())
print()

file = './features/data_1000/urls'
# file = input("Please enter urls filename: ")
urls = load_data(file)
data = []
for key in urls.keys():
    data.append(key)

kf = KFold(n_splits=5, shuffle=True)
precision_sum = 0
recall_sum = 0
f1score_sum = 0
accuracy_sum = 0
# train : test = 4 : 1
for i, (train, test) in enumerate(kf.split(data)):
    init_graph_nodes(G)

    training_set = set(np.array(data)[train])
    test_set = set(np.array(data)[test])
    print(f'Fold {i} init, train : test = {len(training_set)}, {len(test_set)}')

    # set training node, known label
    # ==> assign node['label'], node['cost'],
    train_b = 0
    train_m = 0
    for node in G.nodes:
        if node in training_set:
            # print('Train', node)
            G.nodes[node]['label'] = urls[node]
            if urls[node] == 1:
                G.nodes[node]['cost'] = [0.99, 0.01]
                train_m += 1
            elif urls[node] == 0:
                G.nodes[node]['cost'] = [0.01, 0.99]
                train_b += 1
    print(f"Train benign : malicious = {train_b} : {train_m}")
    # send msg
    # # node with label (label != 0.5) send msg to node without label (label = 0.5)
    k = 0
    z = 0
    num = 1
    step = True
    # list_differ = []
    step_result = []
    stop = 0

    while step:
        # print all nodes check
        print(f'ITERATE {num}')

        # check if msg_sum equal to the sum of msg_nbr
        for node in G.nodes():
            if G.nodes[node]['label'] == 0.5:
                # print(node, "=====", G.nodes[node])
                sum = [0] * 2
                for nbr in G.nodes[node]['msg_nbr']:
                    m = G.nodes[node]['msg_nbr'][nbr]
                    # n = G.nodes[nbr]['msg_sum']
                    # if m != n:
                    #     print(f'{nbr} personal msg != nbr stored msg')
                    for i in range(2):
                        sum[i] += m[i]
                msg_sum_stored = G.nodes[node]['msg_sum']
                same = check_difference(msg_sum_stored, sum)
                if not same:
                    # if  msg_sum_stored != sum:
                    #     print("----------- ----------- ----------- sum of msg_nbr equal to msg_sum")
                    # else:
                    print(f"----------- ----------- -----------{node} WRONG msg sum")
                    print(f'stored {msg_sum_stored}')
                    print(f'cal {sum}')
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        temp = copy.deepcopy(G)  # used for converge, if difference < TH, terminate

        # msg passing: 1. train node send msg to all hidden node 2. hidden node send msg to all hidden node
        progress_bar = tqdm(total=len(G.nodes()), desc="Passing message...")
        for node in G.nodes():
            if G.nodes[node]["label"] != 0.5:
                if k == 0:
                    # print(f"TRAIN NODE {node}")
                    for nbr in G.neighbors(node):
                        # print(f"nbr {nbr}")
                        if G.nodes[nbr]["label"] == 0.5:
                            # print(f"...... RECEIVE MSG ......")
                            send_msg_from_node_with_label(G, node, nbr)
                # k += 1
            if G.nodes[node]["label"] == 0.5:
                if z == 0:
                    # print(f"TEST NODE {node}")
                    for nbr in G.neighbors(node):
                        # print(f"nbr {nbr}")
                        if G.nodes[nbr]["label"] == 0.5:
                            # print(f"...... RECEIVE MSG ......")
                            send_msg_from_node_no_label(G, node, nbr)
            progress_bar.update(1)
        progress_bar.close()
        # print(c_node)
        # z += 1
        # 
        # check if hidden node msg_sum converged (TH = 0.01) 
        num_total = 0
        num_diff = 0
        differ = {}
        for node in G.nodes():
            if G.nodes[node]['label'] == 0.5:
                num_total += 1
                new_msg = G.nodes[node]['msg_sum']
                old_msg = temp.nodes[node]['msg_sum']
                k = 0
                for i in range(2):
                    d = abs(new_msg[i] - old_msg[i])
                    differ.update({node: d})
                    if d < 0.01:
                        k += 1

                if k == 2:
                    num_diff += 1
        # list_differ.append(differ)
        # if num > 2:
        #     fun_dif(list_differ)
        print('--------------------------------------------------')
        print(f'total hidden nodes: {num_total}')
        print(f'nodes without update: {num_diff}')
        print('--------------------------------------------------')
        # step_result.append(num_diff)
        # step_size = len(step_result)
        # if step_size > 2:
        #     new_step = step_result[-1]
        #     old_step = step_result[-2]
        #     if new_step == old_step:
        #         stop += 1
        # if stop > num / 2:
        #     step = False
        #     print(f'TERMINATE ')

        # terminate if converged or the number of iterations is bigger than N (current N = 5) 
        if num_diff == num_total:
            step = False
            print(f'TERMINATE belief not change')
        elif num > 5:
            step = False
            print(f'TERMINATE larger than max iterations')
        else:
            num += 1

    # assign cost to each hidden node, cost used to predict node label
    for node in G.nodes:
        if node not in training_set:
            cost = [0] * 2
            prior_with_label = math.log(1 - G.nodes[node]["label"])
            for i in range(2):
                msg_sum = G.nodes[node]['msg_sum'][i]
                cost[i] += prior_with_label + msg_sum
            G.nodes[node]['cost'] = cost

    # assign predict label (equal to less cost label)
    for node in G.nodes:
        label = G.nodes[node]['label']
        cost_benign = G.nodes[node]['cost'][0]
        cost_phish = G.nodes[node]['cost'][1]
        if cost_benign < cost_phish:
            G.nodes[node]["predict_label"] = 0
        else:
            G.nodes[node]["predict_label"] = 1

    # check train, test accuracy
    train_right = 0
    train_wrong = 0
    test_right = 0
    test_wrong = 0
    for node in G.nodes:
        if node in training_set:
            if G.nodes[node]["label"] == G.nodes[node]["predict_label"]:
                train_right += 1
            else:
                train_wrong += 1
        if node in test_set:
            label = urls[node]
            if label == G.nodes[node]["predict_label"]:
                test_right += 1
            else:
                test_wrong += 1
    train_accuracy = (train_right / (train_right + train_wrong)) * 100
    test_accuracy = (test_right / (test_right + test_wrong)) * 100
    print('Accuracy: train =', "{:.2f}".format(train_accuracy), "%, test = ", "{:.2f}".format(test_accuracy), '%')
    # print(f'train = {train_right} : {train_wrong}')
    # print(f'test = {test_right} : {test_wrong}')

    # test split into benign, malicious for compare predict label and actual label
    benign_test = []
    phish_test = []
    for node in G.nodes:
        if node in test_set:
            if urls[node] == 0:
                benign_test.append(node)
            else:
                phish_test.append(node)
    print(f'TEST SET actual label b : m = {len(benign_test)} : {len(phish_test)}')

    # true_postive - TP_b_b (true positive predict positive benign actual positive benign)
    # true_negative - TN_m_m
    # false_positive - FP_b_m
    # false_negative - FN_m_b
    # calculate accuracy, precision, recall, f1
    TP_B_B = 0
    TN_M_M = 0
    FP_B_M = 0
    FN_M_B = 0
    for node in G.nodes:
        if node in test_set:
            if node in benign_test:
                if G.nodes[node]['predict_label'] == 0:
                    TP_B_B += 1
                else:
                    FN_M_B += 1
            if node in phish_test:
                if G.nodes[node]['predict_label'] == 0:
                    FP_B_M += 1
                else:
                    TN_M_M += 1
    print(f'True Positive : True Negative : False Positive: False Negative = {TP_B_B} : {TN_M_M} : {FP_B_M} : {FN_M_B}')
    accuracy_b = TP_B_B / (TP_B_B + FN_M_B)
    accuracy_m = TN_M_M / (TN_M_M + FP_B_M)
    print(f'Benign accuracy', "{:.4f}".format(accuracy_b), 'Phish accuracy', "{:.4f}".format(accuracy_m))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    if len(test_set) == TP_B_B + FN_M_B + TN_M_M + FP_B_M:
        accuracy_final = (TN_M_M + TP_B_B) / len(test_set)
        print(f'FINAL accuracy: {accuracy_final}')
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
    print(f'Recall', "{:.4f}".format(recall))
    print(f'Precision', "{:.4f}".format(precision))
    print('F1 score', "{:.4f}".format(f1))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    precision_sum += precision
    recall_sum += recall
    f1score_sum += f1
    accuracy_sum += accuracy_final
    print()

print("Done.")
print()

print("Averaged precision: {:.4f}".format(precision_sum / N_FOLDS))
print("Averaged recall: {:.4f}".format(recall_sum / N_FOLDS))
print("Averaged F1 score: {:.4f}".format(f1score_sum / N_FOLDS))
print("Averaged accuracy: {:.4f}".format(accuracy_sum / N_FOLDS))

# check if msg difference bigger

#        send msg from train node to hidden
#        1. msg from train x(l) -> hidden y
#
#
#        2. msg from hidden to hidden: min_sum
#           min {log(1- y/l 0.5) + x_l, y/l + x['msg-sum']-x[msg-nbr][y][l], where y label is l
#               log(1-y/l' 0.5 + x_l & y l' + x['msg-sum']-x[msg-nbr][y][l'] where y label is l'}
