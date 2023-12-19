from new_class import URL, load_data, store_data, unique
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt


def perpendicular_distance(x1, y1, x2, y2, x, y):
    numerator = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1))
    denominator = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distance = numerator / denominator
    return distance


def find_saturation_point(word_counter):
    sorted_words = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)

    freq = []
    for word, f in sorted_words:
        freq.append(f)
    max_x = 0
    max_y = freq[max_x]
    min_x = len(freq) - 1
    min_y = freq[min_x]
    distance = []
    for i in range(len(freq)):
        distance.append(perpendicular_distance(min_x, min_y, max_x, max_y, i, freq[i]))
    distance = np.array(distance)
    max_distance_index = np.argmax(distance)

    saturation_frequency = freq[max_distance_index]
    print(f"Saturation point frequency: {saturation_frequency}")

    # # visualize Saturation point
    # plt.figure(figsize=(10, 6))
    # plt.plot(freq, marker='.')
    # plt.scatter(max_distance_index, freq[max_distance_index], color='red', label='Saturation Point')
    # plt.xlabel('Words Index')
    # plt.ylabel('Frequency')
    # plt.title('Word Frequency Distribution with Saturation Point')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return saturation_frequency


def remove_stop_words(words):
    word_counter = Counter(words)
    threshold = find_saturation_point(word_counter)
    remain_words = []
    for w in word_counter.keys():
        if word_counter[w] <= threshold:
            remain_words.append(w)
    return remain_words


def extract_features_from_nodes(nodes, filename):
    urls = {}
    nameservers = []
    ips = []
    domains = []
    words = []
    for node in nodes:
        node_url = node.get_url()
        node_label = node.get_label()
        urls.update({node_url: node_label})

        node_nameserver = node.get_nameserver()
        nameservers.extend(node_nameserver)

        node_domain = node.get_domain()
        domains.append(node_domain)

        node_ip = node.get_address()
        ips.extend(node_ip)

        node_words = node.get_substrings()
        words.extend(node_words)

    print(
        f'Before: nodes = {len(urls)}, nameserver = {len(nameservers)}, IPs = {len(ips)}, domains = {len(domains)}, '
        f'words = {len(words)}')
    nameservers = unique(nameservers)
    ips = unique(ips)
    domains = unique(domains)
    words = remove_stop_words(words)

    print(
        f'After: nodes = {len(urls)}, nameserver = {len(nameservers)}, IPs = {len(ips)}, domains = {len(domains)}, '
        f'words = {len(words)}')
    #
    # # # store features
    store_data(urls, filename + "urls")
    store_data(domains, filename + "domains")
    store_data(words, filename + "substrings")
    store_data(ips, filename + "address")
    store_data(nameservers, filename + "nameservers")
    store_data(nodes, filename + "nodes")


# nodes = 25816, nameserver = 8617, IPs = 36416, domains = 7577, words = 31563
# extract features from nodes
nodes = load_data("./features/total_nodes/nodes")
# data = nodes[:10]
# print(len(data))
filename = 'features/d/data/'
extract_features_from_nodes(nodes, filename)
