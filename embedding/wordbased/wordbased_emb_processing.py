import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm


def read_pickle_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle_file(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def vectorize_node(node, g, emb):
    node_vector = None
    count = 0
    for _nbr in g.neighbors(node):
        if _nbr in set(emb.keys()):
            count += 1
            if node_vector is None:
                node_vector = emb[_nbr].copy()
            else:
                node_vector += emb[_nbr]
    rs = np.divide(node_vector, count)
    return rs


def get_url_words_vector(model, corpus):
    node_embedding = {}
    for c in corpus:
        rs = model.wv[c]
        node_embedding[c] = rs
    return node_embedding


def calculate_neighbor_average_vector(data, graph, features_vector):
    num_processes = 6

    # Split the data into chunks for multiprocessing
    data_dic = {d: len([nbr for nbr in graph.neighbors(d)]) for d in data}
    sorted_data = sorted(data_dic, key=data_dic.get, reverse=True)
    process_data = {p: [] for p in range(num_processes)}
    idx = 0
    while idx < len(sorted_data):
        for p in range(num_processes):
            index = idx + p
            if index >= len(sorted_data):
                break
            process_data[p].append(sorted_data[index])
        idx += num_processes

    # # # Create a multiprocessing Pool
    pool = mp.Pool(processes=num_processes)

    # Execute the worker function in parallel
    results = pool.map(worker,
                       [(processID, chunk, graph, features_vector) for processID, chunk in process_data.items()])

    combined_result = {}
    for result in results:
        combined_result.update(result)

    # Close the multiprocessing Pool
    pool.close()
    pool.join()

    return combined_result


def worker(args):
    idx, data_chunk, graph, features_vector = args

    data_vector = {}
    with (tqdm(total=len(data_chunk), desc=f'Process {idx}')
          as progress_bar):
        for d in data_chunk:
            # print(f'Neighbors {len([nbr for nbr in graph.neighbors(d)])}')
            vector = vectorize_node(d, graph, features_vector)
            if vector is not None:
                data_vector[d] = vector
            else:
                print("No Vector", d)
            progress_bar.update(1)

    return data_vector


def main():
    """word based emb using nbr average represent vector"""
    # Step 1: load word2vec/Doc2vec model
    # './model/word2vec_model.sav'
    emb_type = input('Please Choose model 1 - word2vec, 2 - doc2vec:')
    model_name = 'word2vec' if emb_type == '1' else 'doc2vec'
    filename = f'./{model_name}_new_dataset_model.sav'
    model = read_pickle_file(filename)

    # Step 2: get words and urls vector from model
    file = '../data_processing/data/dataset_306K.csv'
    df = pd.read_csv(file)
    urls = df['url'].str.cat(sep=', ').split(', ')
    words = df['filtered_substrings'].str.cat(sep=', ').split(', ')
    corpus = []
    corpus.extend(urls)
    corpus.extend(words)
    emb = get_url_words_vector(model, corpus)
    features_vector = emb.copy()

    # load needed files
    graph = read_pickle_file('../data_processing/data/graph_new.pickle')
    # features_vector = read_pickle_file('./address_emb_file.pickle')
    print(f'features size {len(features_vector)}')

    # Step 3: get domain, IP, nameserver vector == nbrs / num of nbr
    domains = df['domain'].str.cat(sep=', ').split(', ')
    domains = list(set(domains))
    domains = [d for d in domains if d not in features_vector.keys()]
    # domains = domains[:2]
    domains_vector = calculate_neighbor_average_vector(domains, graph, features_vector)
    features_vector.update(domains_vector)
    write_pickle_file('./domain_emb_file.pickle', features_vector)
    print(f"Domain saved")

    # # Step 3.2: IP
    address = df['ip_address'].str.cat(sep=', ').split(', ')
    address = list(set(address))
    address = [ip for ip in address if ip not in features_vector.keys()]
    print(f'address size {len(address)}')

    address_vector = calculate_neighbor_average_vector(address, graph, features_vector)
    features_vector.update(address_vector)
    write_pickle_file('./address_emb_file.pickle', features_vector)
    print(f"address saved")

    # Step 3.3: nameserver
    nameserver = df['nameservers'].str.cat(sep=', ').split(', ')
    nameserver = list(set(nameserver))
    nameserver = [server for server in nameserver if server not in features_vector.keys()]
    print(f'nameserver size {len(nameserver)}')
    nameserver_vector = calculate_neighbor_average_vector(nameserver, graph, features_vector)
    features_vector.update(nameserver_vector)

    write_pickle_file('./emb_file.pickle', features_vector)
    print(f"nameserver saved")
    print('Num of vectors', len(features_vector.keys()), 'Num of nodes', graph.number_of_nodes())

    print(f"Data saved to emb_file.pickle")


if __name__ == '__main__':
    main()
