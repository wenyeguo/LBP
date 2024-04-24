import pickle
import networkx as nx
import pandas as pd


def add_edge_between_url_domain(graph, df):
    count = 0
    for idx, row in df.iterrows():
        url = row['url']
        domain = row['domain']
        if not graph.has_edge(url, domain):
            graph.add_edge(url, domain)
        else:
            count += 1
            print(url, ":", domain)
    print(f"url_domain, duplicate edges {count}")


def add_edge_between_domain_ip(graph, df):
    count = 0
    for idx, row in df.iterrows():
        domain = row['domain']
        addresses = row['ip_address'].split(', ')
        for address in addresses:
            if not graph.has_edge(domain, address):
                graph.add_edge(domain, address)
            else:
                count += 1
    print(f"domain_ip, duplicate edges {count}")


def add_edge_between_domain_nameserver(graph, df):
    count = 0
    for idx, row in df.iterrows():
        domain = row['domain']
        nameservers = row['nameservers'].split(', ')
        for server in nameservers:
            if not graph.has_edge(domain, server):
                graph.add_edge(domain, server)
            else:
                count += 1
    print(f"domain_nameserver, duplicate edges {count}")


def add_edge_between_url_substring(graph, df):
    count = 0
    for idx, row in df.iterrows():
        url = row['url']
        if not isinstance(row['filtered_substrings'], str):
            continue
        words = row['filtered_substrings'].split(', ')
        for word in words:
            if not graph.has_edge(url, word):
                graph.add_edge(url, word)
            else:
                count += 1

    print(f"url_filtered_substrings, duplicate edges {count}")


def extract_subset(data):
    """ Extract a subset from the data."""
    benign_data = data[data['type'] == 'benign']
    malicious_data = data[data['type'] == 'phishing']
    size = 100 * 1000 - malicious_data.shape[0]
    random_benign_data = benign_data.sample(n=size, random_state=42)
    print(f' benign : malicious: {random_benign_data.shape}: {malicious_data.shape}')
    combined_df = pd.concat([random_benign_data, malicious_data], ignore_index=True)
    combined_df.to_csv("data/dataset_100K.csv", index=False)


"""
    Create Graph with networkx
    Input: path of url dataset
    Output: Save graph into a pickle file
"""
file = 'data/dataset_20.csv'
df = pd.read_csv(file)
print(df.shape)

"""
    Add edges: 
        url & domain
        domain & IP
        domain & nameserver
        url & substrings
"""
graph = nx.Graph()
add_edge_between_url_domain(graph, df)
add_edge_between_domain_ip(graph, df)
add_edge_between_domain_nameserver(graph, df)
add_edge_between_url_substring(graph, df)
print("FINISH ADD EDGES")
print("Nodes Size:", graph.number_of_nodes())
print("Edges Size:", graph.number_of_edges())

# store graph
graph_name = 'graph_size.pickle'
with open(graph_name, 'wb') as f:
    pickle.dump(graph, f)
