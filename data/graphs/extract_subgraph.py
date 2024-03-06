import copy
import pickle
import networkx as nx
from matplotlib import pyplot as plt

from new_class import URL
# pip install --user python-louvain
#
# def store_data(data, filename):
#     f = open(filename, 'wb')
#     pickle.dump(data, f)
#     f.close()
#     return data
# #
# #
# def draw(graph):
#     pos = nx.spring_layout(graph)
#     nx.draw(graph, pos, with_labels=False, font_weight='bold', node_size=700, node_color='skyblue', font_size=10,
#             edge_color='black', linewidths=1, alpha=0.7)
#     # Highlight the nodes in the extracted cluster
#     nx.draw(cluster_subgraph, pos, with_labels=False, font_weight='bold', node_size=700, node_color='red', font_size=10,
#             edge_color='black', linewidths=1, alpha=0.7)
#     plt.show()
#
#
# def is_self_neighbor(url_objs, nodes):
#     url = ''
#     i = 0
#     for word in nodes:
#         if 'http' in word:
#             url = word
#             i += 1
#     if i == 1:
#         obj = url_objs[url]
#         size = 1 + 1 + len(obj.get_substrings()) + len(obj.get_address()) + len(obj.get_nameserver())
#         if len(nodes) == size:
#             return True
#
#     return False
#
#
#
#
# url_objs = load_data('updated_url_objects.pickle')
# G = load_data('graph_24816.pickle')
# print("Nodes:", G.number_of_nodes(), ", Edges:", G.number_of_edges())
#
#
# largest = 0
# clusters = {}
# for component in nx.connected_components(G):
#     subgraph = G.subgraph(component)
#     length = len(component)
#     largest = max(largest, length)
#     if length not in clusters.keys():
#         clusters[length] = subgraph
#     else:
#         print('already stored')
# #
# store_data(clusters[largest], 'largest_cluster.pickle')
#
#
#
# #
# #
# #
#


# URL object: 1 url, 1 domain
def add_edge_between_url_domain(graph, url_objs):
    for url, obj in url_objs.items():
        url_domain = obj.get_domain()
        if not graph.has_edge(url, url_domain):
            graph.add_edge(url, url_domain)
        else:
            print(f"EDGE ALREADY ADDED")


# each URL object - 1 domain, [] ip address
def add_edge_between_domain_ip(graph, url_objs):
    for url, obj in url_objs.items():
        url_domain = obj.get_domain()
        url_ips = obj.get_address()
        for ip in url_ips:
            if not graph.has_edge(url_domain, ip):
                graph.add_edge(url_domain, ip)
            else:
                print(f"EDGE ALREADY ADDED")


# URL obj: 1 domain, [] servers
def add_edge_between_domain_server(graph, url_objs):
    for url, obj in url_objs.items():
        url_domain = obj.get_domain()
        url_nameservers = obj.get_nameserver()
        for server in url_nameservers:
            if not graph.has_edge(url_domain, server):
                graph.add_edge(url_domain, server)
            else:
                print("EDGE ALREADY ADDED")


# URL object: 1 url, [] substrings
def add_edge_between_url_substring(graph, url_objs):
    for url, obj in url_objs.items():
        url_substrings = obj.get_substrings()
        for word in url_substrings:
            if not graph.has_edge(url, word):
                graph.add_edge(url, word)
            else:
                print("EDGE ALREADY ADDED")



def draw_graph(graph):
    pos = nx.spring_layout(graph)  # Set the positions of nodes using a layout algorithm
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=100, font_weight='bold',font_size=10)
    plt.show()




def get_subgraph(graph, total_urls):
    largest = 0
    clusters = {}
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        length = len(component)
        largest = max(largest, length)
        if length not in clusters.keys():
            clusters[length] = subgraph
    largest_subgraph = clusters[largest]

    rs_graph = nx.Graph(largest_subgraph.edges())
    nodes_to_delete = [node for node, degree in dict(rs_graph.degree()).items() if degree == 1]
    rs_graph.remove_nodes_from(nodes_to_delete)
    urls = []
    for node in rs_graph.nodes():
        if node in total_urls:
            urls.append(node)
    l = list(nx.connected_components(rs_graph))
    print(len(urls))
    draw_graph(rs_graph)
    return rs_graph, len(urls)

def load_data(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
nums = [100, 200, 500, 1000, 5000, 10000, 20000]
for size in nums:

    graph = load_data('graph_final.pickle')
    urls = load_data('../urls/urls_final.pickle')
    url_objs = load_data('url_obj_dic.pickle')

    sub_urls = []
    for node in graph.nodes():
        if len(sub_urls) == size:
            break
        if node in url_objs.keys():
            sub_urls.append(node)
    sub_url_objs = {}
    for i, url in enumerate(sub_urls):
        obj = url_objs[url]
        sub_url_objs[url] = obj

    subgraph = nx.Graph()

    # add edges: url & domain, domain & IP, domain & name server, url & substrings
    add_edge_between_url_domain(subgraph, sub_url_objs)
    add_edge_between_domain_ip(subgraph, sub_url_objs)
    add_edge_between_domain_server(subgraph, sub_url_objs)
    add_edge_between_url_substring(subgraph, sub_url_objs)

    print("Nodes Size:", subgraph.number_of_nodes())
    print("Edges Size:", subgraph.number_of_edges())

    graph_name = f'graph_{size}.pickle'
    with open(graph_name, 'wb') as f:
        pickle.dump(subgraph, f)
    sub_url_label = {}
    for url in sub_urls:
        sub_url_label[url] = urls[url]

    with open(f'../urls/url_{size}.pickle', 'wb') as f:
        pickle.dump(sub_url_label, f)
