import pickle
from new_class import URL, load_data
import networkx as nx


def add_node_to_graph(data, graph):
    for i in data:
        if not graph.has_node(i):
            graph.add_node(i)
        else:
            print("NODE ALREADY ADDED", i)


def add_edge_between_url_domain(graph, nodes, urls, domains):
    for node in nodes:
        url_string = node.get_url()
        url_domain = node.get_domain()
        if url_string in urls:
            if url_domain in domains:
                if not graph.has_edge(url_string, url_domain):
                    graph.add_edge(url_string, url_domain)
                # else:
                #     print("EDGE ALREADY ADDED")


def add_edge_between_domain_IP(graph, nodes, domains, IP):
    for node in nodes:
        url_domain = node.get_domain()
        url_IPs = node.get_address()
        for k in url_IPs:
            if url_domain in domains:
                if k in IP:
                    if not graph.has_edge(url_domain, k):
                        graph.add_edge(url_domain, k)
                    # else:
                    #     print("EDGE ALREADY ADDED")


def add_edge_between_domain_server(graph, nodes, domains, nameservers):
    for node in nodes:
        url_domain = node.get_domain()
        url_nameservers = node.get_nameserver()
        for n in url_nameservers:
            if url_domain in domains:
                if n in nameservers:
                    if not graph.has_edge(url_domain, n):
                        graph.add_edge(url_domain, n)
                    # else:
                    #     print("EDGE ALREADY ADDED")


def add_edge_between_url_substring(graph, nodes, urls, substrings):
    for node in nodes:
        url_string = node.get_url()
        url_substrings = node.get_substrings()
        if url_string in urls:
            for sub in url_substrings:
                if sub in substrings:
                    if not graph.has_edge(url_string, sub):
                        graph.add_edge(url_string, sub)
                    # else:
                    #     print("EDGE ALREADY ADDED")


# main code
dir_name = input("Please enter directory name: ")
# dir_name = './features/data_20000/'
print("Load data from directory: '", dir_name, "'")
url_labels = load_data(dir_name + "urls")
urls = url_labels.keys()
domains = load_data(dir_name + "domains")
substrings = load_data(dir_name + "substrings")
IP = load_data(dir_name + "address")
nameservers = load_data(dir_name + "nameservers")
nodes = load_data(dir_name + "nodes")
print("URLs       ", len(urls))
print("Domains    ", len(domains))
print("substrings ", len(substrings))
print("IP         ", len(IP))
print("Nameserver ", len(nameservers))
print("Nodes      ", len(nodes))

# # create graph
G = nx.Graph()

# add nodes: url, domain, ip, nameserver, substrings
add_node_to_graph(urls, G)
add_node_to_graph(domains, G)
add_node_to_graph(nameservers, G)
add_node_to_graph(substrings, G)
add_node_to_graph(IP, G)
print("FINISH ADD NODE")

# add edges: url & domain, domain & IP, domain & name server, url & substrings
add_edge_between_url_domain(G, nodes, urls, domains)
add_edge_between_domain_IP(G, nodes, domains, IP)
add_edge_between_domain_server(G, nodes, domains, nameservers)
add_edge_between_url_substring(G, nodes, urls, substrings)
print("FINISH ADD EDGE")
# #
print("Nodes Size:", G.number_of_nodes())
print("Edges Size:", G.number_of_edges())

# # store graph
graph = input("Please enter graph filename to store: ")
with open(graph, 'wb') as f:
    pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
