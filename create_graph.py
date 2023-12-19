import pickle
from new_class import URL, load_data
import networkx as nx


def add_node_to_graph(data, G):
    for i in data:
        if not G.has_node(i):
            G.add_node(i)
        else:
            print("NODE ALREADY ADDED", i)


def add_edge_between_url_domain(G, nodes, urls, domains):
    for node in nodes:
        url_string = node.get_url()
        url_domain = node.get_domain()
        if url_string in urls:
            if url_domain in domains:
                if not G.has_edge(url_string, url_domain):
                    G.add_edge(url_string, url_domain)
                # else:
                #     print("EDGE ALREADY ADDED")


def add_edge_between_domain_IP(G, nodes, domains, IP):
    for node in nodes:
        url_domain = node.get_domain()
        url_IPs = node.get_address()
        for k in url_IPs:
            if url_domain in domains:
                if k in IP:
                    if not G.has_edge(url_domain, k):
                        G.add_edge(url_domain, k)
                    # else:
                    #     print("EDGE ALREADY ADDED")


def add_edge_between_domain_server(G, nodes, domains, nameservers):
    for node in nodes:
        url_domain = node.get_domain()
        url_nameservers = node.get_nameserver()
        for n in url_nameservers:
            if url_domain in domains:
                if n in nameservers:
                    if not G.has_edge(url_domain, n):
                        G.add_edge(url_domain, n)
                    # else:
                    #     print("EDGE ALREADY ADDED")


def add_edge_between_url_substring(G, nodes, urls, substrings):
    for node in nodes:
        url_string = node.get_url()
        url_substrings = node.get_substrings()
        if url_string in urls:
            for sub in url_substrings:
                if sub in substrings:
                    if not G.has_edge(url_string, sub):
                        G.add_edge(url_string, sub)
                    # else:
                    #     print("EDGE ALREADY ADDED")


# load features from directory - ./features/d/data/
# dir_name = input("Please enter directory name: ")
dir_name = './features/data/'
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

# for node in nodes:
#     node.print_self()
for s in substrings:
    if s.count('.') != 0:
        print(s)
# list = ['goo.gl', 'http://satreg784.3utilities.com/sat/oauth-hotmail.php', 'https://docs.google.com/document/d/12zC9BxRtuj1tZG3oN5c0yWgfvqiRDAcM27SAPhLcxGI/edit',
#         'https://docs.google.com/forms/d/1eVzDn_8RzxN2pzxJiLhGmmyucJvuVv3bNGbSyRswos0/prefill', 'youtu.be',
#         'http://myfidelitybankbenefits.com/myfidelitybankbenefits.com', 'http://www.myfidelitybankbenefits.com/myfidelitybankbenefits.com/']
# for l in list:
#     print(l)
#     if l in urls:
#         print("URLS")
#     if l in domains:
#         print("domains")
#     if l in substrings:
#         print("words", substrings.index(l))
#     if l in nameservers:
#         print('servers')
#     if l in IP:
#         print('IP')
# print(substrings[11403])
# # create graph
# G = nx.Graph()
# # add nodes: url, domain, ip, nameserver, substrings
# add_node_to_graph(urls, G)
# add_node_to_graph(domains, G)
# add_node_to_graph(nameservers, G)
# add_node_to_graph(substrings, G)
# add_node_to_graph(IP, G)
# print("FINISH ADD NODE")
# # #
# # add edges: url & domain, domain & IP, domain & name server, url & substrings
# add_edge_between_url_domain(G, nodes, urls, domains)
# add_edge_between_domain_IP(G, nodes, domains, IP)
# add_edge_between_domain_server(G, nodes, domains, nameservers)
# add_edge_between_url_substring(G, nodes, urls, substrings)
# print("FINISH ADD EDGE")
# # #
# print("Nodes Size:", G.number_of_nodes())
# print("Edges Size:", G.number_of_edges())
# #
# # # store graph in file "./features/graph_test.gpickle"
# graph = input("Please enter graph filename to store: ")
# with open(graph, 'wb') as f:
#     pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

