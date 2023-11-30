import pickle
from new_class import URL
import networkx as nx
import matplotlib.pyplot as plt 

# load data from filename
def load_data(filename):
    file = open(filename, "rb")
    data = pickle.load(file)
    file.close()
    # print("data read from file", data)
    return data

def print_nodes(data):
    for url_instance in data:
        print("URL:", url_instance.get_url())
        print("Domain:", url_instance.get_domain())
        print("Substrings:", url_instance.get_substrings())
        print("IP:", url_instance.get_IP())
        print("Nameserver:", url_instance.get_nameserver())
        print("Label:", url_instance.get_label())


def add_node_to_graph(data, G):
    for i in data:
        G.add_node(i)

def add_edge_between_url_domain(G, nodes, urls, domains):
    for url_instance in nodes:
        url_string = url_instance.get_url()
        url_domain = url_instance.get_domain()
        print(url_string, url_domain)
        if url_string in urls:
            if url_domain in domains:
                G.add_edge(url_string, url_domain)

def add_edge_between_domain_IP(G, nodes, domains, IP):
    for url_instance in nodes:
        url_domain = url_instance.get_domain()
        url_IPs = url_instance.get_IP()
        for k in url_IPs:
            if url_domain in domains:
                if k in IP:
                    G.add_edge(url_domain, k)

def add_edge_between_domain_server(G, nodes, domains, nameservers):
    for url_instance in nodes:
        url_domain = url_instance.get_domain()
        url_nameservers = url_instance.get_nameserver()
        for n in url_nameservers:
            if url_domain in domains:
                if n in nameservers:
                    G.add_edge(url_domain, n)

def add_edge_between_url_substring(G, nodes, urls, substrings):
    for url_instance in nodes:
        url_string = url_instance.get_url()
        url_substrings = url_instance.get_substrings()
        if url_string in urls:
            for sub in url_substrings:
                if sub in substrings:
                    G.add_edge(url_string, sub)

urls = load_data("./features/urls")
domains = load_data("./features/domains")

# print(domains)
substrings = load_data("./features/substrings")
IP = load_data("./features/address")
nameservers = load_data("./features/nameservers")
nodes = load_data("./features/nodes")
print("URLs length", len(urls))

print("domians length", len(domains))
print("substrings length", len(substrings))
print("IP length", len(IP))
print("nameserver length", len(nameservers))
print("nodes length", len(nodes))
# print_nodes(nodes)

G = nx.Graph()
# add nodes: url, domain, ip, nameserver, substrings
add_node_to_graph(urls, G)
add_node_to_graph(domains, G)
add_node_to_graph(nameservers, G)
add_node_to_graph(substrings, G)
add_node_to_graph(IP, G)
# add edges: url & domain, domain & IP, domain & name server, url & substrings
add_edge_between_url_domain(G, nodes, urls, domains)
add_edge_between_domain_IP(G, nodes, domains, IP)
add_edge_between_domain_server(G, nodes, domains, nameservers)
add_edge_between_url_substring(G, nodes, urls, substrings)

# visualization
pos = nx.spring_layout(G)  # Positions for all nodes
nx.draw(G, pos, with_labels=False, node_color='skyblue', node_size=10)
plt.show()

print("Nodes Size:", G.number_of_nodes() ,len(G.nodes()))
print("Edges Size:", G.number_of_edges(), len(G.edges()))
                    
# nx.write_gpickle(G, "./features/graph.gpickle")      
with open("./features/graph.gpickle", 'wb') as f:
    pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

# add url, domain, ip, nameserver, substrings as nodes

# load all feature files, add them in one array "nodes" [url, words, IP, name servers] 
# n X n matrix, has relation = 1 
