from new_class import URL
from new_class import load_data, store_data
from key import KEY
import requests
import json


def get_ips(domain):
    IP_list = []
    url = "https://www.virustotal.com/api/v3/domains/" + domain + "/resolutions?limit=40"
    headers = {"accept": "application/json",
               "x-apikey": KEY}
    response = requests.get(url, headers=headers)
    rs = response.json()
    # print("rs", rs)
    if "error" in rs:
        print("ERROR", domain)
    else:
        if "data" in rs:
            data = rs["data"]
            for i in range(len(data)):
                attributes = data[i]["attributes"]
                IP = attributes["ip_address"]
                IP_list.append(IP)
    return IP_list


def find_addresses(domains, filename):
    domain_address = {}
    for d in domains:
        IP = get_ips(d)
        print(f"'{d}': {IP}")
        if IP:
            domain_address.update({d: IP})
        else:
            print("No IP", d)
    print(f'Domain {len(domains)}')
    print("Domain has IP", len(domain_address))
    store_data(domain_address, filename)
    return domain_address


def update_node_with_address(nodes, address, filename):
    new_nodes = []
    for node in nodes:
        node_domain = node.get_domain()
        if node_domain in address.keys():
            IPs = address[node_domain]
            node.set_address(IPs)
            new_nodes.append(node)

    print("Nodes", len(new_nodes))
    store_data(new_nodes, filename)


# [start:end] not include end index
def get_k_items(data, start, end):
    rs = dict(list(data.items())[start: end])
    return rs


# domain has IP and nameserver = 7577
nodes = load_data("./features/total_nodes/nodes")
print(f'Total nodes {len(nodes)}')

domain_nameserver = load_data("./features/d/domains_nameserver")
length = len(domain_nameserver)
print(f'Domains with nameserver {length}')

d = get_k_items(domain_nameserver, 0, 100)
num = "1"
address = find_addresses(d, './features/new/domains_address' + num)
update_node_with_address(nodes, address, './features/new/nodes_address' + num)

# # # extract from nodes
# domain_addresses = {}
# for node in nodes:
#     node_domain = node.get_domain()
#     if node_domain in domain_nameserver.keys() and node_domain not in domain_addresses.keys():
#         node_IP = node.get_address()
#         domain_addresses.update({node_domain: node_IP})
# print(f'Domain has IP {len(domain_addresses)}')
# store_data(domain_addresses, 'features/d/domains_address')
#
# print(f'Total nodes {len(nodes)}')
# new_nodes = []
# for node in nodes:
#     node_domain = node.get_domain()
#     if node_domain in domain_addresses.keys():
#         new_nodes.append(node)
#
# print(f'Nodes {len(new_nodes)}')
# store_data(new_nodes, './features/d/nodes_address')
# Domain 7577 Nodes 25816
