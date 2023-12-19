# python -m pip install requests

import subprocess
from new_class import URL
from new_class import load_data
from new_class import store_data


def get_authoritative_nameserver(domain):
    servers = []
    try:
        rs = subprocess.run(['dig', '+short', 'NS', domain], capture_output=True, text=True)
        if rs.returncode == 0:
            out = rs.stdout.strip().split('\n')
            if out != ['']:
                servers = out
    except Exception as e:
        print(f"Error fetching authoritative name server: {e}")

    return servers


dir_name = input("Please Enter Directory Name of Nodes and Domains: ")
nodes = load_data(dir_name + 'nodes')
domains = load_data(dir_name + 'domains')
new_domains = {}
new_nodes = []
k = len(domains)
for d in domains:
    nameserver = get_authoritative_nameserver(d)
    if nameserver:
        new_domains.update({d: nameserver})
    k -= 1
    print(k)

print(f'domains_nameserver = {len(new_domains)}')
for node in nodes:
    node_domain = node.get_domain()
    if node_domain in new_domains.keys():
        node.set_nameserver(new_domains[node_domain])
        new_nodes.append(node)
print(f'new nodes = {len(new_nodes)}')

store_data(new_domains, dir_name + 'domains_nameserver')
store_data(new_nodes, dir_name + 'nodes_nameserver')

# nodes 25837, domains 7691
