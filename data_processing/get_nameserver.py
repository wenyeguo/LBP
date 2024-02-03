
import subprocess
from new_class import load_data, store_data


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


def update_domain(domains):
    new_domains = {}
    k = len(domains)
    for d in domains:
        nameserver = get_authoritative_nameserver(d)
        if nameserver:
            new_domains.update({d: nameserver})
        k -= 1
        print(k)
    print(f'domains_nameserver = {len(new_domains)}')
    return new_domains


def update_nodes_nameserver(nodes, new_domains):
    new_nodes = []
    for node in nodes:
        node_domain = node.get_domain()
        if node_domain in new_domains.keys():
            node.set_nameserver(new_domains[node_domain])
            new_nodes.append(node)
    print(f'new nodes = {len(new_nodes)}')
    return new_nodes


# main code
dir_name = input("Please Enter Directory Name of Nodes and Domains: ")
nodes = load_data(dir_name + 'nodes')
domains = load_data(dir_name + 'domains')
new_domains = update_domain(domains)
store_data(new_domains, dir_name + 'domains_nameserver')
new_nodes = update_nodes_nameserver(nodes, new_domains)
store_data(new_nodes, dir_name + 'nodes_nameserver')

