from new_class import load_data, store_data, URL
from urllib.parse import urlparse
import re


def url_parser(url):
    parts = urlparse(url)
    paras = {
        'scheme': parts.scheme,
        'netloc': parts.netloc,
        'path': parts.path,
        'params': parts.params,
        'query': parts.query,
        'fragment': parts.fragment
    }
    return paras


def segment_netloc(netloc):
    if '@' in netloc:
        userinfo, netloc = netloc.split('@')
        if ':' in userinfo:
            username, password = userinfo.split(':')
        else:
            username = userinfo
            password = None
    else:
        username = None
        password = None
    hostname = netloc.split('.')
    hostname = [x for x in hostname if x != '']
    words = []
    if username is not None:
        words.append(username)
    if password is not None:
        words.append(password)
    words.extend(hostname)
    # print("NETLOC", words)
    return words


def segment_path(path):
    path_parts = re.split(r'[/.!&,#$%;]', path)
    path_parts = [x for x in path_parts if x != '']
    # print("PATH", path_parts)
    return path_parts


def segment_query(query):
    query_parts = re.split(r'[=&]', query)
    query_parts = [x for x in query_parts if x != '']
    # print("QUERY", query_parts)
    return query_parts


def extract_substrings(url):
    paras = url_parser(url)
    # has params, segment it same as path (params using ';' separate with path)
    substrings = []
    if paras["netloc"] != '':
        # print("netloc", paras["netloc"])
        netloc_parts = segment_netloc(paras["netloc"])
        substrings.extend(netloc_parts)
    if paras["path"] != '':
        # print("path", paras["path"])
        path_parts = segment_path(paras["path"])
        substrings.extend(path_parts)
    if paras["params"] != '':
        # print("params", paras["params"])
        path_parts = segment_path(paras["params"])
        substrings.extend(path_parts)
    if paras['query'] != '':
        # print('query', paras['query'])
        query_parts = segment_query(paras['query'])
        substrings.extend(query_parts)
    if paras["fragment"] != '':
        # print("fragment", paras["fragment"])
        substrings.append(paras["fragment"])
    # print(substrings)
    return substrings


# load nodes, extract and update nodes substrings
def update_substring(nodes):
    for node in nodes:
        node_url = node.get_url()
        words = extract_substrings(node_url)
        if words:
            node.set_substrings(words)
    return nodes


# main code
file = input("Please Enter nodes filename")
# file = './features/d/nodes_address'
nodes = load_data(file)
new_nodes = update_substring(nodes)
store_data(new_nodes, './features/d/nodes_substrings')

