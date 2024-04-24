import pandas as pd
from urllib.parse import urlparse
import re


def url_parser(url):
    if "http" not in url:
        url = "http://" + url
    try:
        parsed_url = urlparse(url)
    except ValueError as e:
        if str(e) == "Invalid IPv6 URL":
            print("Skipping invalid IPv6 URL:", url)
            parsed_url = None
        else:
            raise e
    if parsed_url:
        paras = {
            'scheme': parsed_url.scheme,
            'netloc': parsed_url.netloc,
            'path': parsed_url.path,
            'params': parsed_url.params,
            'query': parsed_url.query,
            'fragment': parsed_url.fragment
        }
    else:
        paras = None
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


def get_substrings(url):
    paras = url_parser(url)
    substrings = []
    if not paras:
        return substrings
    if paras["netloc"] != '':
        # print("netloc", paras["netloc"])
        netloc_parts = segment_netloc(paras["netloc"])
        substrings.extend(netloc_parts)
    if paras["path"] != '':
        # print("path", paras["path"])
        path_parts = segment_path(paras["path"])
        substrings.extend(path_parts)
    if paras["params"] != '':
        # print(url)
        # print("params", paras["params"])
        # print(paras['path'])
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


'''
    Input: Path of urls dataset
    Output: Save the dataset including substrings into a separate CSV file
'''
file = './data/final_dataset.csv'
df = pd.read_csv(file)
df['substrings'] = df['url'].apply(get_substrings)
df['substrings'] = df['substrings'].apply(lambda x: ', '.join(x))
df.to_csv("filtered_dataset_with_substrings.csv", index=False)

