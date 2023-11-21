from urllib.parse import urlparse
import csv
import os
import glob
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import re
import matplotlib.pyplot as plt

class URL:
    def __init__(self, url, domain, substrings, IP, nameserver, label):
        self.url = url
        self.domain = domain
        self.substring = substrings
        self.IP = IP
        self.nameserver = nameserver
        self.label = label
class g:
     def __init__(self, nodes, edges):
          self.nodes = nodes
          self.edges = edges

# read all urls and label, convert into dic {url: label}
def read_data():
    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', './data/test.csv'))))
    X = df["URL"].values
    Y = df["Productivity"].values
    Y = Y.astype('int')
    urls = X.tolist()
    labels = Y.tolist()
    d = {}
    for i in range(len(urls)):
        # if (urls[i]) in d:
        #     print("duplicated url", urls[i], ". Label =", labels[i])
        d[urls[i]] = labels[i]
    return d


def url_parser(url):
    parts = urlparse(url)
    eles = {
        'scheme': parts.scheme,
        'netloc': parts.netloc,
        'path': parts.path,
        'params': parts.params,
        'query': parts.query,
        'fragment': parts.fragment
    }
    # if parts.username:
    #     print("URL", url, ", username", parts.username, parts.password)
    # if parts.hostname:
    #     print("URL", url, ", hostname", parts.hostname)
    return eles

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
    hostname = [x for x in hostname if x!='']
    words = []
    if username != None:
        words.append(username)
    if password != None:
        words.append(password)
    words.extend(hostname)
    # print("NETLOC", words)
    return words

def segment_path(path):
    path_parts = re.split(r'[/.!&,#$%;]', path)
    path_parts = [x for x in path_parts if x!='']
    # print("PATH", path_parts)
    return path_parts

def segment_query(query):
    query_parts = re.split(r'[=&]', query)
    query_parts = [x for x in query_parts if x!='']
    # print("QUERY", query_parts)
    return query_parts


# words.extend

    d = parts.path.strip('/').split('/')
    queries = parts.query.strip('/').split('&')

def get_substrings(url):
    eles = url_parser(url)
    # if has params, segment it same as path (params using ';' seperate with path)
    # if eles["params"] != '':
    #     print("URL HAS PARAMS", url )
    #     print(eles)
    # print("URL     ", url)
    # print('ELEMENTS', eles)
    substrings = []
    if eles["netloc"] != '':
        # print("netloc", eles["netloc"])
        netloc_parts = segment_netloc(eles["netloc"])
        substrings.extend(netloc_parts)
    if eles["path"] != '':
        # print("path", eles["path"])
        path_parts = segment_path(eles["path"])
        substrings.extend(path_parts)
    if eles["params"] != '':
        # print("params", eles["params"])
        path_parts = segment_path(eles["params"])
        substrings.extend(path_parts)
    if eles['query'] != '':
        # print('query', eles['query'])
        query_parts = segment_query(eles['query'])
        substrings.extend(query_parts)
    if eles["fragment"] != '':
        # print("fragment", eles["fragment"])
        substrings.append(eles["fragment"])
    # print(substrings)
    return substrings



# def romove_stop_words(url):

# extract domain from URL
def get_domain(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    return domain
# python -m pip install requests
import requests
def get_IP(domain):
    url = "https://www.virustotal.com/api/v3/domains/" + domain + "/relationship?limit=10"
    # url = "https://www.virustotal.com/api/v3/domains/" + domain
    print("url", url)
    headers = {"accept": "application/json",
               "x-apikey": "key"}
    # response = requests.get(url, headers=headers)
    # print(response.text)
    return 0

import dns.resolver
def get_autoritative_nameserver(url, doamin):
    # find domain name, query DNS (dnspython)
    try:
        rs = dns,resolver.resolve(doamin, 'NS')
        server = [str(r) for r in rs]
    except Exception as e:
        print(f"Error fetching authoritative name server for {url}: {e}")
        server = []
    return server

from collections import Counter
def remove_stop_words(words):
    word_couter = Counter(words)
    # sorted_words = sorted(word_couter.items(), key = lambda item: item[1], reverse = True)
    # print("words", len(word_couter))
    threshold = 5
    remain_words = []
    # if freq > 800, stop words not add it to words list
    for w in word_couter.keys():
        # print(w, ":", word_couter[w])
        if word_couter[w] <= threshold:
            remain_words.append(w)
    # print('remain', len(remain_words))
    return remain_words

# def get_nameserver(url):
# main code: 
# 1. read data X=[urls], Y = [lables] 
# 2. extract domain of each url, assign it to URL

# each url is present as URL, for each url get its paras
# 54721 different urls, total 68602 urls
# words used to store all words appeared in URLs, then using elbow method remove high freq words. then rest words used in network
data = read_data()
nodes = []
words = []
for key, value in data.items():
    rs = URL('','','','','','')
    rs.url = key
    rs.label = value
    rs.domain = get_domain(key)
    print("domain",rs.domain)
#     rs.IP = get_IP(rs.domain)
#     rs.substring = get_substrings(key)
#     # rs.nameserver = get_autoritative_nameserver(key, rs.domain)
#     if rs.substring:
#         words.extend(rs.substring)
#     nodes.append(rs)
# # print("init words length", len(words))
# network_words = remove_stop_words(words)
# print("network word number", len(network_words))
# for each url, find if network_word included in url. If yes, add new edge
get_IP("www.avclub.com")

     