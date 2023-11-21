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
    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', './data/*.csv'))))
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

def get_substrings(url):
    eles = url_parser(url)
    # if has params, segment it same as path (params using ';' seperate with path)
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

# pip install tldextract
def get_hostname(url):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    return hostname

import tldextract
def get_domain(hostname):
    extracted = tldextract.extract(hostname)
    # print("subdomain", extracted.subdomain)
    # print("domain", extracted.domain)
    # print("TLD", extracted.suffix)
    domain = extracted.domain
    TLD = extracted.suffix
    domain_name = domain + '.' + TLD
    return domain_name

# python -m pip install requests
from key import KEY
import requests
def get_IP(domain):
    url = "https://www.virustotal.com/api/v3/domains/" + domain + "/resolutions?limit=40"
    headers = {"accept": "application/json",
               "x-apikey": KEY}
    response = requests.get(url, headers=headers)
    rs = response.json()
    data = rs["data"]
    IP_list = []
    for i in range(len(data)):
        attributes = data[i]["attributes"]
        IP = attributes["ip_address"]
        IP_list.append(IP)
    return IP_list

import subprocess
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

# threshold = 800 
from collections import Counter
def remove_stop_words(words):
    word_couter = Counter(words)
    threshold = 5
    remain_words = []
    for w in word_couter.keys():
        if word_couter[w] <= threshold:
            remain_words.append(w)
    return remain_words

# def get_nameserver(url):

def unique(l):
    list_set = set(l)
    unique_list = (list(list_set))
    return unique_list

import pickle
def store_data(data, filename):
    file = open(filename, 'wb')
    pickle.dump(data, file)
    file.close()


# main code: 
# 1. read data X=[urls], Y = [lables] 
# 2. extract domain of each url, assign it to URL

# each url is present as URL, for each url get its paras
# 54721 different urls, total 68602 urls
# words used to store all words appeared in URLs, then using elbow method remove high freq words. then rest words used in network
data = read_data()
nodes = []
words = []
IP_addresses = []
nameservers = []
for key, value in data.items():
    hostname = get_hostname(key)
    if hostname:
        rs = URL('','','','','','')
        rs.url = key
        rs.label = value
        rs.domain = get_domain(hostname)
        rs.substring = get_substrings(key)
        if rs.substring:
            words.extend(rs.substring) 
#       rs.IP = get_IP(rs.domain)
#       if rs.IP:
#           IP_addresses.extend(rs.IP)
#       rs.nameserver = get_authoritative_nameserver(rs.domain)
#       if rs.nameserver:
#           nameservers.extend(rs.nameserver)
    else:
        print("No hostname", key)



# #     nodes.append(rs)
# # print("nodes RESULT", nodes)
# # IP, nameserver only contains distinct ones
# IP_addresses = unique(IP_addresses)
# print("IPs", len(IP_addresses))
# nameservers = unique(nameservers)
# print("nameservers", len(nameservers))
# remove stop words (freq > 800) from substrings
print("before words", len(words))
network_words = remove_stop_words(words)
print("after words", len(network_words))
store_data(network_words, './features/substrings')
# for each url, find if network_word included in url. If yes, add new edge
# graph - 
#         nodes=[urls(54721), IPs(), words(), nameserver()]
#         edges = matrix (same x,y which is nodes, if has relation then 1, else 0)
# extract data stored:
# 1. words (substrings) 
# 2. IPs 
# 3. nameservers
# 4. urls (URL which contains ralationships with other part)
# 5. ground_truth_urls (ground_truth = [url : label])
# 6. community_truth (all nodes = [node : label], label get from ?????)
# get_authoritative_nameserver("avclub.com")