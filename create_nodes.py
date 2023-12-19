import os
import glob
import pandas as pd
from urllib.parse import urlparse
from new_class import URL
from new_class import store_data
from new_class import unique
import tldextract


# read all urls and label, convert into dic {url: label}
# delete url which has conflicting label
def read_data():
    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', './data/*.csv'))))
    X = df["URL"].values
    Y = df["Productivity"].values
    Y = Y.astype('int')
    urls = X.tolist()
    labels = Y.tolist()
    d = {}
    differ_urls = set()
    for i in range(len(urls)):
        url = urls[i]
        if url in d:
            if d[url] != labels[i]:
                differ_urls.add(url)
        else:
            d[url] = labels[i]
    print(f'URLs read from csv file {len(d)}')
    for url in differ_urls:
        if url in d.keys():
            del d[url]
        else:
            print(url)
    print(f'Correct URLs {len(d)}')
    return d


# pip install tldextract
def get_hostname(url):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    return hostname


def extract_domain(hostname):
    extracted = tldextract.extract(hostname)
    # print("subdomain", extracted.subdomain)
    # print("domain", extracted.domain)
    # print("TLD", extracted.suffix)
    domain = extracted.domain
    TLD = extracted.suffix
    domain_name = domain + '.' + TLD
    # print(hostname, ":", domain_name)
    return domain_name


# main code:
# 1. read data X=[urls], Y = [lables] 
# 2. extract domain of each url, assign it to URL

# each url is present as URL, for each url get its paras
# 54721 different urls, total 68602 urls
# words used to store all words appeared in URLs, then using elbow method remove high freq words. then rest words used in network


# initialize each url with Object URL. assign url, label, domain and nameservers(no limitation)
# nodes 53299, unique domains 19162
def create_nodes(data, file):
    hostnames = []
    nodes = []
    urls = []
    domains = []
    nameservers = []
    c = 0
    for key, value in data.items():
        hostname = get_hostname(key)
        hostnames.append(hostname)
        if hostname:
            rs = URL('', '', '', '', '', '')
            rs.url = key
            rs.label = value
            rs.domain = extract_domain(hostname)
            if rs.domain:
                domains.append(rs.domain)
                nodes.append(rs)
        else:
            print("No hostname", key)

    print("nodes", len(nodes))
    store_data(nodes, file + 'nodes')
    # # file domains => used to get IPs
    print("domains", len(domains))
    domains = unique(domains)
    print("unique domains", len(domains))
    store_data(domains, file + 'domains')
    # # file nameservers => used to add nodes and edges
    # nameservers = unique(nameservers)
    # print("nameservers", len(nameservers))
    # store_data(nameservers, file + 'nameservers')


# ./features/d/
data = read_data()
print("Total URL ", len(data))
file = input("Please Enter Directory Name To Store Data: ")
create_nodes(data, file)

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

# each nodes = vector representation ==> later used to calculate edge potential(joint probbility)
# msg passing using min value ==> exchange many times (cost of different labels)

# 1. get IP website has constraint 40/day
# 2. can some url don't have IP/name server


# after has graph, 1st init node class
