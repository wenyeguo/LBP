import os
import glob
import pandas as pd
from urllib.parse import urlparse
from new_class import URL, load_data
from new_class import store_data
from new_class import unique
import tldextract


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


def create_nodes(data, file):
    hostnames = []
    nodes = []
    urls = []
    domains = []
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


# main code, pip install tldextract
data = read_data()
print("Total URL ", len(data))
file = input("Please Enter Directory Name To Store Data: ")
create_nodes(data, file)
