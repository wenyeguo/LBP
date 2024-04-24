import pandas as pd
from tldextract import tldextract


def get_domain(url):
    extracted = tldextract.extract(url)
    domain = extracted.domain
    TLD = extracted.suffix
    # print("subdomain", extracted.subdomain)
    # print("domain", extracted.domain)
    # print("TLD", extracted.suffix)
    domain_name = domain + '.' + TLD
    # print(url, ":", domain_name)
    return domain_name


'''
    Input: Path of urls dataset
    Output: Save the dataset including domain into a separate CSV file
'''
file = 'data/dataset_306K.csv'
df = pd.read_csv(file)
df['domain'] = df['url'].apply(get_domain)
df.to_csv("dataset_with_all_features.csv", index=False)
