import pandas as pd
from key import KEY
import requests


def get_ip_addresses(domain):
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


'''
    Input: Path of urls dataset
    Output: Save the dataset including IP addresses into a separate CSV file
    additional library: python -m pip install requests
'''
file = 'filtered_dataset.csv'
df = pd.read_csv(file)
df['domain'] = df['url'].apply(get_ip_addresses)
df.to_csv("domain_ip_addresses.csv", index=False)
