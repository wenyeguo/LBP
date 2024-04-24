import subprocess
import pandas as pd


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


'''
    Input: Path of urls dataset
    Output: Save the dataset including nameserver into a separate CSV file
'''
file = 'domain.csv'
df = pd.read_csv(file)
df['nameservers'] = df['domain'].apply(get_authoritative_nameserver)
df.to_csv("domain_nameservers.csv", index=False)
