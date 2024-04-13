import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize(X, min_val, max_val):
    # min_max scaling
    if min_val == 0 and max_val == 0:
        return X
    x_normalized = [0] * len(X)
    for i, x in enumerate(X):
        x_normalized[i] = (x - min_val) / (max_val - min_val)
    return x_normalized


# Read the features file into a DataFrame
features_file = "features/features_default.csv"
data = pd.read_csv(features_file)

# Extract features (assuming all columns except the target column are features)
# columnsName = [
#     'url', 'type', 'label', 'kl_divergence', 'entropy', 'digit_letter_ratio', 'top_level_domains_count',
#     'dash_count', 'url_length', 'digits_in_domain', 'suspicious_words_count', 'subdomains_count',
#     'brand_name_modified', 'long_hostname_phishy', 'punctuation_symbols_count',
#     'colons_in_hostname_count', 'ip_address_or_hexadecimal', 'vowel_consonant_ratio',
#     'short_hostname_phishy', 'at_symbol'
# ]
columnsName =['url', 'type', 'label', "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
             "q", "r",
             "s", "t", "u",
             "v", "w", "x", "y", "z", 'ratio_of_domain', 'num_of_punctuation', 'num_of_specific_symbols',
             'domain_contain_address', 'num_of_suspiciousWords', 'euclidean_distance', 'divergence_KL', 'distance_KS']


dataColumns = {'url': data['url'].values, 'type': data['type'].values, 'label': data['label'].values}
for c in columnsName[3:]:
    X = data[c].values
    min_val = np.min(X)
    max_val = np.max(X)
    print(c, min_val, max_val)
    x_normalized = normalize(X, min_val, max_val)
    dataColumns[c] = x_normalized
normalizedData = zip(*[dataColumns[key] for key in dataColumns])
normalizedData = list(normalizedData)

csv_file = "./features/normalized_features.csv"

# Write data to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(columnsName)
    for column_data in normalizedData:
        writer.writerow(column_data)

print("Data has been written to", csv_file)
