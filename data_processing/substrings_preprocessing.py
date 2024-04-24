import math
import numpy as np
import pandas as pd
from collections import Counter


def perpendicular_distance(x1, y1, x2, y2, x, y):
    numerator = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1))
    denominator = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distance = numerator / denominator
    return distance


def find_saturation_point(word_counter):
    sorted_words = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)

    freq = []
    for word, f in sorted_words:
        freq.append(f)
    max_x = 0
    max_y = freq[max_x]
    min_x = len(freq) - 1
    min_y = freq[min_x]
    distance = []
    for i in range(len(freq)):
        distance.append(perpendicular_distance(min_x, min_y, max_x, max_y, i, freq[i]))
    distance = np.array(distance)
    max_distance_index = np.argmax(distance)

    saturation_frequency = freq[max_distance_index]
    print(f"Saturation point frequency: {saturation_frequency}")

    """visualize Saturation point"""
    # plt.figure(figsize=(10, 6))
    # plt.plot(freq, marker='.')
    # plt.scatter(max_distance_index, freq[max_distance_index], color='red', label='Saturation Point')
    # plt.xlabel('Words Index')
    # plt.ylabel('Frequency')
    # plt.title('Word Frequency Distribution with Saturation Point')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return saturation_frequency


def remove_stopwords(words):
    word_counter = Counter(words)
    threshold = find_saturation_point(word_counter)
    remain_words = []
    for w in word_counter.keys():
        if word_counter[w] <= threshold:
            remain_words.append(w)
    return remain_words


def preprocess_and_remove_stopwords(df):
    substrings = df['substrings'].str.cat(sep=', ').split(', ')
    remained_substrings = remove_stopwords(substrings)
    return remained_substrings


def print_dataset_information(data):
    urls = data['url'].str.cat(sep=', ').split(', ')
    nameservers = data['nameservers'].str.cat(sep=', ').split(', ')
    ips = data['ip_address'].str.cat(sep=', ').split(', ')
    domains = data['domain'].str.cat(sep=', ').split(', ')
    words = data['substrings'].str.cat(sep=', ').split(', ')
    print(
        f'Total: URLs = {len(urls)}, nameserver = {len(nameservers)}, IPs = {len(ips)}, domains = {len(domains)}, '
        f'words = {len(words)}')

    nameservers = set(nameservers)
    ips = set(ips)
    domains = set(domains)
    words = set(words)
    print(
        f'Unique: URLs = {len(urls)}, nameserver = {len(nameservers)}, IPs = {len(ips)}, domains = {len(domains)}, '
        f'words = {len(words)}')


def filter_substrings(words, valid_words):
    words_list = words.split(', ')
    filtered_words = [word for word in words_list if word in valid_words]
    return filtered_words


"""
    Remove words whose frequency is higher than Saturation point frequency.
    Input: Path of urls dataset
    Output: Save the dataset including filtered substrings into a separate CSV file
"""

file = 'data/unique_url_ip_nameserver_dataset.csv'
df = pd.read_csv(file)
unique_urls = df['url'].unique()
print(f'URLs: {unique_urls.shape}')
# create a new csv store dataset which all feature exists
print_dataset_information(df)
valid_substrings = preprocess_and_remove_stopwords(df)
valid_substrings = set(valid_substrings)
df['filtered_substrings'] = df['substrings'].apply(filter_substrings, args=(valid_substrings,))
df['filtered_substrings'] = df['filtered_substrings'].apply(lambda x: ', '.join(x))
df.to_csv("data/final_dataset.csv", index=False)
