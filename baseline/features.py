import csv
import math
import pandas as pd
from scipy.stats import ks_2samp
import ipaddress
from collections import Counter
from urllib.parse import urlparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def draw_cnf_matrix(cnf_matrix):
    classNames = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')


def read_data_from_csvFile(csv_file_path):
    url_label_dic = {}
    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row != ['URL', 'label']:
                url_label_dic[row[0]] = row[1]
    return url_label_dic


def normalization(numsList):
    # totalValue = sum(numsList)
    # normalizedList = [num / totalValue for num in numsList]
    # return normalizedList
    max_value = max(numsList)
    min_value = min(numsList)
    if max_value != min_value:
        for idx, num in enumerate(numsList):
            numsList[idx] = round(((num - min_value) / (max_value - min_value)), 2)
    return numsList


class Feature:
    def __init__(self, url):
        self.url = url
        self.url_without_scheme = None
        self.characterFrequency = None
        self.EuclideanDistance = None
        self.KolmogorovSmirnovDistance = None
        self.KullbackLeiblerDivergence = None
        self.domainLengthRatio = None
        self.specificSymbols = None
        self.punctuations = None
        self.isIPAddress = None
        self.TLDCounts = None
        self.suspiciousWordsCounts = None
        self.process_data()

    def process_data(self):
        self.url_without_scheme = self.remove_prefix().lower()
        self.characterFrequency = self.calculate_character_frequency()
        self.EuclideanDistance = self.calculate_Euclidean_Distance()
        self.KolmogorovSmirnovDistance = self.calculate_Kolmogorov_Smirnov_Distance()
        self.KullbackLeiblerDivergence = self.calculate_Kullback_Leibler_Divergence()
        self.domainLengthRatio = self.calculate_domain_length_ratio_over_total_length()
        self.specificSymbols = self.calculate_occurrence_of_specific_symbols()
        self.punctuations = self.calculate_occurrence_of_punctuation_symbols()
        self.isIPAddress = self.is_domain_address()
        self.TLDCounts = self.get_TLDCounts()
        self.suspiciousWordsCounts = self.calculate_num_of_suspicious_words()

    def get_character_frequency(self):
        return self.characterFrequency

    def get_Euclidean_Distance(self):
        return self.EuclideanDistance

    def get_KolmogorovSmirnovDistance(self):
        return self.KolmogorovSmirnovDistance

    def get_KullbackLeiblerDivergence(self):
        return self.KullbackLeiblerDivergence

    def get_domainLengthRatio(self):
        return self.domainLengthRatio

    def get_specificSymbols(self):
        return self.specificSymbols

    def get_punctuations(self):
        return self.punctuations

    def get_isIPAddress(self):
        return self.isIPAddress

    def get_TLDCounts(self):
        return self.TLDCounts

    def get_suspiciousWordsCounts(self):
        return self.suspiciousWordsCounts

    def remove_prefix(self):
        # occurrences = self.url.count('http://') + self.url.count('https://')
        url_without_scheme = self.url[:9].replace('http://', "") + self.url[9:]
        url_without_scheme = url_without_scheme[:9].replace('https://', "") + url_without_scheme[9:]
        return url_without_scheme

    def count_occurrence_of_char(self, char):
        char_count = 0
        for c in self.url_without_scheme:
            if c == char:
                char_count += 1
        return char_count

    def calculate_character_frequency(self):
        chars = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                 "u", "v", "w", "x", "y", "z"]
        totalChars = 0
        char_counts = {}
        for char in self.url_without_scheme:
            if char in chars:
                totalChars += 1
                char_counts[char] = char_counts.get(char, 0) + 1
        url_char_frequency = [0] * len(chars)
        for i in range(len(chars)):
            char = chr(97 + i)
            url_char_frequency[i] = round((char_counts.get(char, 0) / totalChars), 3)

        return url_char_frequency

    def standard_english_freq(self):
        standard_english_letter_approximation = {'a': 8.167, 'b': 1.492, 'c': 2.782, 'd': 4.253, 'e': 12.702,
                                                 'f': 2.228,
                                                 'g': 2.015, 'h': 6.094, 'i': 6.966,
                                                 'j': 0.153, 'k': 0.772, 'l': 4.025, 'm': 2.406, 'n': 6.749, 'o': 7.507,
                                                 'p': 1.929, 'q': 0.095, 'r': 5.987,
                                                 's': 6.327, 't': 9.056, 'u': 2.758, 'v': 0.978, 'w': 2.360, 'x': 0.150,
                                                 'y': 1.974, 'z': 0.074}
        standard_freq = [freq / 100 for key, freq in standard_english_letter_approximation.items()]
        return normalization(standard_freq)

    def calculate_domain_length_ratio_over_total_length(self):
        parsed_url = urlparse(self.url)
        domain = parsed_url.netloc
        return round(len(domain) / len(self.url), 2)

    def calculate_occurrence_of_specific_symbols(self):
        totalOccurrence = 0
        symbols = ['@', '-']
        char_counter = Counter(self.url_without_scheme)
        for s in symbols:
            totalOccurrence += char_counter[s]
        return totalOccurrence

    def calculate_occurrence_of_punctuation_symbols(self):
        totalOccurrence = 0
        punctuation = ['.', '!', '#', '$', '%', '&', '*', ',', ';', ':', '’']
        char_counter = Counter(self.url_without_scheme)
        for s in punctuation:
            totalOccurrence += char_counter[s]
        return totalOccurrence

    def is_domain_address(self):
        try:
            ipaddress.ip_address(self.url)
            return 1
        except ValueError:
            return 0

    def calculate_num_of_suspicious_words(self):
        suspicious = ["confirm", "account", "secure", "ebayisapi", "webscr", "login", "signin", "submit", "update"]
        totalNum = 0
        for word in suspicious:
            if word in self.url_without_scheme:
                totalNum += 1
        return totalNum

    def calculate_Euclidean_Distance(self):
        normalized_standard_english = self.standard_english_freq()
        totalSum = 0
        for i, charFreq in enumerate(self.characterFrequency):
            distance = math.sqrt((charFreq - normalized_standard_english[i]) ** 2)
            totalSum += distance
        return totalSum

    def calculate_cdf(self, data):
        sorted_keys = sorted(data.keys())
        cumulative_freq = np.cumsum([data[key] for key in sorted_keys])
        cumulative_freq /= cumulative_freq[-1]  # Normalize to [0, 1]
        return dict(zip(sorted_keys, cumulative_freq))

    def ks_test(self, url_cdf, standard_cdf):
        # two-sample KS test
        ks_p_value = ks_2samp(list(url_cdf.values()), list(standard_cdf.values())).pvalue
        return ks_p_value

    def calculate_Kolmogorov_Smirnov_Distance(self):
        # only consider char that present in URL
        normalized_standard_english = self.standard_english_freq()
        url_freq_dic = {}
        standard_english_freq_dic = {}
        for i, charFreq in enumerate(self.characterFrequency):
            if charFreq > 0:
                char = chr(97 + i)
                url_freq_dic[char] = self.characterFrequency[i]
                standard_english_freq_dic[char] = normalized_standard_english[i]

        url_cdf = self.calculate_cdf(url_freq_dic)
        standard_english_cdf = self.calculate_cdf(standard_english_freq_dic)

        ks_similarity = self.ks_test(url_cdf, standard_english_cdf)

        return ks_similarity

    def calculate_Kullback_Leibler_Divergence(self):
        divergence = 0
        normalized_standard_english = self.standard_english_freq()
        for i, charFreq in enumerate(self.characterFrequency):
            if charFreq > 0 and normalized_standard_english[i]:
                divergence += math.log(normalized_standard_english[i] / charFreq) * self.characterFrequency[i]
        return round(divergence, 2)
