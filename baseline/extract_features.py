import csv
from features import Feature, normalization, read_data_from_csvFile


def store_features_into_csvFile(csv_file_path, data):
    with open(csv_file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['URL', 'Label', "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
             "s", "t", "u",
             "v", "w", "x", "y", "z", 'ratio_of_domain', 'num_of_punctuation', 'num_of_specific_symbols',
             'domain_contain_address', 'num_of_suspiciousWords', 'euclidean_distance', 'divergence_KL', 'distance_KS'])
        for url, features in data.items():
            row = [url]
            for feature in features:
                row.append(feature)
            row = [str(value) for value in row]
            writer.writerows([row])


url_label_dic = read_data_from_csvFile('./data/data.csv')

fList0, urls = [], []
fList1, fList2, fList3, fList4, fList5, fList6, fList7, fList8 = [], [], [], [], [], [], [], []
for url, label in url_label_dic.items():
    urls.append(url)
    f = Feature(url)
    normalized_char_frequency = f.get_character_frequency()
    fList0.append(normalized_char_frequency)
    ratio_of_domain = f.get_domainLengthRatio()
    fList1.append(ratio_of_domain)
    num_of_punctuation = f.get_punctuations()
    fList2.append(num_of_punctuation)
    num_of_specific_symbols = f.get_specificSymbols()
    fList3.append(num_of_specific_symbols)
    domain_contain_address = f.get_isIPAddress()
    fList4.append(domain_contain_address)
    num_of_suspiciousWords = f.get_suspiciousWordsCounts()
    fList5.append(num_of_suspiciousWords)
    euclidean_distance = f.get_Euclidean_Distance()
    fList6.append(euclidean_distance)
    divergence_KL = f.get_KullbackLeiblerDivergence()
    fList7.append(divergence_KL)
    distance_KS = f.get_KolmogorovSmirnovDistance()
    fList8.append(distance_KS)

fList1 = normalization(fList1)
fList2 = normalization(fList2)
fList3 = normalization(fList3)
fList4 = normalization(fList4)
fList5 = normalization(fList5)
fList6 = normalization(fList6)
fList7 = normalization(fList7)
fList8 = normalization(fList8)
url_features = {}
for i, url in enumerate(urls):
    features = [url_label_dic[url]]
    features.extend(fList0[i])
    features.append(fList1[i])
    features.append(fList2[i])
    features.append(fList3[i])
    features.append(fList4[i])
    features.append(fList5[i])
    features.append(fList6[i])
    features.append(fList7[i])
    features.append(fList8[i])
    url_features[url] = features
store_features_into_csvFile('features/url_features.csv', url_features)
