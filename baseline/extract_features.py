import pandas as pd
from features import Feature, normalization, read_data_from_csvFile


def main():
    urls_file = "./data/new_benign_phishing_urls.csv"
    df = pd.read_csv(urls_file)
    num_rows, num_columns = df.shape

    # Sample URLs for demonstration (replace with your DataFrame)
    size = num_rows
    df_sample = df.sample(n=size, random_state=42)
    features = []
    # Iterate over each URL in the sampled DataFrame
    for index, row in df_sample.iterrows():
        url = row['url']
        url_type = row['type']
        url_label = row['label']
        f = Feature(url)
        normalized_char_frequency = f.get_character_frequency()
        ratio_of_domain = f.get_domainLengthRatio()
        num_of_punctuation = f.get_punctuations()
        num_of_specific_symbols = f.get_specificSymbols()
        domain_contain_address = f.get_isIPAddress()
        num_of_suspiciousWords = f.get_suspiciousWordsCounts()
        euclidean_distance = f.get_Euclidean_Distance()
        divergence_KL = f.get_KullbackLeiblerDivergence()
        distance_KS = f.get_KolmogorovSmirnovDistance()
        features.append([
            url,
            url_type,
            url_label,
            *[freq for freq in normalized_char_frequency],
            ratio_of_domain,
            num_of_punctuation,
            num_of_specific_symbols,
            domain_contain_address,
            num_of_suspiciousWords,
            euclidean_distance,
            divergence_KL,
            distance_KS
        ])
    columns = ['url', 'type', 'Label', "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
               "q", "r", "s", "t", "u",
               "v", "w", "x", "y", "z", 'ratio_of_domain', 'num_of_punctuation', 'num_of_specific_symbols',
               'domain_contain_address', 'num_of_suspiciousWords', 'euclidean_distance', 'divergence_KL', 'distance_KS']
    features_df = pd.DataFrame(features, columns=columns)

    # Save the features DataFrame to a file (e.g., CSV)
    features_df.to_csv(f"./features/new_dataset_features_default.csv", index=False, mode='w')

    # Display the features DataFrame
    print(features_df.head())


if __name__ == "__main__":
    main()
