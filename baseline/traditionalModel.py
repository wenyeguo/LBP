import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.naive_bayes import GaussianNB

K_FOLD = KFold(n_splits=5, shuffle=True, random_state=0)


def get_metric(model, filename):
    """train model"""
    features_df = pd.read_csv(filename)
    features = features_df.drop(['url', 'type', 'label'], axis=1) 
    labels = features_df['label']

    model_name = type(model).__name__
    precisions = cross_val_score(model, features, labels, cv=K_FOLD, scoring='precision')
    recalls = cross_val_score(model, features, labels, cv=K_FOLD, scoring='recall')
    f1scores = cross_val_score(model, features, labels, cv=K_FOLD, scoring='f1')
    accuracy = cross_val_score(model, features, labels, cv=K_FOLD, scoring='accuracy')
    print(f'{model_name}:')
    print(
        f"Precision, Recall, F1 Score: {round(precisions.mean(), 4)}, {round(recalls.mean(), 4)}, {round(f1scores.mean(), 4)}")
    print("Accuracy: ", round(accuracy.mean(), 4))


def store_probability(model, filename):
    """store probability into csv file"""
    features_df = pd.read_csv(filename)
    features = features_df.drop(['url', 'label'], axis=1)
    urls = features_df['url']
    labels = features_df['label']
    predictions = cross_val_predict(model, features, labels, cv=K_FOLD, method='predict_proba')
    url_probability = []
    for i in range(len(urls)):
        url_probability.append([urls[i], labels[i], round(predictions[i][0], 4), round(predictions[i][1], 4)])

    model_name = type(model).__name__
    columns = ['url', 'label', 'Probability (benign)', 'Probability (malicious)']
    probability_df = pd.DataFrame(url_probability, columns=columns)
    probability_df.to_csv(f"probability/benign_probability_{model_name}.csv", index=False, mode='w')


models = [RandomForestClassifier(random_state=0), LogisticRegression(random_state=0), GaussianNB()]
for model in models:
    # change filename according resource
    filename = f"features/normalized_features.csv" 
    get_metric(model, filename)
    # store_probability(model, filename)
