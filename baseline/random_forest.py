import csv
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

df = pd.read_csv('./features/url_features.csv')
urls, features, labels = df["URL"].values, df.drop(labels=['URL', 'Label'], axis=1), df["Label"].values.astype('int')

# X - feature, Y - labels
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)

model = RandomForestClassifier(random_state=16)
cv_scores = cross_val_score(model, features, labels, cv=5)
predictions = cross_val_predict(model, features, labels, cv=5, method='predict_proba')
url_benign_probability = {}
for i in range(len(urls)):
    url_benign_probability[urls[i]] = round(predictions[i][0], 4)

# store probability to csv file
print(url_benign_probability)

with open('./probability/pickle/benign_probability_RandomForest.pickle', 'wb') as f:
    pickle.dump(url_benign_probability, f)
probaFile = './probability/csv/benign_probability_RandomForest.csv'
with open(probaFile, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['URL', 'Probability (benign)'])
    for url, proba in url_benign_probability.items():
        row = [url, proba]
        writer.writerow(row)
