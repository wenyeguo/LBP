import pickle
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
"""
Read URLs and Substrings from dataset
"""
file = '../data_processing/data/dataset_306K.csv'
df = pd.read_csv(file)
urls = df['url'].str.cat(sep=', ').split(', ')
words = df['filtered_substrings'].str.cat(sep=', ').split(', ')

corpus = []
corpus.extend(urls)
corpus.extend(words)
sentences = [[c] for c in corpus]
""""
word2vec
"""
model = Word2Vec(sentences=sentences, vector_size=128, min_count=1, workers=4, epochs=10)
word2vec_file = 'word2vec_new_dataset_model.sav'
pickle.dump(model, open(word2vec_file, 'wb'))
print(f'Successfully store word2vec model to {word2vec_file}')

"""
doc2vec
"""
# Create tagged data (associate each sentence with a unique tag/ID)
tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(sentences)]
# Train the Doc2Vec model
model = Doc2Vec(vector_size=100, min_count=1, workers=4, epochs=10)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
doc2vec_file = 'doc2vec_new_dataset_model.sav'
pickle.dump(model, open(doc2vec_file, 'wb'))
print(f'Successfully store doc2vec model to {doc2vec_file}')
