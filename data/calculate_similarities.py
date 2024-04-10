from module.similarityModule import Similarity
from module.fileModule import File

# Load essential data
suffix = input('Please Enter Graph File Suffix: ')
emb_prefix = input('Please Enter Embedding File Prefix: ')
sim_type = input('Please Enter calculation type of similarity: ')
graph_file = File(f'./data/graphs/graph_{suffix}.pickle')
emb_file = File(f'./data/{emb_prefix}_embeddings.pickle')
G = graph_file.get_data()
embedding = emb_file.get_data()

# calculate the similarity of each edge
sim = Similarity(G, sim_type, emb_file)
sim.init_edge_similarity()
sims = sim.get_similarity()
similarityFile = File(f'./data/similarity/{emb_prefix}_{sim_type}_similarity.pickle')
similarityFile.store_data(sims)

min_sim, max_sim = sim.get_similarity_range()
print(f'similarity range with {sim_type}: [{min_sim}, {max_sim}]')


