import pickle
from module.similarityModule import Similarity
from module.fileModule import File

"""
    Calculate similarity of edges in the graph
    Input: 
        emb_file: vector representation of each node in the graph
        graph_file: graph that used to train the model
    Output:
        Save the similarity of edges based on rbf and cos 
"""
# Load essential data
# suffix = input('Please Enter Graph File Suffix: ')
# emb_prefix = input('Please Enter Embedding File Prefix: ')
# sim_type = input('Please Enter calculation type of similarity: ')
emb_file = File(f'emb_file.pickle')
embedding = emb_file.get_data()
for emb_prefix in ['new']:
    for sim_type in ['rbf', 'cos']:
        graph_file = File(f'./graphs/graph_{emb_prefix}.pickle')
        G = graph_file.get_data()
        # calculate the similarity of each edge
        sim = Similarity(G, sim_type, emb_file)
        sim.init_edge_similarity()
        sims = sim.get_similarity()
        with open(f'word2vec_{emb_prefix}_{sim_type}_similarity.pickle', 'wb') as f:
            pickle.dump(sims, f)

        min_sim, max_sim = sim.get_similarity_range()
        print(f'similarity range with {sim_type}: [{min_sim}, {max_sim}]')


