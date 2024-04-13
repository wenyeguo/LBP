## Software Environments
> Python 3.10.9  
> Scikit Learn 1.3.1   
> NetworkX 3.2.1  
> tqdm 4.64.1 

## Run Code
* python3 lbp.py 
* python3 lbp.py DATA_SUFFIX EMBEDDING_PREFIX TYPE_SIMILARITY TYPE_EDGE_POTENTIAL DELETE_CYCLE  ADD_PRIOR_PROBABILITY CLASSIFY_THRESHOLD
  * example: python3 lbp.py final word2vec rbf sim_only True True 0.5
* python3 lbp.py DATA_SUFFIX EMBEDDING_PREFIX TYPE_SIMILARITY TYPE_EDGE_POTENTIAL DELETE_CYCLE  ADD_PRIOR_PROBABILITY CLASSIFY_THRESHOLD SIMILARITY_THRESHOLD1 SIMILARITY_THRESHOLD2 
  * example: python3 lbp.py final word2vec rbf sim_only True True 0.50.1 0.2)
* DATA_SUFFIX = ["final", '100', '5000']
* EMBEDDING_PREFIX = ["word2vec", "doc2vec", 'deepwalk', '' ]
* TYPE_SIMILARITY = ['rbf', 'cos']
* TYPE_EDGE_POTENTIAL = [t1, sim_only, sim, sim_max]
* DELETE_CYCLE = [True, False] (True - delete unknown cycles)
* ADD_PRIOR_PROBABILITY = [True, False] (True - test urls add prior probability)
* CLASSIFY_THRESHOLD = [0.0, 0.1, ..., 1.0]
* SIMILARITY_THRESHOLD1 = float [0.0 ~ 1.0]
* SIMILARITY_THRESHOLD2 = float [0.0 ~ 1.0]


## Main Purpose
* This is a graph-based machine learning method, using Loopy Belief Propagation algorithm predict unknown URLs.

## Main codes
* baseline
  * extract_features.py - extract features from URLs
  * normalization.py - normalize features
  * traditionalModel.py - test traditional machine learning models
* data_processing
    * create_nodes.py - read url & label, store them with Object URL and assign 'domain'  
    * get_nameserver.py - update Object URL 'nameserver'  
    * get_address.py - update Object URL 'IP'  
    * get_substrings.py - Update Object URL 'substrings' 
    * extract_features.py - extract features from URL objects 
    * create_graph.py - create graph using networkx
* module
  * Contains different modules used in lbp.py
* lbp.py - train model

