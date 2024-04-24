## Software Environments
> Python 3.10.9  
> Scikit Learn 1.3.1   
> NetworkX 3.2.1  
> tqdm 4.64.1 

## Run Code - Shell Scripts
* chmod +x script.sh
* ./run.sh

## Run Code - Terminal Command lines
* python3 lbp.py 
* python3 lbp.py DATA_SUFFIX EMBEDDING_PREFIX TYPE_SIMILARITY TYPE_EDGE_POTENTIAL DELETE_CYCLE  ADD_PRIOR_PROBABILITY CLASSIFY_THRESHOLD
  * example: python3 lbp.py final word2vec rbf sim_only True True 0.5
* python3 lbp.py DATA_SUFFIX EMBEDDING_PREFIX TYPE_SIMILARITY TYPE_EDGE_POTENTIAL DELETE_CYCLE  ADD_PRIOR_PROBABILITY CLASSIFY_THRESHOLD SIMILARITY_THRESHOLD1 SIMILARITY_THRESHOLD2 
  * example: python3 lbp.py final word2vec rbf sim_only True True 0.5 0.1 0.2)
* DATA_SUFFIX = ["final", '53K', '100K']
* EMBEDDING_PREFIX = ["word2vec", "doc2vec", 'deepwalk']
* TYPE_SIMILARITY = ['rbf', 'cos']
* TYPE_EDGE_POTENTIAL = [t1, sim_only, sim, sim_max]
* DELETE_CYCLE = [True, False] (True - delete unknown cycles)
* ADD_PRIOR_PROBABILITY = [True, False] (True - test urls add prior probability)
* CLASSIFY_THRESHOLD = [0.0, 0.1, ..., 1.0]
* SIMILARITY_THRESHOLD1 = float [0.0 ~ 1.0]
* SIMILARITY_THRESHOLD2 = float [0.0 ~ 1.0]

## Main Purpose
* This is a graph-based machine learning method, using Loopy Belief Propagation algorithm predict unknown URL labels.

## Main codes
* baseline
  * extract_features.py - extract features from URLs.
  * normalization.py - normalize features.
  * traditionalModel.py - test traditional machine learning models.
* data_processing
    * get_domain.py - update csv file with 'domain'.
    * get_nameserver.py - update csv file with 'nameserver'. 
    * get_address.py - update csv file with 'IP'.
    * get_substrings.py - update csv file with 'substrings'. 
    * substrings_preprocessing.py - remove stop words, update csv file with 'filtered_substrings'.
    * create_graph.py - create graph using networkx.
* module
  * Contains different modules used in lbp.py.
* lbp.py - main code

## Dataset
* data (Files too large for upload. Please find them on google driver)
  * graphs - Contains graph files.
  * probability - Represents the prior probability of URLs returned from RandomForest.
  * similarity - Indicates the similarity of graph edges.
  * urls - Consists of URL and label pairs.

