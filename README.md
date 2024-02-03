## Software Environments
> Python 3.10.9  
> Scikit Learn 1.3.1   
> NetworkX 3.2.1  
> tqdm 4.64.1 

## Run Code
* python3 prediction_lbp.py

## Main Purpose
* This is a graph-based machine learning method, using Loopy Belief Propagation algorithm predict unknown URLs.

## Main codes
* data_processing
    * create_nodes.py - read url & label, store them with Object URL and assign 'domain'  
    * get_nameserver.py - update Object URL 'nameserver'  
    * get_address.py - update Object URL 'IP'  
    * get_substrings.py - Update Object URL 'substrings' 
    * extract_features.py - extract features from nodes 
    * create_graph.py - create graph using networkx
* prediction_lbp.py - train model