# -*- coding: utf-8 -*-
from helper_functions import *
from sklearn.utils import shuffle
import os
import pickle
import pandas as pd
"""
    - Dataset can be downloaded here: https://github.com/marjanhs/prnn
    - Pretrained word embeddings (binary file): https://fasttext.cc/docs/en/english-vectors.html
    
"""
##########################################
print("extract and pre-process documents")
##########################################
dataset_path = os.path.join("..", "data", "amazon.csv")
data = pd.read_csv('{}'.format(dataset_path), sep='\t')
docs_L_raw, docs_R_raw, labels = extract_text(data)

# shuffle
docs_L_raw, docs_R_raw, labels = shuffle(docs_L_raw, docs_R_raw, labels)

with open(os.path.join("..", "data", "data_Amazon_9000_raw"), 'wb') as f:
    pickle.dump((docs_L_raw, docs_R_raw, labels), f)

############################
print("text pre-processing")
############################
docs_L, docs_R = preprocess(docs_L_raw, docs_R_raw)

#####################
print("count tokens")
#####################
dict_tok_counts, dict_chr_counts = count_tokens_and_characters(docs_L, docs_R)

##########################
print("remove rare words")
##########################
list_tok, list_chr = remove_rare_tok_chr(dict_tok_counts,
                                         dict_chr_counts,
                                         min_freq_of_token=5,
                                         min_freq_of_chr=15,
                                         )

############################
print("add special symbols")
############################
docs_L, docs_R = add_special_tokens(docs_L, docs_R, T_w=20)

#################################################
print("make word vocabulary and word embeddings")
#################################################
V_w, E_w, V_c = make_wrd_chr_vocabularies(list_tok, list_chr, D_w=300)

##########################################
print("store results for Siamese network")
##########################################
with open(os.path.join("..", "data", "data_Amazon_9000"), 'wb') as f:
    pickle.dump((docs_L, docs_R, labels, V_w, E_w, V_c), f)