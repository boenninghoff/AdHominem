# -*- coding: utf-8 -*-
from helper_functions import Corpus
import os
import pickle
"""
    - Dataset can be downloaded here: https://github.com/marjanhs/prnn
    - Pretrained word embeddings (binary file): https://fasttext.cc/docs/en/english-vectors.html
    
"""

corpus = Corpus()
corpus.extract_docs()
corpus.remove_rare_tok_chr()
corpus.make_wrd_chr_vocabularies()

with open(os.path.join("data", "data_Amazon_9000"), 'wb') as f:
    pickle.dump((corpus.docs_L_tr, corpus.docs_R_tr, corpus.labels_tr,
                 corpus.docs_L_te, corpus.docs_R_te, corpus.labels_te,
                 corpus.V_w, corpus.E_w, corpus.V_c), f)
