# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import random
import spacy
import textacy.preprocessing as tp
import numpy as np
import os
from tqdm import tqdm
import fasttext

# tokenizer
nlp_token = spacy.load("en_core_web_lg")


#######################
# pre-process documents
#######################
def preprocess(docs_L, docs_R):

    N = len(docs_L)
    
    for i in tqdm(range(N), desc="Use Textacy and Spacy"):
        docs_L[i] = preprocess_siamese(docs_L[i])
        docs_R[i] = preprocess_siamese(docs_R[i])

    return docs_L, docs_R


#################################################
# pre-process single document for Siamese network
#################################################
def preprocess_siamese(doc):

    # pre-process data
    doc = tp.normalize.normalize_unicode(doc)
    doc = tp.normalize_whitespace(doc)
    doc = tp.normalize_quotation_marks(doc)
    doc = tp.replace_emails(doc, replace_with="<EMAIL>")
    doc = tp.replace_urls(doc, replace_with="<URL>")
    doc = tp.replace_hashtags(doc, replace_with="<HASHTAG>")
    doc = tp.replace_emojis(doc, replace_with="<EMOJI>")
    doc = tp.replace_phone_numbers(doc, replace_with="<PHONE>")

    # apply spaCy to tokenize doc
    doc = nlp_token(doc)

    # build new sentences for pre-processed doc
    doc_new = []
    for sent in doc.sents:
        sent_new = ""
        for token in sent:
            token = token.text
            token = token.replace("\n", "")
            token = token.replace("\t", "")
            token = token.strip()
            sent_new += token + " "

        doc_new.append(sent_new[:-1])

    return doc_new


##############
# extract text
##############
def extract_text(data_panda):

    docs_L, docs_R, labels = [], [], []

    for idx in range(data_panda.review.shape[0]):

        temp = data_panda.review[idx].split('$$$')

        doc_1 = BeautifulSoup(temp[0], "html.parser").get_text().encode("utf-8").decode("utf-8")
        doc_2 = BeautifulSoup(temp[1], "html.parser").get_text().encode("utf-8").decode("utf-8")

        r = random.uniform(0, 1)
        if r < 0.5:
            docs_L.append(doc_1)
            docs_R.append(doc_2)
        else:
            docs_L.append(doc_2)
            docs_R.append(doc_1)

        labels.append(data_panda.sentiment[idx])

    return docs_L, docs_R, labels


#########################################
# count total number of tokens/characters
#########################################
def count_tokens_and_characters(docs_L, docs_R):

    dict_token_counts = {}
    dict_chr_counts = {}

    for doc in docs_L + docs_R:
        for sent in doc:
            tokens = sent.split()
            for token in tokens:
                for chr in token:
                    if chr not in dict_chr_counts:
                        dict_chr_counts[chr] = 0
                    dict_chr_counts[chr] += 1
                if token not in dict_token_counts:
                    dict_token_counts[token] = 0
                dict_token_counts[token] += 1

    return dict_token_counts, dict_chr_counts


###################################
# remove rare tokens and characters
###################################
def remove_rare_tok_chr(dict_tok_counts, dict_chr_counts, min_freq_of_token, min_freq_of_chr):

    # remove rare token types
    list_tokens = list(dict_tok_counts.keys())
    print("# tokens before removing: " + str(len(list_tokens)))
    for token in list_tokens:
        if dict_tok_counts[token] < min_freq_of_token:
            del dict_tok_counts[token]

    # final token type list
    list_tokens = list(dict_tok_counts.keys())
    print("# tokens after removing: " + str(len(list_tokens)))

    # remove rare character types
    list_chr = list(dict_chr_counts.keys())
    print("# characters before removing: " + str(len(list_chr)))
    for c in list_chr:
        if dict_chr_counts[c] < min_freq_of_chr:
            del dict_chr_counts[c]

    # final character type list
    list_chr = list(dict_chr_counts.keys())
    print("# characters after removing: " + str(len(list_chr)))

    return list_tokens, list_chr


############################
# make word-based vocabulary
############################
def make_wrd_chr_vocabularies(list_tokens, list_chr, D_w):

    # token/word-based vocabulary
    V_w = {"<ZP>": 0,  # zero-padding
           "<UNK>": 1,  # unknown token
           "<SOS>": 2,  # start of sentence
           "<EOS>": 3,  # end of sentence
           "<SLB>": 4,  # start with line-break
           "<ELB>": 5,  # end with line-break
           }

    # load pre-trained fastText word embedding model
    WE_dic = fasttext.load_model(os.path.join("..", "data", "cc.en.300.bin"))

    # add tokens to vocabulary and assign an integer
    for token in list_tokens:
        if token not in V_w:
            V_w[token] = len(V_w)

    # word embedding matrix
    E_w = np.zeros(shape=(len(V_w), D_w), dtype="float32")
    r = np.sqrt(3.0 / D_w)
    for token in V_w.keys():
        idx = V_w[token]
        if token in ["<UNK>", "<SOS>", "<EOS>", "<SLB>", "<ELB>"]:
            # initialize special tokens
            E_w[idx, :] = np.random.uniform(low=-r, high=r, size=(1, D_w))
        elif token in ["<ZP>"]:
            # zero-padding token
            continue
        else:
            # initialize pre-trained tokens
            E_w[idx, :] = WE_dic[token]

    # character vocabulary
    V_c = {"<ZP>": 0,
           "<UNK>": 1,
           }

    for c in list_chr:
        if c not in V_c:
            V_c[c] = len(V_c)

    return V_w, E_w, V_c


#################################
# add special tokens to documents
#################################
def add_special_tokens(docs_L, docs_R, T_w):

    ##############################
    # function for single document
    ##############################
    def add_special_tokens_doc(doc):

        ###########
        # add <SOS>
        ###########
        N_w = []
        for i, sent in enumerate(doc):
            tokens = sent.split()
            doc[i] = ["<SOS>"] + tokens
            N_w.append(len(tokens))

        #############################
        # add <EOS> or <ELB> or <SLB>
        #############################
        doc_new = []
        for i, sent in enumerate(doc):
            # short sentence
            if N_w[i] <= T_w - 1:
                tokens = sent + ["<EOS>"]
                doc_new.append(' '.join(tokens))
            # long sentence
            else:
                while len(sent) > 1:
                    if len(sent) <= T_w - 1:
                        tokens = sent[:T_w - 1] + ["<EOS>"]
                        doc_new.append(' '.join(tokens))
                    else:
                        tokens = sent[:T_w - 1] + ["<ELB>"]
                        doc_new.append(' '.join(tokens))
                    sent = ["<SLB>"] + sent[T_w - 1:]

        return doc_new

    ######################################
    # add special tokens for all documents
    ######################################
    for i, doc in enumerate(docs_L):
        doc = add_special_tokens_doc(doc)
        docs_L[i] = doc
    for j, doc in enumerate(docs_R):
        doc = add_special_tokens_doc(doc)
        docs_R[j] = doc

    return docs_L, docs_R

