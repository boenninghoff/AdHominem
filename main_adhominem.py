# -*- coding: utf-8 -*-
from adhominem import AdHominem
import pickle
import os
import argparse


def main():

    parser = argparse.ArgumentParser(description='AdHominem - Siamese Network for Authorship Verification')
    parser.add_argument('-D_c', default=10, type=int)  # character embedding dimension
    parser.add_argument('-D_r', default=30, type=int)  # character representation dimension
    parser.add_argument('-w', default=4, type=int)  # length of 1D-CNN sliding window
    parser.add_argument('-D_w', default=300, type=int)  # dimension of word embeddings
    parser.add_argument('-D_s', default=50, type=int)  # dimension sentence embeddings
    parser.add_argument('-D_d', default=50, type=int)  # dimension of document embedding
    parser.add_argument('-D_mlp', default=60, type=int)  # final output dimension
    parser.add_argument('-T_c', default=15, type=int)  # maximum number of characters per words
    parser.add_argument('-T_w', default=20, type=int)  # maximum number of words per sentence
    parser.add_argument('-T_s', default=50, type=int)  # maximum number of sentences per document
    parser.add_argument('-t_s', default=0.99, type=float)  # boundary for similar pairs (close to one)
    parser.add_argument('-t_d', default=0.01, type=float)  # boundary for dissimilar pairs (close to zero)
    parser.add_argument('-epochs', default=100, type=int)  # total number of epochs
    parser.add_argument('-train_word_embeddings', default=False, type=bool)  # fine-tune pre-trained word embeddings
    parser.add_argument('-batch_size_tr', default=32, type=int)  # batch size for training
    parser.add_argument('-batch_size_te', default=128, type=int)  # batch size for evaluation
    parser.add_argument('-initial_learning_rate', default=0.002, type=float)  # initial learning rate
    parser.add_argument('-keep_prob_cnn', default=0.8, type=float)  # dropout for 1D-CNN layer
    parser.add_argument('-keep_prob_lstm', default=0.8, type=float)  # variational dropout for BiLSTM layer
    parser.add_argument('-keep_prob_att', default=0.8, type=float)  # dropout for attention layer
    parser.add_argument('-keep_prob_metric', default=0.8, type=float)  # dropout for metric learning layer
    hyper_parameters = vars(parser.parse_args())

    # create folder for results
    dir_results = os.path.join('results')
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)

    # load docs, vocabularies and initialized word embeddings
    with open(os.path.join("data", "data_Amazon_9000"), 'rb') as f:
        docs_L_tr, docs_R_tr, labels_tr, \
        docs_L_te, docs_R_te, labels_te, \
        V_w, E_w, V_c = pickle.load(f)

    # add vocabularies to dictionary
    hyper_parameters['V_w'] = V_w
    hyper_parameters['V_c'] = V_c
    hyper_parameters['N_tr'] = len(labels_tr)
    hyper_parameters['N_dev'] = len(labels_te)

    # file to store results epoch-wise
    file_results = os.path.join(dir_results, 'results.txt')

    # delete already existing files
    if os.path.isfile(file_results):
        os.remove(file_results)

    # write hyper-parameters setup into file (results.txt)
    open(file_results, 'a').write('\n'
                                   + '--------------------------------------------------------------------------------'
                                   + '\nPARAMETER SETUP:\n'
                                   + '--------------------------------------------------------------------------------'
                                   + '\n'
                                   )
    for hp in hyper_parameters.keys():
        if hp in ['V_c', 'V_w']:
            open(file_results, 'a').write('num ' + hp + ': ' + str(len(hyper_parameters[hp])) + '\n')
        else:
            open(file_results, 'a').write(hp + ': ' + str(hyper_parameters[hp]) + '\n')

    # load neural network model
    adhominem = AdHominem(hyper_parameters=hyper_parameters,
                          E_w_init=E_w,
                          )
    # start training
    train_set = (docs_L_tr, docs_R_tr, labels_tr)
    test_set = (docs_L_te, docs_R_te, labels_te)
    adhominem.train_model(train_set, test_set, file_results)

    # close session
    adhominem.sess.close()


if __name__ == '__main__':
    main()
