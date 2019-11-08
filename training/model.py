# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from math import ceil
import os
import pickle
import datetime
from sklearn.utils import shuffle
from feature_extraction import feature_extraction
from initialize_placeholders import initialize_placeholders
from metric_learning import metric_learning
from training import loss_optimizer, compute_labels_euclidean


###########################
# class for Siamese Network
###########################
class SiameseNetwork:
    """
        Main class for Siamese Network Topologies for Authorship Verification (Forensic Text Comparison)
        Two implemented models:
            - HRSN [1]
            - AdHominem [2]

        [1] Benedikt Boenninghoff, Robert M. Nickel, Steffen Zeiler, Dorothea Kolossa, "Similarity Learning for
            Authorship Verification in Social Media", IEEE ICASSP 2019.
        [2] Benedikt Boenninghoff, Steffen Hessler, Dorothea Kolossa, Robert M. Nickel "Explainable Authorship
            Verification in Social Media via Attention-based Similarity Learning", IEEE BigData 2019.

    """

    def __init__(self,
                 hyper_parameters,  # dictionary with all hyper-parameters
                 E_w,  # pre-processed word embedding matrix
                 ):

        # reset graph
        tf.reset_default_graph()

        ############################
        # setup for hyper-parameters
        ############################
        self.hyper_parameters = hyper_parameters

        #########################
        # initialize placeholders
        #########################
        self.placeholders, self.thetas_E = initialize_placeholders(E_w, hyper_parameters)

        ####################
        # feature extraction
        ####################
        self.thetas_feature, self.features = feature_extraction(self.placeholders, self.hyper_parameters)

        ###################################################
        # metric learning (and adversarial domain adaption)
        ###################################################
        self.thetas_metric, self.out = metric_learning(self.placeholders, self.hyper_parameters, self.features)

        ##########
        # training
        ##########
        thetas = (self.thetas_feature, self.thetas_E, self.thetas_metric)
        self.training, self.distance = loss_optimizer(self.placeholders, self.hyper_parameters, self.out, thetas)

        ################
        # launch session
        ################
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    ###########################
    # get current learning rate
    ###########################
    def get_learning_rate(self):

        # get current learning rates
        learning_rate = self.sess.run(self.training["learning_rate"])

        return learning_rate

    ##########################
    # update model "AdHominem"
    ##########################
    def update_adhominem(self, x_w_L, x_w_R, x_c_L, x_c_R, labels, N_w, N_s, B):

        feed_dict = {self.placeholders["x_w_L"]: x_w_L,
                     self.placeholders["x_w_R"]: x_w_R,
                     self.placeholders["x_c_L"]: x_c_L,
                     self.placeholders["x_c_R"]: x_c_R,
                     self.placeholders["labels"]: labels,
                     self.placeholders["N_w"]: N_w,
                     self.placeholders["N_s"]: N_s,
                     self.placeholders["is_training"]: True,
                     self.placeholders["B"]: B,
                     }
        _, loss = self.sess.run([self.training["optimizer"],
                                             self.training["loss"]],
                                             feed_dict=feed_dict,
                                            )

        return loss

    #####################
    # update model "HRSN"
    #####################
    def update_hrsn(self, x_w_L, x_w_R, labels, N_w, N_s, B):

        feed_dict = {self.placeholders["x_w_L"]: x_w_L,
                     self.placeholders["x_w_R"]: x_w_R,
                     self.placeholders["labels"]: labels,
                     self.placeholders["N_w"]: N_w,
                     self.placeholders["N_s"]: N_s,
                     self.placeholders["is_training"]: True,
                     self.placeholders["B"]: B,
                     }
        _, loss = self.sess.run([self.training["optimizer"], self.training["loss"]], feed_dict=feed_dict)

        return loss

    ############################
    # evaluate model "AdHominem"
    ############################
    def evaluate_dev_test_adhominem(self, docs_L, docs_R, labels, batch_size):
        """
            Compute accuracy for test/dev set
        """
        num_batches = ceil(len(labels) / batch_size)

        TP, FP, TN, FN = 0, 0, 0, 0

        for i in range(num_batches):

            # get next batch
            docs_L_i, docs_R_i, labels_i = self.next_batch(i * batch_size,
                                                           (i + 1) * batch_size,
                                                           docs_L,
                                                           docs_R,
                                                           labels,
                                                           )
            B = len(labels_i)

            if B > 0:
                # word/character embeddings
                x_w_L, N_w_L, N_s_L, x_c_L = self.doc2mat(docs_L_i)
                x_w_R, N_w_R, N_s_R, x_c_R = self.doc2mat(docs_R_i)

                # true sentence lengths
                N_w = np.concatenate((N_w_L, N_w_R), axis=1)
                N_s = np.concatenate((N_s_L, N_s_R), axis=1)

                # accuracy for training set
                curr_TP, curr_FP, curr_TN, curr_FN \
                    = self.compute_eval_measures_adhominem(x_w_L=x_w_L,
                                                        x_w_R=x_w_R,
                                                        x_c_L=x_c_L,
                                                        x_c_R=x_c_R,
                                                        labels=np.array(labels_i).reshape((B, 1)),
                                                        N_w=N_w,
                                                        N_s=N_s,
                                                        B=B,
                                                        )

                TP += curr_TP
                FP += curr_FP
                TN += curr_TN
                FN += curr_FN

        acc = self.compute_accuracy(TP, FP, TN, FN)

        return acc

    ###############################################
    # evaluate model "AdHominem" for a single batch
    ###############################################
    def compute_eval_measures_adhominem(self, x_w_L, x_w_R, x_c_L, x_c_R, labels, N_w, N_s, B):
        """
            Compute TP, FP, TN, FN for a single batch
        """
        # compute distances
        distance = self.sess.run(self.distance, feed_dict={self.placeholders["x_w_L"]: x_w_L,
                                                           self.placeholders["x_w_R"]: x_w_R,
                                                           self.placeholders["x_c_L"]: x_c_L,
                                                           self.placeholders["x_c_R"]: x_c_R,
                                                           self.placeholders["labels"]: labels,
                                                           self.placeholders["N_w"]: N_w,
                                                           self.placeholders["N_s"]: N_s,
                                                           #
                                                           self.placeholders["is_training"]: False,
                                                           self.placeholders["B"]: B,
                                                           #
                                                           })

        # execute label computation function
        labels_hat = compute_labels_euclidean(labels, distance, self.hyper_parameters["t_s"], self.hyper_parameters["t_d"])

        # compute values for accuracy, F1-score and c@1
        TP, FP, TN, FN = self.compute_TP_FP_TN_FN(labels, labels_hat)

        return TP, FP, TN, FN

    #######################
    # evaluate model "HRSN"
    #######################
    def evaluate_dev_test_hrsn(self, docs_L, docs_R, labels, batch_size):
        """
            Compute TP, FP, TN, FN for a single batch

        """

        num_batches = ceil(len(labels) / batch_size)

        TP, FP, TN, FN = 0, 0, 0, 0

        for i in range(num_batches):

            # get next batch
            docs_L_i, docs_R_i, labels_i = self.next_batch(i * batch_size,
                                                           (i + 1) * batch_size,
                                                           docs_L,
                                                           docs_R,
                                                           labels,
                                                           )
            B = len(labels_i)

            if B > 0:
                # word/character embeddings
                x_w_L, N_w_L, N_s_L = self.doc2mat_hrsn(docs_L_i)
                x_w_R, N_w_R, N_s_R = self.doc2mat_hrsn(docs_R_i)

                # true sentence lengths
                N_w = np.concatenate((N_w_L, N_w_R), axis=1)
                N_s = np.concatenate((N_s_L, N_s_R), axis=1)

                # accuracy for training set
                curr_TP, curr_FP, curr_TN, curr_FN \
                    = self.compute_eval_measures_hrsn(x_w_L=x_w_L,
                                                      x_w_R=x_w_R,
                                                      labels=np.array(labels_i).reshape((B, 1)),
                                                      N_w=N_w,
                                                      N_s=N_s,
                                                      B=B,
                                                      )

                TP += curr_TP
                FP += curr_FP
                TN += curr_TN
                FN += curr_FN

        acc = self.compute_accuracy(TP, FP, TN, FN)

        return acc

    ##########################################
    # evaluate model "HRSN" for a single batch
    ###########################################
    def compute_eval_measures_hrsn(self, x_w_L, x_w_R, labels, N_w, N_s, B):
        """
            Compute TP, FP, TN, FN for a single batch
        """

        # compute distances
        distance = self.sess.run(self.distance, feed_dict={self.placeholders["x_w_L"]: x_w_L,
                                                           self.placeholders["x_w_R"]: x_w_R,
                                                           self.placeholders["labels"]: labels,
                                                           self.placeholders["N_w"]: N_w,
                                                           self.placeholders["N_s"]: N_s,
                                                           self.placeholders["is_training"]: False,
                                                           self.placeholders["B"]: B,
                                                           })

        # execute label computation function
        labels_hat = compute_labels_euclidean(labels, distance, self.hyper_parameters["t_s"], self.hyper_parameters["t_d"])

        # compute values for accuracy, F1-score and c@1
        TP, FP, TN, FN = self.compute_TP_FP_TN_FN(labels, labels_hat)

        return TP, FP, TN, FN

    ##########################
    # calculate TP, FP, TN, FN
    ##########################
    @staticmethod
    def compute_TP_FP_TN_FN(labels, labels_hat):

        TP, FP, TN, FN = 0, 0, 0, 0

        for i in range(len(labels_hat)):
            if labels[i] == 1 and labels_hat[i] == 1:
                TP += 1
            if labels[i] == 0 and labels_hat[i] == 1:
                FP += 1
            if labels[i] == 0 and labels_hat[i] == 0:
                TN += 1
            if labels[i] == 1 and labels_hat[i] == 0:
                FN += 1

        return TP, FP, TN, FN

    ##################
    # compute accuracy
    ##################
    @staticmethod
    def compute_accuracy(TP, FP, TN, FN):

        acc = (TP + TN) / (TP + FP + TN + FN)

        return acc

    ################
    # get next batch
    ################
    @staticmethod
    def next_batch(t_s, t_e, docs_L, docs_R, labels):
        """
        t_s, t_e:   start, end values to define current batch

        """

        docs_L = docs_L[t_s:t_e]
        docs_R = docs_R[t_s:t_e]
        labels = labels[t_s:t_e]

        return docs_L, docs_R, labels

    #########################################################
    # transform document to a tensor with embeddings for HRSN
    #########################################################
    def doc2mat_hrsn(self, docs):
        """
            docs:   list of documents, where len(docs) = B (batch size)

        """
        T_w = self.hyper_parameters["T_w"]
        T_s = self.hyper_parameters["T_s"]
        V_w = self.hyper_parameters["V_w"]

        # batch size
        B = len(docs)
        N_w = np.zeros((B, 1, T_s), dtype=np.int32)
        N_s = np.zeros((B, 1), dtype=np.int32)

        # word-based tensor, shape = [B, T_s, T_w]
        x_w = np.zeros((B, T_s, T_w), dtype=np.int32)

        # current document
        for i, doc in enumerate(docs):
            N_s[i, :] = len(doc[:T_s])
            # current sentence
            for j, sentence in enumerate(doc[:T_s]):
                N_w[i, :, j] = len(sentence[:T_w])
                # current token
                tokens = sentence.split()
                for k, token in enumerate(tokens[:T_w]):
                    if token in V_w:
                        x_w[i, j, k] = V_w[token]
                    else:
                        x_w[i, j, k] = V_w["<UNK>"]

        return x_w, N_w, N_s

    ##############################################################
    # transform document to a tensor with embeddings for AdHominem
    ##############################################################
    def doc2mat(self, docs):
        """
            docs:   list of documents, where len(docs) = B (batch size)

        """
        T_c = self.hyper_parameters["T_c"]
        T_w = self.hyper_parameters["T_w"]
        T_s = self.hyper_parameters["T_s"]
        V_c = self.hyper_parameters["V_c"]
        V_w = self.hyper_parameters["V_w"]

        # batch size
        B = len(docs)
        N_w = np.zeros((B, 1, T_s), dtype=np.int32)
        N_s = np.zeros((B, 1), dtype=np.int32)

        # word-based tensor, shape = [B, T_s, T_w]
        x_w = np.zeros((B, T_s, T_w), dtype=np.int32)
        # character-based tensor
        x_c = np.zeros((B, T_s, T_w, T_c), dtype=np.int32)

        # current document
        for i, doc in enumerate(docs):
            N_s[i, :] = len(doc[:T_s])
            # current sentence
            for j, sentence in enumerate(doc[:T_s]):
                N_w[i, :, j] = len(sentence[:T_w])
                # current token
                tokens = sentence.split()
                for k, token in enumerate(tokens[:T_w]):
                    if token in V_w:
                        x_w[i, j, k] = V_w[token]
                    else:
                        x_w[i, j, k] = V_w["<UNK>"]
                    # current character
                    for l, chr in enumerate(token[:T_c]):
                        if chr in V_c:
                            x_c[i, j, k, l] = V_c[chr]
                        else:
                            x_c[i, j, k, l] = V_c["<UNK>"]
        return x_w, N_w, N_s, x_c

    ####################################################################################################################
    # train siamese network (HRSN)
    ####################################################################################################################
    def train_model_hrsn(self, train_set, test_set, file_results):
        """
            main function to train HRSN

        """

        # total number of epochs
        epochs = self.hyper_parameters["epochs"]

        # number of batches for dev/test set
        batch_size = self.hyper_parameters["batch_size"]
        batch_size_dev_te = self.hyper_parameters["batch_size_dev_te"]

        # extract train and test sets
        docs_L_tr, docs_R_tr, labels_tr = train_set
        docs_L_te, docs_R_te, labels_te = test_set

        # number of batches for training
        num_batches_tr = ceil(len(labels_tr) / batch_size)

        ################
        # start training
        ################
        for epoch in range(epochs):

            # shuffle data
            docs_L_tr, docs_R_tr, labels_tr = shuffle(docs_L_tr, docs_R_tr, labels_tr)

            # average loss and accuracy
            loss = []
            TP, FP, TN, FN = 0, 0, 0, 0

            # loop over all batches
            for i in range(num_batches_tr):

                # get next batch
                docs_L_i, docs_R_i, labels_i = self.next_batch(i * batch_size,
                                                               (i + 1) * batch_size,
                                                               docs_L_tr,
                                                               docs_R_tr,
                                                               labels_tr,
                                                               )

                # current batch size
                B = len(labels_i)

                if B > 0:

                    # word / character embeddings
                    x_w_L, N_w_L, N_s_L = self.doc2mat_hrsn(docs_L_i)
                    x_w_R, N_w_R, N_s_R = self.doc2mat_hrsn(docs_R_i)

                    # true sentence/document lengths
                    N_w = np.concatenate((N_w_L, N_w_R), axis=1)
                    N_s = np.concatenate((N_s_L, N_s_R), axis=1)

                    # update model parameters
                    curr_loss = self.update_hrsn(x_w_L=x_w_L,
                                                             x_w_R=x_w_R,
                                                             labels=np.array(labels_i).reshape((B, 1)),
                                                             N_w=N_w,
                                                             N_s=N_s,
                                                             B=B,
                                                             )
                    loss.append(curr_loss)

                    # accuracy for training set:
                    curr_TP, curr_FP, curr_TN, curr_FN = \
                        self.compute_eval_measures_hrsn(x_w_L=x_w_L,
                                                        x_w_R=x_w_R,
                                                        labels=np.array(labels_i).reshape((B, 1)),
                                                        N_w=N_w,
                                                        N_s=N_s,
                                                        B=B,
                                                        )
                    TP += curr_TP
                    FP += curr_FP
                    TN += curr_TN
                    FN += curr_FN

                    curr_acc = self.compute_accuracy(curr_TP, curr_FP, curr_TN, curr_FN)

                    s = "epoch:" + str(epoch) \
                        + ", batch: " + str(round(100 * (i + 1) / num_batches_tr, 2)) \
                        + ", Loss: " + str(np.mean(loss)) \
                        + ", acc: " + str(round(100 * (TP + TN) / (TP + FP + TN + FN), 2)) \
                        + ", curr Loss: " + str(round(curr_loss, 2)) \
                        + ", curr Acc: " + str(round(100 * curr_acc, 2)) \
                        + ", lr: " + str(round(float(self.get_learning_rate()), 6))
                    print(s)

            acc_tr = self.compute_accuracy(TP, FP, TN, FN)

            #######################
            # compute test accuracy
            #######################
            acc_te = self.evaluate_dev_test_hrsn(docs_L_te, docs_R_te, labels_te, batch_size_dev_te)

            #####################
            # update progress bar
            #####################
            time = str(datetime.datetime.now()).split(".")[0]
            s = "Time: {:s}, " \
                "Epoch: {:d}, " \
                "Loss: {:.4f}, " \
                "Acc (tr): {:.4f}, " \
                "Acc (te): {:.4f}".format(
                time,
                epoch,
                np.mean(loss),
                round(100 * acc_tr, 4),
                round(100 * acc_te, 4),
            )
            open(file_results, "a").write(s + "\n")

    ####################################################################################################################
    # train siamese network (without data augmentation)
    ####################################################################################################################
    def train_model_adhominem(self, train_set, test_set, file_results):
        """
            main function to train AdHominem

        """

        # total number of epochs
        epochs = self.hyper_parameters["epochs"]

        # number of batches for dev/test set
        batch_size = self.hyper_parameters["batch_size"]
        batch_size_dev_te = self.hyper_parameters["batch_size_dev_te"]

        # extract dev, test sets
        docs_L_tr, docs_R_tr, labels_tr = train_set
        docs_L_te, docs_R_te, labels_te = test_set

        # number of batches for training
        num_batches_tr = ceil(len(labels_tr) / batch_size)

        ################
        # start training
        ################
        for epoch in range(epochs):

            # average loss and accuracy
            loss = []
            TP, FP, TN, FN = 0, 0, 0, 0

            # shuffle training data
            docs_L_tr, docs_R_tr, labels_tr = shuffle(docs_L_tr, docs_R_tr, labels_tr)

            # loop over all batches
            for i in range(num_batches_tr):

                # get next batch
                docs_L_i, docs_R_i, labels_i = self.next_batch(i * batch_size,
                                                               (i + 1) * batch_size,
                                                               docs_L_tr,
                                                               docs_R_tr,
                                                               labels_tr,
                                                               )

                # current batch size
                B = len(labels_i)

                if B > 0:

                    # word / character embeddings
                    x_w_L, N_w_L, N_s_L, x_c_L = self.doc2mat(docs_L_i)
                    x_w_R, N_w_R, N_s_R, x_c_R = self.doc2mat(docs_R_i)

                    # true sentence/document lengths
                    N_w = np.concatenate((N_w_L, N_w_R), axis=1)
                    N_s = np.concatenate((N_s_L, N_s_R), axis=1)

                    # update model parameters
                    curr_loss = self.update_adhominem(x_w_L=x_w_L,
                                                        x_w_R=x_w_R,
                                                        x_c_L=x_c_L,
                                                        x_c_R=x_c_R,
                                                        labels=np.array(labels_i).reshape((B, 1)),
                                                        N_w=N_w,
                                                        N_s=N_s,
                                                        B=B,
                                                        )
                    loss.append(curr_loss)

                    # accuracy for training set:
                    curr_TP, curr_FP, curr_TN, curr_FN = \
                        self.compute_eval_measures_adhominem(x_w_L=x_w_L,
                                                   x_w_R=x_w_R,
                                                   x_c_L=x_c_L,
                                                   x_c_R=x_c_R,
                                                   labels=np.array(labels_i).reshape((B, 1)),
                                                   N_w=N_w,
                                                   N_s=N_s,
                                                   B=B,
                                                   )
                    TP += curr_TP
                    FP += curr_FP
                    TN += curr_TN
                    FN += curr_FN

                    curr_acc = self.compute_accuracy(curr_TP, curr_FP, curr_TN, curr_FN)

                    # print current results
                    s = "epoch:" + str(epoch) \
                        + ", batch: " + str(round(100 * (i + 1) / num_batches_tr, 2)) \
                        + ", Loss: " + str(np.mean(loss)) \
                        + ", acc: " + str(round(100 * (TP + TN) / (TP + FP + TN + FN), 2)) \
                        + ", curr Loss: " + str(round(curr_loss, 2)) \
                        + ", curr Acc: " + str(round(100 * curr_acc, 2)) \
                        + ", lr: " + str(round(float(self.get_learning_rate()), 6))
                    print(s)

            acc_tr = self.compute_accuracy(TP, FP, TN, FN)

            #######################
            # compute test accuracy
            #######################
            acc_te = self.evaluate_dev_test_adhominem(docs_L_te, docs_R_te, labels_te, batch_size_dev_te)

            #####################
            # update progress bar
            #####################
            time = str(datetime.datetime.now()).split(".")[0]
            s = "Time: {:s}, " \
                "Epoch: {:d}, " \
                "Loss: {:.4f}, " \
                "Acc (tr): {:.4f}, " \
                "Acc (te): {:.4f}".format(
                time,
                epoch,
                np.mean(loss),
                round(100 * acc_tr, 4),
                round(100 * acc_te, 4),
            )
            open(file_results, "a").write(s + "\n")