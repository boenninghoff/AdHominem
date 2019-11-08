# -*- coding: utf-8 -*-
import tensorflow as tf


###############
# main function
###############
def initialize_placeholders(E_w_init, hyper_parameters):
    """
        main function to initialize placeholders

    """
    model_type = hyper_parameters["model_type"]

    func = {
        "AdHominem": adhominem,
        "HRSN": hrsn,
    }[model_type]

    with tf.variable_scope("placeholders"):
        placeholders = func(E_w_init, hyper_parameters)

    return placeholders


##################################
# initialize placeholders for HRSN
##################################
def hrsn(E_w_init, hyper_parameters):
    """
    word-level variables:
        x_w_L, x_w_R:   word-based left/right input documents (represented by integers), shape = [B, T_s, T_w]
        e_w_L, e_w_R:   word embeddings of left/right documents, shape = [B, T_s, T_w, D_w]
        N_w             true number of words per sentence, shape = [B, 2, T_s], where N_w[:, 0, :] contains
                        sentence lengths for e_w_L and N_w[:, 1, :] for e_w_R
        N_s:            true number of sentences per document, shape = [B, 2]
        E_w:            word embedding matrix, shape = [len(V_w), D_w]

    training variables:
        labels:         labels, shape = [B, 1]
        is_training:    boolean to assign training or test/dev phase for dropout
        B:              batch size

    """

    T_w = hyper_parameters["T_w"]
    T_s = hyper_parameters["T_s"]
    D_w = hyper_parameters["D_w"]
    train_word_embeddings = hyper_parameters["train_word_embeddings"]

    ######################
    # word-level variables
    ######################
    # word-based placeholder for two documents
    x_w_L = tf.placeholder(dtype=tf.int32, shape=[None, T_s, T_w], name='x_w_L')
    x_w_R = tf.placeholder(dtype=tf.int32, shape=[None, T_s, T_w], name='x_w_R')

    # true sentence / document lengths
    N_w = tf.placeholder(dtype=tf.int32, shape=[None, 2, T_s], name='N_w')
    N_s = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='N_s')

    # matrix for word embeddings, shape=[len(V_w), D_w]
    with tf.variable_scope("word_embedding_matrix"):
        # zero-padding embedding
        E_w_0 = tf.zeros(shape=[1, D_w], dtype=tf.float32)
        # trainable special tokens
        E_w_1 = tf.Variable(E_w_init[1:6, :],
                            name='E_trainable_special_tokens',
                            trainable=True,
                            dtype=tf.float32,
                            )
        # pre-trained word embeddings
        E_w_2 = tf.Variable(E_w_init[6:, :],
                            name='E_pretrained_tokens',
                            trainable=train_word_embeddings,
                            dtype=tf.float32,
                            )
        # concatenate special-token embeddings + regular-token embeddings
        E_w = tf.concat([E_w_0, E_w_1, E_w_2], axis=0)

    # word embeddings, shape=[B, T_s, T_w, D_w]
    e_w_L = tf.nn.embedding_lookup(E_w, x_w_L)
    e_w_R = tf.nn.embedding_lookup(E_w, x_w_R)

    ####################
    # training variables
    ####################
    # labels
    labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')
    # training mode (for dropout regularization)
    is_training = tf.placeholder(dtype=tf.bool, name="training_mode")
    # learning rate for adversarial domain adaption
    B = tf.placeholder(tf.int32, [], name="batch_size")

    #############
    # make tuples
    #############
    placeholders = {"x_w_L": x_w_L,
                    "x_w_R": x_w_R,
                    "e_w_L": e_w_L,
                    "e_w_R": e_w_R,
                    "N_w": N_w,
                    "N_s": N_s,
                    "labels": labels,
                    "is_training": is_training,
                    "B": B,
                    }

    thetas_E = {"E_w": E_w}

    return placeholders, thetas_E


#######################################
# initialize placeholders for AdHominem
#######################################
def adhominem(E_w_init, hyper_parameters):

    """
    character-level variables:
        x_c_L, x_c_R:   character-based left/right input documents (represented by integers),
                        shape = [B, T_s, T_w, T_c]
        e_c_L, e_c_R:   character embeddings of left/right documents, shape = [B, T_s, T_w, T_c, D_w]
        E_c:            character embedding matrix, shape = [len(V_c), D_c]

    word-level variables:
        x_w_L, x_w_R:   word-based left/right input documents (represented by integers), shape = [B, T_s, T_w]
        e_w_L, e_w_R:   word embeddings of left/right documents, shape = [B, T_s, T_w, D_w]
        N_w             true number of words per sentence, shape = [B, 2, T_s], where N_w[:, 0, :] contains
                        sentence lengths for e_w_L and N_w[:, 1, :] for e_w_R
        N_s:            true number of sentences per document, shape = [B, 2]
        E_w:            word embedding matrix, shape = [len(V_w), D_w]

    training variables:
        labels:         labels, shape = [B, 1]
        is_training:    boolean to assign training or test/dev phase for dropout
        B:              batch size

    """

    T_c = hyper_parameters["T_c"]
    T_w = hyper_parameters["T_w"]
    T_s = hyper_parameters["T_s"]
    D_c = hyper_parameters["D_c"]
    D_w = hyper_parameters["D_w"]
    V_c = hyper_parameters["V_c"]
    r = hyper_parameters["r_cnn_emb"]
    train_word_embeddings = hyper_parameters["train_word_embeddings"]

    ###########################
    # character-level variables
    ###########################
    # input character placeholder
    x_c_L = tf.placeholder(dtype=tf.int32,
                           shape=[None, T_s, T_w, T_c],
                           name='x_c_L',
                           )
    x_c_R = tf.placeholder(dtype=tf.int32,
                           shape=[None, T_s, T_w, T_c],
                           name='x_c_R',
                           )
    # initialize embedding matrix for characters
    with tf.variable_scope("character_embedding_matrix"):
        # zero-padding embedding
        E_c_0 = tf.zeros(shape=[1, D_c], dtype=tf.float32)
        # trainable embeddings
        E_c_1 = tf.get_variable(name='E_c_1',
                                shape=[len(V_c) - 1, D_c],
                                initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                trainable=True,
                                dtype=tf.float32,
                                )
        # concatenate special-token embeddings + regular-token embeddings
        E_c = tf.concat([E_c_0, E_c_1], axis=0)

    # character embeddings, shape=[B, T_s, T_w, T_c, D_c]
    e_c_L = tf.nn.embedding_lookup(E_c, x_c_L)
    e_c_R = tf.nn.embedding_lookup(E_c, x_c_R)

    ######################
    # word-level variables
    ######################
    # word-based placeholder for two documents
    x_w_L = tf.placeholder(dtype=tf.int32, shape=[None, T_s, T_w], name='x_w_L')
    x_w_R = tf.placeholder(dtype=tf.int32, shape=[None, T_s, T_w], name='x_w_R')

    # true sentence / document lengths
    N_w = tf.placeholder(dtype=tf.int32, shape=[None, 2, T_s], name='N_w')
    N_s = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='N_s')

    # matrix for word embeddings, shape=[len(V_w), D_w]
    with tf.variable_scope("word_embedding_matrix"):
        # zero-padding embedding
        E_w_0 = tf.zeros(shape=[1, D_w], dtype=tf.float32)
        # trainable special tokens
        E_w_1 = tf.Variable(E_w_init[1:6, :],
                            name='E_trainable_special_tokens',
                            trainable=True,
                            dtype=tf.float32,
                            )
        # pre-trained word embeddings
        E_w_2 = tf.Variable(E_w_init[6:, :],
                            name='E_pretrained_tokens',
                            trainable=train_word_embeddings,
                            dtype=tf.float32,
                            )
        # concatenate special-token embeddings + regular-token embeddings
        E_w = tf.concat([E_w_0, E_w_1, E_w_2], axis=0)

    # word embeddings, shape=[B, T_s, T_w, D_w]
    e_w_L = tf.nn.embedding_lookup(E_w, x_w_L)
    e_w_R = tf.nn.embedding_lookup(E_w, x_w_R)

    ####################
    # training variables
    ####################
    # labels
    labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')

    # training mode (for dropout regularization)
    is_training = tf.placeholder(dtype=tf.bool, name="training_mode")
    # learning rate for adversarial domain adaption
    B = tf.placeholder(tf.int32, [], name="batch_size")

    #############
    # make tuples
    #############
    placeholders = {"x_c_L": x_c_L,
                    "x_c_R": x_c_R,
                    "e_c_L": e_c_L,
                    "e_c_R": e_c_R,
                    "x_w_L": x_w_L,
                    "x_w_R": x_w_R,
                    "e_w_L": e_w_L,
                    "e_w_R": e_w_R,
                    "N_w": N_w,
                    "N_s": N_s,
                    "labels": labels,
                    "is_training": is_training,
                    "B": B,
                    }

    thetas_E = {"E_c": E_c,
               "E_w": E_w,
               }

    return placeholders, thetas_E
