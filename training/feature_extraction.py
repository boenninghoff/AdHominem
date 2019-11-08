# -*- coding: utf-8 -*-
import tensorflow as tf


######################################
# main function for feature extraction
######################################
def feature_extraction(placeholders, hyper_parameters):
    """
        main function for feature extraction

    """
    ################################
    # initialize trainable variables
    ################################
    model_type = hyper_parameters["model_type"]
    func = {
        "AdHominem": initialize_adhominem,
        "HRSN": initialize_hrsn,
    }[model_type]
    with tf.variable_scope("initialize_feature_extraction"):
        thetas = func(hyper_parameters)

    #################
    # prepare dropout
    #################
    func = {
        "AdHominem": dropout_adhominem,
        "HRSN": dropout_hrsn,
    }[model_type]
    with tf.variable_scope("dropout_feature_extraction"):
        dropout = func(hyper_parameters, placeholders["B"])

    ##################
    # extract features
    ##################
    func = {
        "AdHominem": features_adhominem,
        "HRSN": features_hrsn,
    }[model_type]
    with tf.variable_scope("apply_feature_extraction"):
        features = func(hyper_parameters, placeholders, dropout, thetas)

    return thetas, features


###########################################
# initialize (domain-adversarial) AdHominem
###########################################
def initialize_adhominem(hyper_parameters):
    """
    Trainable parameters:
        theta_cnn:          characters-to-word encoding
        theta_rnn_ws_f:     forward words-to-sentence encoding
        theta_rnn_ws_b:     backward words-to-sentence encoding
        theta_rnn_sd_f:     forward sentences-to-document encoding
        theta_rnn_sd_b:     backward sentences-to-document encoding
        theta_att_ws:       attentions layer for words-to-sentence encoding
        theta_att_sd:       attentions layer for sentences-to-document encoding

    """

    D_r = hyper_parameters["D_r"]
    D_w = hyper_parameters["D_w"]
    D_s = hyper_parameters["D_s"]
    D_d = hyper_parameters["D_d"]

    # CNN layer for characters-to-word encoding
    with tf.variable_scope("layer_cnn"):
        theta_cnn = initialize_cnn(hyper_parameters)

    # forward-LSTM layer for words-to-sentence / sentences-to-document encoding
    with tf.variable_scope("layer_lstm_f"):
        theta_rnn_ws_f, theta_rnn_sd_f = initialize_rnn(D_in_ws=D_w + D_r,
                                                        D_out_ws=D_s,
                                                        D_in_sd=2 * D_s,
                                                        D_out_sd=D_d,
                                                        hyper_parameters=hyper_parameters,
                                                        )

    # backward-LSTM layer for words-to-sentence / sentences-to-document encoding
    with tf.variable_scope("layer_rnn_b"):
        theta_rnn_ws_b, theta_rnn_sd_b = initialize_rnn(D_in_ws=D_w + D_r,
                                                        D_out_ws=D_s,
                                                        D_in_sd=2 * D_s,
                                                        D_out_sd=D_d,
                                                        hyper_parameters=hyper_parameters,
                                                        )

    # parameters for attention layers
    with tf.variable_scope("layer_att"):
        theta_att_ws, theta_att_sd = initialize_att(hyper_parameters)

    thetas = {"theta_cnn": theta_cnn,
              "theta_rnn_ws_f": theta_rnn_ws_f,
              "theta_rnn_sd_f": theta_rnn_sd_f,
              "theta_rnn_ws_b": theta_rnn_ws_b,
              "theta_rnn_sd_b": theta_rnn_sd_b,
              "theta_att_ws": theta_att_ws,
              "theta_att_sd": theta_att_sd,
              }

    return thetas


#################
# initialize HRSN
#################
def initialize_hrsn(hyper_parameters):
    """
    Trainable parameters:
        theta_rnn_ws:     (forward) words-to-sentence encoding
        theta_rnn_sd:     (forward) sentences-to-document encoding

    """

    D_w = hyper_parameters["D_w"]
    D_s = hyper_parameters["D_s"]
    D_d = hyper_parameters["D_d"]

    # forward-LSTM layer for words-to-sentence / sentences-to-document encoding
    with tf.variable_scope("layer_lstm"):
        theta_rnn_ws, theta_rnn_sd = initialize_rnn(D_in_ws=D_w,
                                                    D_out_ws=D_s,
                                                    D_in_sd=D_s,
                                                    D_out_sd=D_d,
                                                    hyper_parameters=hyper_parameters,
                                                    )

    thetas = {"theta_rnn_ws": theta_rnn_ws,
              "theta_rnn_sd": theta_rnn_sd,
              }

    return thetas


##################
# dropout function
##################
def make_dropout_mask(shape, keep_prob):
    """
    Variational dropout [1] for RNNs (also used for "normal" dropout)

    [1] Yarin Gal, Ghahramani Zoubin, "A theoretically grounded application of dropout in recurrent neural
        networks", NIPS, 2016.

    """
    keep_prob = tf.convert_to_tensor(keep_prob)
    random_tensor = keep_prob + tf.random_uniform(shape)
    binary_tensor = tf.floor(random_tensor)
    dropout_mask = tf.divide(binary_tensor, keep_prob)

    return dropout_mask


##################
# dropout for HRSN
##################
def dropout_hrsn(hyper_parameters, B):

    T_s = hyper_parameters["T_s"]
    D_w = hyper_parameters["D_w"]
    D_s = hyper_parameters["D_s"]
    D_d = hyper_parameters["D_d"]

    with tf.variable_scope("dropout_rnn_f"):
        keep_prob = hyper_parameters["keep_prob_rnn"]
        dropout_rnn_x_w = make_dropout_mask(shape=[B * T_s, D_w], keep_prob=keep_prob)
        dropout_rnn_h_w = make_dropout_mask(shape=[B * T_s, D_s], keep_prob=keep_prob)
        dropout_rnn_x_s = make_dropout_mask(shape=[B, D_s], keep_prob=keep_prob)
        dropout_rnn_h_s = make_dropout_mask(shape=[B, D_d], keep_prob=keep_prob)

    dropout = {"rnn_x_w": dropout_rnn_x_w,
               "rnn_h_w": dropout_rnn_h_w,
               "rnn_x_s": dropout_rnn_x_s,
               "rnn_h_s": dropout_rnn_h_s,
               }

    return dropout


#########################################
# dropout for adversarial domain adaption
#########################################
def dropout_adhominem(hyper_parameters, B):

    T_c = hyper_parameters["T_c"]
    T_w = hyper_parameters["T_w"]
    T_s = hyper_parameters["T_s"]

    D_c = hyper_parameters["D_c"]
    D_r = hyper_parameters["D_r"]
    D_w = hyper_parameters["D_w"]
    D_s = hyper_parameters["D_s"]
    D_d = hyper_parameters["D_d"]
    D_a_ws = hyper_parameters["D_a_ws"]
    D_a_sd = hyper_parameters["D_a_sd"]

    with tf.variable_scope("dropout_cnn"):
        keep_prob = hyper_parameters["keep_prob_cnn"]
        dropout_cnn = make_dropout_mask(shape=[B * T_s * T_w, T_c, D_c], keep_prob=keep_prob)

    with tf.variable_scope("dropout_rnn_f"):
        keep_prob = hyper_parameters["keep_prob_rnn"]
        dropout_rnn_x_w_f = make_dropout_mask(shape=[B * T_s, D_w + D_r], keep_prob=keep_prob)
        dropout_rnn_h_w_f = make_dropout_mask(shape=[B * T_s, D_s], keep_prob=keep_prob)
        dropout_rnn_x_s_f = make_dropout_mask(shape=[B, 2 * D_s], keep_prob=keep_prob)
        dropout_rnn_h_s_f = make_dropout_mask(shape=[B, D_d], keep_prob=keep_prob)

    with tf.variable_scope("dropout_rnn_b"):
        keep_prob = hyper_parameters["keep_prob_rnn"]
        dropout_rnn_x_w_b = make_dropout_mask(shape=[B * T_s, D_w + D_r], keep_prob=keep_prob)
        dropout_rnn_h_w_b = make_dropout_mask(shape=[B * T_s, D_s], keep_prob=keep_prob)
        dropout_rnn_x_s_b = make_dropout_mask(shape=[B, 2 * D_s], keep_prob=keep_prob)
        dropout_rnn_h_s_b = make_dropout_mask(shape=[B, D_d], keep_prob=keep_prob)

    with tf.variable_scope("dropout_att"):
        keep_prob = hyper_parameters["keep_prob_att"]
        dropout_att_wb_w = make_dropout_mask(shape=[B * T_s, 2 * D_s], keep_prob=keep_prob)
        dropout_att_v_w = make_dropout_mask(shape=[B * T_s, D_a_ws], keep_prob=keep_prob)
        dropout_att_wb_s = make_dropout_mask(shape=[B, 2 * D_d], keep_prob=keep_prob)
        dropout_att_v_s = make_dropout_mask(shape=[B, D_a_sd], keep_prob=keep_prob)

    dropout = {"cnn": dropout_cnn,
               "rnn_x_w_f": dropout_rnn_x_w_f,
               "rnn_h_w_f": dropout_rnn_h_w_f,
               "rnn_x_s_f": dropout_rnn_x_s_f,
               "rnn_h_s_f": dropout_rnn_h_s_f,
               "rnn_x_w_b": dropout_rnn_x_w_b,
               "rnn_h_w_b": dropout_rnn_h_w_b,
               "rnn_x_s_b": dropout_rnn_x_s_b,
               "rnn_h_s_b": dropout_rnn_h_s_b,
               "att_wb_w": dropout_att_wb_w,
               "att_v_w": dropout_att_v_w,
               "att_wb_s": dropout_att_wb_s,
               "att_v_s": dropout_att_v_s,
               }

    return dropout


##################################
# feature extraction for AdHominem
##################################
def features_adhominem(hyper_parameters, placeholders, dropout, thetas):

    """
    Build Siamese network to encode a document into a feature vector

    Input:
        e_c_L, e_c_R:   character embeddings
        e_w_L, e_w_R:   word embeddings

    Output:
        r_c_L, r_c_R:   character representations
        e_s_L, e_s_R:   sentence embeddings
        e_d_L, e_d_R:   document embeddings
        y_L, y_R:       neural features, shape = [B, D_mlp]

    """

    T_c = hyper_parameters["T_c"]
    T_w = hyper_parameters["T_w"]
    T_s = hyper_parameters["T_s"]

    D_c = hyper_parameters["D_c"]
    D_r = hyper_parameters["D_r"]
    D_w = hyper_parameters["D_w"]
    D_s = hyper_parameters["D_s"]
    D_d = hyper_parameters["D_d"]

    B = placeholders["B"]
    h = hyper_parameters["h"]
    is_training = placeholders["is_training"]

    with tf.variable_scope("feature_extraction") as scope:

        ######################
        # first neural network
        ######################
        # character encoding
        r_c_L = compute_chr_representations(placeholders["e_c_L"],
                                            T_s, T_w, T_c, D_c, D_r, h, is_training,
                                            dropout["cnn"],
                                            thetas["theta_cnn"],
                                            B,
                                            )
        e_cw_L = tf.concat([placeholders["e_w_L"], r_c_L], axis=3)

        # word-to-sentence encoding
        e_s_L, prob_ws_L = BiLSTM_ATT_ws(e_cw_L,  placeholders["N_w"][:, 0, :],
                                     T_w, T_s, D_r, D_w, D_s,
                                     thetas["theta_rnn_ws_f"], thetas["theta_rnn_ws_b"], thetas["theta_att_ws"],
                                     dropout["rnn_x_w_f"], dropout["rnn_h_w_f"],
                                     dropout["rnn_x_w_b"], dropout["rnn_h_w_b"],
                                     dropout["att_wb_w"], dropout["att_v_w"],
                                     B, is_training,
                                     )

        # sentence-to-document encoding
        e_d_L, prob_sd_L = BiLSTM_ATT_sd(e_s_L, placeholders["N_s"][:, 0],
                                         T_s, D_d,
                                         thetas["theta_rnn_sd_f"], thetas["theta_rnn_sd_b"], thetas["theta_att_sd"],
                                         dropout["rnn_x_s_f"], dropout["rnn_h_s_f"],
                                         dropout["rnn_x_s_b"], dropout["rnn_h_s_b"],
                                         dropout["att_wb_s"], dropout["att_v_s"],
                                         B, is_training,
                                         )

        #######################
        # second neural network
        #######################
        scope.reuse_variables()
        # character encoding
        r_c_R = compute_chr_representations(placeholders["e_c_R"],
                                            T_s, T_w, T_c, D_c, D_r, h, is_training,
                                            dropout["cnn"],
                                            thetas["theta_cnn"],
                                            B,
                                            )
        e_cw_R = tf.concat([placeholders["e_w_R"], r_c_R], axis=3)

        # word-to-sentence encoding
        e_s_R, prob_ws_R = BiLSTM_ATT_ws(e_cw_R, placeholders["N_w"][:, 1, :],
                                         T_w, T_s, D_r, D_w, D_s,
                                         thetas["theta_rnn_ws_f"], thetas["theta_rnn_ws_b"], thetas["theta_att_ws"],
                                         dropout["rnn_x_w_f"], dropout["rnn_h_w_f"],
                                         dropout["rnn_x_w_b"], dropout["rnn_h_w_b"],
                                         dropout["att_wb_w"], dropout["att_v_w"],
                                         B, is_training,
                                         )

        # sentence-to-document encoding
        e_d_R, prob_sd_R = BiLSTM_ATT_sd(e_s_R, placeholders["N_s"][:, 1],
                                         T_s, D_d,
                                         thetas["theta_rnn_sd_f"], thetas["theta_rnn_sd_b"], thetas["theta_att_sd"],
                                         dropout["rnn_x_s_f"], dropout["rnn_h_s_f"],
                                         dropout["rnn_x_s_b"], dropout["rnn_h_s_b"],
                                         dropout["att_wb_s"], dropout["att_v_s"],
                                         B, is_training,
                                         )

    features = {"e_d_L": e_d_L,
                "e_d_R": e_d_R,
                "prob_ws_L": prob_ws_L,
                "prob_ws_R": prob_ws_R,
                "prob_sd_L": prob_sd_L,
                "prob_sd_R": prob_sd_R,
                }

    return features


#############################
# feature extraction for HRSN
#############################
def features_hrsn(hyper_parameters, placeholders, dropout, thetas):
    """
        compute features for HRSN

    """
    T_w = hyper_parameters["T_w"]
    T_s = hyper_parameters["T_s"]

    D_w = hyper_parameters["D_w"]
    D_s = hyper_parameters["D_s"]
    D_d = hyper_parameters["D_d"]

    B = placeholders["B"]
    is_training = placeholders["is_training"]

    with tf.variable_scope("feature_extraction") as scope:

        ######################
        # first neural network
        ######################
        # word-to-sentence encoding
        e_s_L = LSTM_ws(placeholders["e_w_L"],
                        placeholders["N_w"][:, 0, :],
                        T_w, T_s, D_w, D_s,
                        thetas["theta_rnn_ws"],
                        dropout["rnn_x_w"], dropout["rnn_h_w"],
                        B,
                        is_training,
                        )

        # sentence-to-document encoding
        e_d_L = LSTM_sd(e_s_L,
                        placeholders["N_s"][:, 0],
                        T_s, D_d,
                        thetas["theta_rnn_sd"],
                        dropout["rnn_x_s"], dropout["rnn_h_s"],
                        B, is_training,
                        )

        #######################
        # second neural network
        #######################
        scope.reuse_variables()
        # word-to-sentence encoding
        e_s_R = LSTM_ws(placeholders["e_w_R"],
                        placeholders["N_w"][:, 1, :],
                        T_w, T_s, D_w, D_s,
                        thetas["theta_rnn_ws"],
                        dropout["rnn_x_w"], dropout["rnn_h_w"],
                        B,
                        is_training,
                        )

        # sentence-to-document encoding
        e_d_R = LSTM_sd(e_s_R,
                        placeholders["N_s"][:, 1],
                        T_s, D_d,
                        thetas["theta_rnn_sd"],
                        dropout["rnn_x_s"], dropout["rnn_h_s"],
                        B, is_training,
                        )

    features = {"e_d_L": e_d_L,
                "e_d_R": e_d_R,
                }

    return features


################################################
# initialize CNN for characters-to-word encoding
################################################
def initialize_cnn(hyper_parameters):
    """
        W: shape = [window_length, D_c, D_r]
        b: shape = [D_r, 1]

    """
    # boundary value for random uniform initialization
    r = hyper_parameters["r_cnn_W"]
    # input/output dimensions
    D_c = hyper_parameters["D_c"]
    D_r = hyper_parameters["D_r"]
    # sliding window size
    h = hyper_parameters["h"]

    ##############################
    # initialize variables for CNN
    ##############################
    theta = {"W": tf.get_variable(name="W_cnn",
                                  shape=[h, D_c, D_r],
                                  initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                  dtype=tf.float32),
             "b": tf.get_variable(name="b_cnn",
                                  shape=[D_r],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32),
             }

    return theta


##########################################################################
# initialize RNNs for words-to-sentence and sentences-to-document encoding
##########################################################################
def initialize_rnn(D_in_ws, D_out_ws, D_in_sd, D_out_sd, hyper_parameters):
    """
    Choose RNN cell and initialize trainable and shared parameters (weight matrices and bias vectors)

    theta_ws:   trainable parameters for word-to-sentence encoding
    theta_sd:   trainable parameters for sentence-to-document encoding

    """

    with tf.variable_scope("word_to_sentence"):
        theta_ws = initialize_LSTM(D_in_ws, D_out_ws,
                                   hyper_parameters["r_rnn_ws_W"],
                                   hyper_parameters["r_rnn_ws_U"],
                                   bf=hyper_parameters["bf_init"],
                                   )
    with tf.variable_scope("sentence_to_document"):
        theta_sd = initialize_LSTM(D_in_sd, D_out_sd,
                                   hyper_parameters["r_rnn_sd_W"],
                                   hyper_parameters["r_rnn_sd_U"],
                                   bf=hyper_parameters["bf_init"],
                                   )

    return theta_ws, theta_sd


######################
# initialize LSTM cell
######################
def initialize_LSTM(D_in, D_out, rW, rU, bf):
    """
    shared parameters for single LSTM cell

    """

    theta = {"W_i": tf.get_variable('W_i', shape=[D_in, D_out],
                                    initializer=tf.initializers.random_uniform(minval=-rW, maxval=rW),
                                    trainable=True),
             "U_i": tf.get_variable('U_i', shape=[D_out, D_out],
                                    initializer=tf.initializers.random_uniform(minval=-rU, maxval=rU),
                                    trainable=True),
             "b_i": tf.get_variable('b_i', shape=[1, D_out],
                                    initializer=tf.constant_initializer(0.0), trainable=True),
             "W_f": tf.get_variable('W_f', shape=[D_in, D_out],
                                    initializer=tf.initializers.random_uniform(minval=-rW, maxval=rW),
                                    trainable=True),
             "U_f": tf.get_variable('U_f', shape=[D_out, D_out],
                                    initializer=tf.initializers.random_uniform(minval=-rU, maxval=rU),
                                    trainable=True),
             "b_f": tf.get_variable('b_f', shape=[1, D_out],
                                    initializer=tf.constant_initializer(bf), trainable=True),
             "W_c": tf.get_variable('W_c', shape=[D_in, D_out],
                                    initializer=tf.initializers.random_uniform(minval=-rW, maxval=rW),
                                    trainable=True),
             "U_c": tf.get_variable('U_c', shape=[D_out, D_out],
                                    initializer=tf.initializers.random_uniform(minval=-rU, maxval=rU),
                                    trainable=True),
             "b_c": tf.get_variable('b_c', shape=[1, D_out],
                                    initializer=tf.constant_initializer(0.0), trainable=True),
             "W_o": tf.get_variable('W_o', shape=[D_in, D_out],
                                    initializer=tf.initializers.random_uniform(minval=-rW, maxval=rW),
                                    trainable=True),
             "U_o": tf.get_variable('U_o', shape=[D_out, D_out],
                                    initializer=tf.initializers.random_uniform(minval=-rU, maxval=rU),
                                    trainable=True),
             "b_o": tf.get_variable('b_o', shape=[1, D_out],
                                    initializer=tf.constant_initializer(0.0), trainable=True),
             }

    return theta


############################
# initialize attention layer
############################
def initialize_att(hyper_parameters):
    """
    Initialize attention layer for
        theta_ws:   word-to-sentence encoding
        theta_sd:   sentence-to-document encoding

    """

    D_s = hyper_parameters["D_s"]
    D_a_ws = hyper_parameters["D_a_ws"]
    D_d = hyper_parameters["D_d"]
    D_a_sd = hyper_parameters["D_a_sd"]

    with tf.variable_scope("attention_ws"):
        theta_ws = initialize_single_att(2 * D_s, D_a_ws,
                                         hyper_parameters["r_att_ws_W"],
                                         hyper_parameters["r_att_ws_u"],
                                         )
    with tf.variable_scope("attention_sd"):
        theta_sd = initialize_single_att(2 * D_d, D_a_sd,
                                         hyper_parameters["r_att_sd_W"],
                                         hyper_parameters["r_att_sd_u"],
                                         )

    return theta_ws, theta_sd


def initialize_single_att(D_in, D_out, rW, ru):

    theta = {"W_a": tf.get_variable('W_a',
                                    shape=[D_in, D_out],
                                    initializer=tf.initializers.random_uniform(minval=-rW, maxval=rW),
                                    trainable=True),
             "v_a": tf.get_variable('v_a', shape=[D_out, 1],
                                    initializer=tf.initializers.random_uniform(minval=-ru, maxval=ru),
                                    trainable=True),
             "b_a": tf.get_variable('b_a', shape=[D_out],
                                    initializer=tf.constant_initializer(0.0),
                                    trainable=True),
             }

    return theta


#############################
# characters-to-word encoding
#############################
def compute_chr_representations(e_c, T_s, T_w, T_c, D_c, D_r, h, is_training, dropout_mask, theta, B):
    """
    input:
        e_c:    character embeddings, shape = [B, T_s, T_w, T_c, D_c]

    output:
        r_c:    character representations, shape = [B, T_s, T_w, D_r]

    """

    ##########################
    # dropout and zero-padding
    ##########################
    # reshape: [B, T_s, T_w, T_c, D_c] --> [B * T_s * T_w, T_c, D_c]
    e_c = tf.reshape(e_c, shape=[B * T_s * T_w, T_c, D_c])
    # dropout
    e_c = tf.cond(tf.equal(is_training, tf.constant(True)),
                  lambda: tf.multiply(dropout_mask, e_c),
                  lambda: e_c,
                  )
    # zero-padding, shape = [B * T_s * T_w, T_c + 2 * (h-1), D_c]
    e_c = tf.pad(e_c,
                 tf.constant([[0, 0], [h - 1, h - 1], [0, 0]]),
                 mode='CONSTANT',
                 )

    ################
    # 1D convolution
    ################
    # shape = [B * T_s * T_w, T_c + 2 * (h-1) - h + 1, D_r] = [B * T_s * T_w, T_c + h - 1, D_r]
    r_c = tf.nn.conv1d(e_c,
                       theta["W"],
                       stride=1,
                       padding="VALID",
                       name="chraracter_1D_cnn",
                       )
    # apply bias term
    r_c = tf.nn.bias_add(r_c, theta["b"])
    # apply nonlinear function
    r_c = tf.nn.tanh(r_c)

    #######################
    # max-over-time pooling
    #######################
    # shape = [B * T_s * T_w, T_c + h - 1, D_r, 1]
    r_c = tf.expand_dims(r_c, -1)
    # max-over-time-pooling, shape = [B * T_s * T_w, 1, D_r, 1]
    r_c = tf.nn.max_pool(r_c,
                         ksize=[1, T_c + h - 1, 1, 1],
                         strides=[1, 1, 1, 1],
                         padding='VALID',
                         )
    # shape = [B * T_s * T_w, D_r]
    r_c = tf.squeeze(r_c)
    #  shape = [B, T_s, T_w, D_r]
    r_c = tf.reshape(r_c, [B, T_s, T_w, D_r])

    return r_c


#########################
# compute attention score
#########################
def compute_attention_score(h, theta, dropout_wb, dropout_u, is_training):
    """
    Compute attention score for single hidden state

    Input:
        h:      hidden states at time t, shape = [B * T_s, 2 * D_s] for word-to-sentence encoding
                or [B, 2 * D_d] for sentence-to-document encoding
    Output:
        score:  un-normalized scoring value, shape = [B * T_s, 1] or [B, 1]

    """
    # apply dropout
    h = tf.cond(tf.equal(is_training, tf.constant(True)),
                lambda: tf.multiply(dropout_wb, h),
                lambda: h,
                )
    # word attention, shape=[B * T_s, D_a_w] or [B, D_a_s]
    score = tf.nn.tanh(tf.nn.xw_plus_b(h, theta["W_a"], theta["b_a"]))

    # apply dropout
    score = tf.cond(tf.equal(is_training, tf.constant(True)),
                    lambda: tf.multiply(dropout_u, score),
                    lambda: score,
                    )

    # shape=[B * T_s, 1] or [B, 1]
    score = tf.matmul(score, theta["v_a"])

    return score


##################
# single LSTM cell
##################
def LSTM_cell(h_prev, c_prev, x_t, theta, dropout_x, dropout_h, is_training):
    """
    single forward step of the LSTM cell

    hc_prev:    hidden, memory states at timestep "t-1", shape = [B, D_out, 2]
    e_t:        input data at timestep "t", shape = [B, D_in]
    hc_next:    next hidden, memory states, shape = [B, D_out, 2]

    where
    D_in:       dimension of input embedding vector
    D_out:      dimension of output embedding vector


    """

    # apply dropout
    x_t = tf.cond(tf.equal(is_training, tf.constant(True)),
                    lambda: tf.multiply(dropout_x, x_t),
                    lambda: x_t,
                    )
    h_prev = tf.cond(tf.equal(is_training, tf.constant(True)),
                       lambda: tf.multiply(dropout_h, h_prev),
                       lambda: h_prev,
                       )
    # forget
    i_t = tf.sigmoid(tf.matmul(x_t, theta["W_i"]) + tf.matmul(h_prev, theta["U_i"]) + theta["b_i"])
    # input
    f_t = tf.sigmoid(tf.matmul(x_t, theta["W_f"]) + tf.matmul(h_prev, theta["U_f"]) + theta["b_f"])
    # new memeory state
    c_tilde = tf.tanh(tf.matmul(x_t, theta["W_c"]) + tf.matmul(h_prev, theta["U_c"]) + theta["b_c"])
    # final memeory state
    c_next = tf.multiply(i_t, c_tilde) + tf.multiply(f_t, c_prev)
    # output
    o_t = tf.sigmoid(tf.matmul(x_t, theta["W_o"]) + tf.matmul(h_prev, theta["U_o"]) + theta["b_o"])
    # next hidden state
    h_next = tf.multiply(o_t, tf.tanh(c_next))

    return h_next, c_next


###########################
# word-to-sentence encoding
###########################
def BiLSTM_ATT_ws(e_w_f, N_w, T_w, T_s, D_r, D_w, D_s,
                  theta_f, theta_b, theta_att,
                  dropout_x_f, dropout_h_f, dropout_x_b, dropout_h_b,
                  dropout_att_wb, dropout_att_v,
                  B, is_training,
                  ):
    """
    Word-to-sentence encoding: word-embeddings to sentence-embeddings forward propagation

    Input:
        e_w:        word embeddings, shape = [B, T_s, T_w, D_w]
        N_w:        tensor for true number of words per sentence, shape = [B, T_s]

    Output:
        e_s:        sentence embeddings, shape = [B, T_s, 2 * D_s]
        prob_ws:    word-based attentions (pseudo-probabilities), shape = [B, T_s, T_w]

    """

    # extract dimensions to initialize state(s)
    dims = tf.stack([B * T_s, D_s])

    # reshape N_w, shape = [B * T_s]
    N_w = tf.reshape(N_w, shape=[B * T_s])

    # reshape input word embeddings, shape = [B * T_s, T_w, D_w]
    e_w_f = tf.reshape(e_w_f, shape=[B * T_s, T_w, D_w + D_r])
    # reverse input sentences
    e_w_b = tf.reverse_sequence(e_w_f, seq_lengths=N_w, seq_axis=1)

    # initialize state(s) with zeros
    h_prev_f = tf.fill(dims, 0.0)
    h_prev_b = tf.fill(dims, 0.0)
    c_prev_f = tf.fill(dims, 0.0)
    c_prev_b = tf.fill(dims, 0.0)

    # store all hidden states
    h_f = []
    h_b = []

    ##########################################
    # compute forward / backward hidden states
    ###########################################
    for t_w in range(T_w):

        # shapes = [B * T_s, D_s, 2]
        h_next_f, c_next_f = LSTM_cell(h_prev_f, c_prev_f,
                                       e_w_f[:, t_w, :],
                                       theta_f,
                                       dropout_x_f, dropout_h_f,
                                       is_training,
                                       )
        h_next_b, c_next_b = LSTM_cell(h_prev_b, c_prev_b,
                                       e_w_b[:, t_w, :],
                                       theta_b,
                                       dropout_x_b, dropout_h_b,
                                       is_training,
                                       )

        # t < T
        condition = tf.less(t_w, N_w)

        # copy-through states if t > T
        h_prev_f = tf.where(condition, h_next_f, h_prev_f)
        c_prev_f = tf.where(condition, c_next_f, c_prev_f)
        h_prev_b = tf.where(condition, h_next_b, h_prev_b)
        c_prev_b = tf.where(condition, c_next_b, c_prev_b)

        # store states, list of T_w tensors with shape=[B * T_s, D_s]
        h_f.append(h_prev_f)
        h_b.append(h_prev_b)

    ####################
    # compute attentions
    ####################

    # list of T_w tensors --> tensor with shape = [B * T_s, T_w, D_s]
    h_f = tf.stack(h_f, axis=1)
    h_b = tf.stack(h_b, axis=1)
    # reverse again backward state
    h_b = tf.reverse_sequence(h_b, seq_lengths=N_w, seq_axis=1)
    # zero vector
    log_zero_vec = tf.fill(tf.stack([B * T_s, 1]), -1000.0)
    # attention weights for all sentences
    prob_ws = []

    for t_w in range(T_w):
        # stacked hidden state, shape=[B * T_s, 2 * D_s]
        h = tf.concat([h_f[:, t_w, :], h_b[:, t_w, :]], axis=1)
        # word attention, shape=[B * T_s, 1]
        score = compute_attention_score(h,
                                        theta_att,
                                        dropout_att_wb, dropout_att_v,
                                        is_training,
                                        )
        # condition based on true sentence length: t_w < T_w
        condition = tf.less(t_w, N_w)
        # set scoring value to zero if t > T
        score = tf.where(condition, score, log_zero_vec)
        # list of T_w vectors with shape=[B * T_s, 1]
        prob_ws.append(score)

    # list of T_w tensors --> tensor with shape = [B * T_s, T_w]
    prob_ws = tf.concat(prob_ws, axis=1)
    # softmax layer to get pseudo-probabilities
    prob_ws = tf.nn.softmax(prob_ws, axis=1)
    # expand to shape=[B * T_s, T_w, 1]
    alpha = tf.expand_dims(prob_ws, axis=2)
    # fill up to shape=[B * T_s, T_w, D_s]
    alpha = tf.tile(alpha, tf.stack([1, 1, 2 * D_s]))
    # tensor with shape = [B * T_s, T_w, 2 * D_s]
    h = tf.concat([h_f, h_b], axis=2)

    # combine to get sentence representations, shape=[B * T_s, 2 * D_s]
    e_s = tf.reduce_sum(tf.multiply(alpha, h), axis=1, keepdims=False)

    # shape = [B, T_s, 2 * D_s]
    e_s = tf.reshape(e_s, shape=[B, T_s, 2 * D_s])
    # word probabilities
    prob_ws = tf.reshape(prob_ws, shape=[B, T_s, T_w])

    return e_s, prob_ws


def LSTM_ws(e_w, N_w, T_w, T_s, D_w, D_s, theta, dropout_x, dropout_h, B, is_training):

    # extract dimensions to initialize state(s)
    dims = tf.stack([B * T_s, D_s])

    # reshape N_w, shape = [B * T_s]
    N_w = tf.reshape(N_w, shape=[B * T_s])

    # reshape input word embeddings, shape = [B * T_s, T_w, D_w]
    e_w = tf.reshape(e_w, shape=[B * T_s, T_w, D_w])

    # initialize state(s) with zeros
    h_prev = tf.fill(dims, 0.0)
    c_prev = tf.fill(dims, 0.0)

    ##########################################
    # compute forward / backward hidden states
    ###########################################
    for t_w in range(T_w):

        # shapes = [B * T_s, D_s, 2]
        h_next, c_next = LSTM_cell(h_prev, c_prev,
                                   e_w[:, t_w, :],
                                   theta,
                                   dropout_x, dropout_h,
                                   is_training,
                                   )

        # t < T
        condition = tf.less(t_w, N_w)

        # copy-through states if t > T
        h_prev = tf.where(condition, h_next, h_prev)
        c_prev = tf.where(condition, c_next, c_prev)

    # shape = [B, T_s, D_s]
    e_s = tf.reshape(h_prev, shape=[B, T_s, D_s])

    return e_s


###############################
# sentence-to-document encoding
###############################
def BiLSTM_ATT_sd(e_s_f, N_s, T_s, D_d,
                  theta_f, theta_b, theta_att,
                  dropout_x_f, dropout_h_f, dropout_x_b, dropout_h_b,
                  dropout_att_wb, dropout_att_v,
                  B, is_training,
                  ):
    """
    Sentence-to-document encoding: sentence-embeddings to document-embeddings forward propagation

    Input:
        e_s:        sentence embeddings, shape = [B, T_s, 2 * D_s]
        N_s:        index tensor for true number of sentences per document, shape = [B]

    Output:
        e_d:        final document embeddings, shape = [B, 2 * D_d]
        prob_sd:    sentence-based attentions (pseudo-probabilities), shape = [B, T_s]

    """

    # extract dimensions to initialize state(s)
    dims = tf.stack([B, D_d])

    # initialize states with zeros
    h_prev_f = tf.fill(dims, 0.0)
    h_prev_b = tf.fill(dims, 0.0)
    c_prev_f = tf.fill(dims, 0.0)
    c_prev_b = tf.fill(dims, 0.0)

    # reverse input sentence embeddings
    e_s_b = tf.reverse_sequence(e_s_f, seq_lengths=N_s, seq_axis=1)

    # store all states
    h_f = []
    h_b = []

    for t_s in range(T_s):
        # shapes = [B, D_d]
        h_next_f, c_next_f = LSTM_cell(h_prev_f, c_prev_f,
                                       e_s_f[:, t_s, :],
                                       theta_f,
                                       dropout_x_f,
                                       dropout_h_f,
                                       is_training,
                                       )
        h_next_b, c_next_b = LSTM_cell(h_prev_b, c_prev_b,
                                       e_s_b[:, t_s, :],
                                       theta_b,
                                       dropout_x_b,
                                       dropout_h_b,
                                       is_training,
                                       )
        # n < T_s
        condition = tf.less(t_s, N_s)

        # copy-through states if t > T
        h_prev_f = tf.where(condition, h_next_f, h_prev_f)
        c_prev_f = tf.where(condition, c_next_f, c_prev_f)
        h_prev_b = tf.where(condition, h_next_b, h_prev_b)
        c_prev_b = tf.where(condition, c_next_b, c_prev_b)

        # states, list of T_w tensors with shape=[B, D_d]
        h_f.append(h_prev_f)
        h_b.append(h_prev_b)

    ####################
    # compute attentions
    ####################

    # list of T_s tensors --> tensor with shape = [B, T_s, D_d]
    h_f = tf.stack(h_f, axis=1)
    h_b = tf.stack(h_b, axis=1)
    # reverse backward state
    h_b = tf.reverse_sequence(h_b, seq_lengths=N_s, seq_axis=1)
    # zero vector
    log_zero_vec = tf.fill(tf.stack([B, 1]), -1000.0)
    # attention weights
    prob_sd = []

    # compute weights
    for t_s in range(T_s):
        # stacked hidden state, shape=[B, 2 * D_d]
        h = tf.concat([h_f[:, t_s, :], h_b[:, t_s, :]], axis=1)
        # word attention, shape=[B, 1]
        score = compute_attention_score(h, theta_att, dropout_att_wb, dropout_att_v, is_training)
        # condition based on true length: t_s < T_s
        condition = tf.less(t_s, N_s)
        # set scoring value to zero if t > T
        score = tf.where(condition, score, log_zero_vec)
        # list of T_w vectors with shape=[B, 1]
        prob_sd.append(score)

    # list of T_s tensors --> tensor with shape = [B, T_s]
    prob_sd = tf.concat(prob_sd, axis=1)
    # softmax layer to get pseudo-probabilities
    prob_sd = tf.nn.softmax(prob_sd, axis=1)
    # expand to shape=[B, T_s, 1]
    alpha = tf.expand_dims(prob_sd, axis=2)
    # fill up to shape=[B, T_s, 2 * D_d]
    alpha = tf.tile(alpha, tf.stack([1, 1, 2 * D_d]))
    # memory states, list of T_s tensors --> tensor with shape = [B , T_s, 2 * D_d]
    h = tf.concat([h_f, h_b], axis=2)

    # context, combine to get sentence representations, shape=[B, 2 * D_d]
    e_d = tf.reduce_sum(tf.multiply(alpha, h), axis=1, keepdims=False)

    return e_d, prob_sd


def LSTM_sd(e_s, N_s, T_s, D_d, theta, dropout_x, dropout_h, B, is_training):
    """
    Sentence-to-document encoding: sentence-embeddings to document-embeddings forward propagation

    Input:
        e_s:        sentence embeddings, shape = [B, T_s, 2 * D_s]
        N_s:        index tensor for true number of sentences per document, shape = [B]

    Output:
        e_d:        final document embeddings, shape = [B, 2 * D_d]
        prob_sd:    sentence-based attentions (pseudo-probabilities), shape = [B, T_s]

    """

    # extract dimensions to initialize state(s)
    dims = tf.stack([B, D_d])

    # initialize states with zeros
    h_prev = tf.fill(dims, 0.0)
    c_prev = tf.fill(dims, 0.0)

    for t_s in range(T_s):
        # shapes = [B, D_d]
        h_next, c_next = LSTM_cell(h_prev, c_prev,
                                   e_s[:, t_s, :],
                                   theta,
                                   dropout_x,
                                   dropout_h,
                                   is_training,
                                   )
        # n < T_s
        condition = tf.less(t_s, N_s)

        # copy-through states if t > T
        h_prev = tf.where(condition, h_next, h_prev)
        c_prev = tf.where(condition, c_next, c_prev)

    # shape=[B, D_d]
    e_d = h_prev

    return e_d
