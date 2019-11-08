# -*- coding: utf-8 -*-
import tensorflow as tf
from feature_extraction import make_dropout_mask


###################################
# main function for metric learning
###################################
def metric_learning(placeholders, hyper_parameters, features):

    ################################
    # initialize trainable variables
    ################################
    model_type = hyper_parameters["model_type"]
    func = {
        "AdHominem": initialize_metric_adhominem,
        "HRSN": initialize_metric_hrsn,
    }[model_type]
    with tf.variable_scope("initialize_metric_learning"):
        thetas = func(hyper_parameters)

    #################
    # prepare dropout
    #################
    func = {
        "AdHominem": dropout_metric_adhominem,
        "HRSN": dropout_metric_hrsn,
    }[model_type]
    with tf.variable_scope("dropout_metric"):
        dropout = func(hyper_parameters, placeholders["B"])

    #######################
    # apply metric learning
    #######################
    func = {
        "AdHominem": metric_adhominem,
        "HRSN": metric_hrsn,
    }[model_type]
    with tf.variable_scope("apply_metric_layer"):
        out = func(placeholders, dropout, thetas, features)

    return thetas, out


##############################
# initialize metric and domain
##############################
def initialize_metric_adhominem(hyper_parameters):

    r = hyper_parameters["r_mlp"]
    D_d = hyper_parameters["D_d"]
    D_mlp = hyper_parameters["D_mlp"]

    # metric learning
    with tf.variable_scope("layer_metric"):
        theta_metric = initialize_metric(D_in=2 * D_d,
                                         D_out=D_mlp,
                                         r=r,
                                         )

    thetas = {"theta_metric": theta_metric}

    return thetas


def initialize_metric_hrsn(hyper_parameters):

    """
    r = hyper_parameters["r_mlp"]
    D_d = hyper_parameters["D_d"]
    D_mlp = hyper_parameters["D_mlp"]

    # metric learning
    with tf.variable_scope("layer_metric"):
        theta_metric = initialize_metric(D_in=D_d,
                                         D_out=D_mlp,
                                         r=r,
                                         )

    thetas = {"theta_metric": theta_metric}
    """
    thetas = None

    return thetas


####################
# dropout for metric
####################
def dropout_metric_adhominem(hyper_parameters, B):

    D_d = hyper_parameters["D_d"]

    with tf.variable_scope("dropout_metric"):
        keep_prob = hyper_parameters["keep_prob_metric"]
        dropout_metric = make_dropout_mask(shape=[B, 2 * D_d], keep_prob=keep_prob)

    dropout = {"metric": dropout_metric}

    return dropout


def dropout_metric_hrsn(hyper_parameters, B):

    """
    D_d = hyper_parameters["D_d"]

    with tf.variable_scope("dropout_metric"):
        keep_prob = hyper_parameters["keep_prob_metric"]
        dropout_metric = make_dropout_mask(shape=[B, D_d], keep_prob=keep_prob)

    dropout = {"metric": dropout_metric}
    """
    dropout = None

    return dropout


##########################
# metric learning for HRSN
##########################
def metric_hrsn(placeholders, dropout, thetas, features):

    out = {"y_L": features["e_d_L"],
           "y_R": features["e_d_R"],
           }

    return out


##################################################
# metric learning for domain-adversarial AdHominem
##################################################
def metric_adhominem(placeholders, dropout, thetas, features):

    ######################
    # siamese metric layer
    ######################
    with tf.variable_scope("metric") as scope:

        ######################
        # first neural network
        ######################
        # metric learning
        y_L = metric_layer(features["e_d_L"], dropout["metric"], placeholders["is_training"], thetas["theta_metric"])

        #######################
        # second neural network
        #######################
        scope.reuse_variables()
        y_R = metric_layer(features["e_d_R"], dropout["metric"], placeholders["is_training"], thetas["theta_metric"])

    out = {"y_L": y_L,
           "y_R": y_R,
           }

    return out


##################################
# initialize metric learning layer
##################################
def initialize_metric(D_in, D_out, r):

    # dimensions for all layers
    theta = {"W": tf.get_variable("W",
                                  shape=[D_in, D_out],
                                  initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                  trainable=True,
                                  ),
             "b": tf.get_variable("b",
                                  shape=[D_out],
                                  initializer=tf.constant_initializer(0.0),
                                  trainable=True,
                                  ),
             }
    return theta


#######################
# metric learning layer
#######################
def metric_layer(e_d, dropout_mask, is_training, theta):

    """
    Fully-connected MLP layer for nonlinear metric learning

    Input:
        e_d:    document embeddings, shape = [B, 2 * D_d]

    Output:
        y:      document features, shape = [B, D_mlp]

    """
    y = e_d

    # apply dropout
    y = tf.cond(tf.equal(is_training, tf.constant(True)),
                  lambda: tf.multiply(dropout_mask, y),
                  lambda: y,
                  )
    # fully-connected layer
    y = tf.nn.xw_plus_b(y, theta["W"], theta["b"])
    # nonlinear output
    y = tf.nn.tanh(y)

    return y
