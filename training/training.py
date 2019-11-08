# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


###################################
# main function for metric learning
###################################
def loss_optimizer(placeholders, hyper_parameters, inputs, thetas):

    ##################
    # compute distance
    ##################
    with tf.variable_scope("distance"):
        distance = euclidean_distance(inputs["y_L"], inputs["y_R"])

    ############################
    # compute loss and optimizer
    ############################
    model_type = hyper_parameters["model_type"]
    func = {
        "AdHominem": loss_optimizer_adhominem,
        "HRSN": loss_optimizer_adhominem,
    }[model_type]
    with tf.variable_scope("loss_optimizer_domain"):
        training = func(distance, hyper_parameters, placeholders, inputs, thetas)

    return training, distance


#########################################
# function for loss and optimizer of HRSN
#########################################
def loss_optimizer_adhominem(distance, hyper_parameters, placeholders, inputs, thetas):

    ###############
    # loss function
    ###############
    loss = loss_siamese(hyper_parameters["t_s"], hyper_parameters["t_d"], placeholders["labels"], distance)

    ###############
    # learning rate
    ###############
    # global step counter
    global_step = tf.Variable(0, dtype=tf.float32, trainable=False)
    if hyper_parameters["opt"] == "Adam":
        learning_rate = tf.train.cosine_decay_restarts(learning_rate=hyper_parameters["initial_learning_rate"],
                                                       global_step=global_step,
                                                       first_decay_steps=hyper_parameters["first_decay_steps"],
                                                       t_mul=hyper_parameters["t_mul"],
                                                       m_mul=hyper_parameters["m_mul"],
                                                       alpha=0.0001,
                                                       )
    else:
        # fixed initial learning rate
        learning_rate = tf.Variable(hyper_parameters["initial_learning_rate"], dtype=tf.float32, trainable=False)

    ###########
    # optimizer
    ###########
    if hyper_parameters["opt"] == "Adam":
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:
        opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)

    # local gradient normalization
    grads_and_vars = opt.compute_gradients(loss)
    clipped_grads_and_vars = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in grads_and_vars]
    optimizer = opt.apply_gradients(clipped_grads_and_vars, global_step=global_step)

    training = {"loss": loss,
                "optimizer": optimizer,
                "global_step": global_step,
                "learning_rate": learning_rate,
                }

    return training


####################
# Euclidean distance
####################
def euclidean_distance(y_L, y_R):
    """
    y_L, y_R    document features, shape = [B, D_mlp]
    distance    tensor containing the distance values, shape = [B, 1]

    """

    # define euclidean distance, shape = (B, D_h)
    distance = tf.subtract(y_L, y_R)
    distance = tf.square(distance)
    distance = tf.maximum(distance, 1e-8)
    # shape = (B, 1)
    distance = tf.reduce_sum(distance, 1, keepdims=True)
    distance = tf.sqrt(distance)

    return distance


def compute_labels_euclidean(labels, distance, t_s, t_d):

    # threshold
    th = 0.5 * (t_s + t_d)

    # numpy array for estimated labels
    labels_hat = np.ones(labels.shape)
    # dissimilar pairs --> 0, similar pairs --> 1
    labels_hat[distance > th] = 0

    return labels_hat


###############
# loss function
###############
def loss_siamese(t_s, t_d, labels, distance):

    ####################################
    # modified contrastive loss function
    ####################################

    # define contrastive loss:
    l1 = tf.multiply(labels, tf.square(tf.maximum(tf.subtract(distance, t_s), 0.0)))
    l2 = tf.multiply(tf.subtract(1.0, labels), tf.square(tf.maximum(tf.subtract(t_d, distance), 0.0)))
    loss = tf.add(l1, l2)
    loss = tf.reduce_mean(loss)

    return loss