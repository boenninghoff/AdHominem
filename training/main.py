# -*- coding: utf-8 -*-
from model import SiameseNetwork
import pickle
import os
import argparse
from sklearn.model_selection import train_test_split

##################################
# define model: AdHominem vs. HRSN
##################################
parser = argparse.ArgumentParser()
parser.add_argument('-mt', '--model_type', required=True)
args = parser.parse_args()
hyper_parameters = {"model_type": args.model_type}

##################
print("load data")
##################
with open(os.path.join("..", "data", "data_Amazon_9000"), 'rb') as f:
    docs_L, docs_R, labels, V_w, E_w, V_c = pickle.load(f)

########################################
# split data into training and test sets
########################################
docs_L_tr, docs_L_te, docs_R_tr, docs_R_te, labels_tr, labels_te = train_test_split(docs_L,
                                                                                    docs_R,
                                                                                    labels,
                                                                                    test_size=0.2,
                                                                                    shuffle=True,
                                                                                    )

###############################################
print("define neural network hyper-parameters")
###############################################

# lengths of train/test sets
hyper_parameters["N_tr"] = len(labels_tr)
hyper_parameters["N_te"] = len(labels_te)

# token/character (types) vocabularies
hyper_parameters["V_w"] = V_w
hyper_parameters["V_c"] = V_c

# dimension character embeddings/representations
hyper_parameters["D_c"] = 10
hyper_parameters["D_r"] = 20
# length of CNN sliding window (characters-to-word encoding)
hyper_parameters["h"] = 4
# dimension word embeddings (words-to-sentence encoding)
hyper_parameters["D_w"] = 300
# dimension sentence embeddings (sentences-to-document encoding)
hyper_parameters["D_s"] = 50
# dimension of document embeddings
hyper_parameters["D_d"] = 50
# dimension of neural features (deep metric learning)
hyper_parameters["D_mlp"] = 30
# attention dimensions
hyper_parameters["D_a_ws"] = 50
hyper_parameters["D_a_sd"] = 50

# maximum number of characters per words
hyper_parameters["T_c"] = 15
# maximum number of words per sentence
hyper_parameters["T_w"] = 20
# maximum number of sentences per document
hyper_parameters["T_s"] = 40

# thresholds for loss function
hyper_parameters["t_s"] = 1.0  # boundary for similar pairs
hyper_parameters["t_d"] = 3.0  # boundary for dissimilar pairs

# define initial value for forget state in LSTM cell
hyper_parameters["bf_init"] = 2.5

# define range to initialize trainable weights
hyper_parameters["r_cnn_emb"] = 0.1  # character embeddings
hyper_parameters["r_cnn_W"] = 0.1  # CNN for characters-to-word encoding
hyper_parameters["r_rnn_ws_W"] = 0.05  # LSTM (words-to-sentence encoding)
hyper_parameters["r_rnn_ws_U"] = 0.05  # LSTM (words-to-sentence encoding)
hyper_parameters["r_att_ws_W"] = 0.03  # attention layer (words-to-sentence encoding)
hyper_parameters["r_att_ws_u"] = 0.03  # attention layer (words-to-sentence encoding)
hyper_parameters["r_rnn_sd_W"] = 0.05  # LSTM (sentences-to-document encoding)
hyper_parameters["r_rnn_sd_U"] = 0.05  # LSTM (sentences-to-document encoding)
hyper_parameters["r_att_sd_W"] = 0.03  # attention (sentences-to-document encoding)
hyper_parameters["r_att_sd_u"] = 0.03  # attention (sentences-to-document encoding)
hyper_parameters["r_mlp"] = 0.4  # feed-forward network for deep metric learning

# train word embeddings
hyper_parameters["train_word_embeddings"] = False
# total number of epochs
hyper_parameters["epochs"] = 100
# batch size for training
hyper_parameters["batch_size"] = 32
# batch size for test sets
hyper_parameters["batch_size_dev_te"] = 128

# initial learning rate
hyper_parameters["initial_learning_rate"] = 0.0015
hyper_parameters["opt"] = "Adam"
# parameters for warm restarts
hyper_parameters["first_decay_steps"] = 300
hyper_parameters["t_mul"] = 1.0
hyper_parameters["m_mul"] = 1.0

# keep probabilities for (variational) dropout regularization
hyper_parameters["keep_prob_cnn"] = 0.7
hyper_parameters["keep_prob_rnn"] = 0.7
hyper_parameters["keep_prob_att"] = 0.7
hyper_parameters["keep_prob_metric"] = 0.7
hyper_parameters["keep_prob_domain"] = 0.7


##################################
# write hyper-parameters into file
##################################
if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists(os.path.join("results", hyper_parameters["model_type"])):
    os.makedirs(os.path.join("results", hyper_parameters["model_type"]))
file_results = os.path.join("results", hyper_parameters["model_type"], "results.txt")

open(file_results, "a").write("\n"
                               + "-------------------------------------------------------------------------------------"
                               + "\nPARAMETER SETUP:\n"
                               + "-------------------------------------------------------------------------------------"
                               + "\n"
                               )

for hp in sorted(hyper_parameters.keys()):
    if hp in ["V_c", "V_w"]:
        open(file_results, "a").write("num " + hp + ": " + str(len(hyper_parameters[hp])) + "\n")
    else:
        open(file_results, "a").write(hp + ": " + str(hyper_parameters[hp]) + "\n")


###############################
print("build tensorflow graph")
###############################
model = SiameseNetwork(hyper_parameters=hyper_parameters,
                       E_w=E_w,
                       )

#######################################
print("start siamese network training")
#######################################
train_set = (docs_L_tr, docs_R_tr, labels_tr)
test_set = (docs_L_te, docs_R_te, labels_te)


model_type = hyper_parameters["model_type"]
func = {
    "HRSN": model.train_model_hrsn,
    "AdHominem": model.train_model_adhominem,
}[model_type]

func(train_set, test_set, file_results)

######################
print("close session")
######################
model.sess.close()
