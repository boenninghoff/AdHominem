# AdHominem: A tool for automatically analyzing the writing style in social media messages

<img src="pic_attention.png" width="600">

This repository contains the source code used in our paper [_Explainable Authorship Verification in Social Media via Attention-based Similarity Learning_](https://arxiv.org/abs/1910.08144) published at [_2019 IEEE International Conference on Big Data (IEEE BigData 2019)_](http://bigdataieee.org/BigData2019/)

Please, feel free to send any comments or suggestions! (benedikt.boenninghoff[at]rub.de)

# Installation

We used Python 3.6 (Anaconda 3.6). The following libraries are required:

* Tensorflow 1.12. - 1.15
* spacy 2.3.2 (download tokenizer via "python -m spacy download en_core_web_lg")
* textacy 0.8.0
* fasttext 0.9.2
* numpy 1.18.1
* scipy 1.4.1
* pandas 1.0.4
* scikit-learn 0.20.3
* bs4 0.0.1

# Dataset

This repository works with a [_small Amazon review dataset_](https://github.com/marjanhs/prnn), including 9000 review pairs written by 300 distinct authors. Since you can achieve 99% accuracy, this dataset provides a simple sanity check for new authorship verification methods.

    mkdir data
    cd data
    wget https://github.com/marjanhs/prnn/raw/master/data/amazon.7z
    sudo apt-get install p7zip-full
    7z x amazon.7z

# Download pretrained word embeddings

We used [_pretrained word embeddings_](https://fasttext.cc/). You may prepare them as follows:
    
    cd data
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
    gunzip cc.en.300.bin.gz

# Data preprocessing
    
    python main_preprocess.py


# Training
You can train AdHominem as follows:
    
    python main_adhominem.py


Using the evaluation script of the [_PAN 2020 AV challenge_](https://pan.webis.de/clef20/pan20-web/author-identification.html), the results may look like this:

| AUC   |  c@1  | f_0.5_u |  F1    | overall |
|:-----:|:-----:|:-------:|:------:|:-------:|
| 0.991 | 0.988 | 0.991   | 0.992  |  0.992  |


# Cite the paper

If you use our code or data, please cite the papers using the following BibTeX entries:

    @inproceedings{Boenninghoff2019b,
    author={Benedikt Boenninghoff, Steffen Hessler, Dorothea Kolossa and Robert M. Nickel},
    title={Explainable Authorship Verification in Social Media via Attention-based Similarity Learning},
    booktitle={IEEE International Conference on Big Data (IEEE Big Data 2019), Los Angeles, CA, USA, December 9-12, 2019},
    year={2019},
    }


