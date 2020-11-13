import json
import gensim
import numpy as np
from utils import tokenize
from k_means import KM
from tsne import data_clustering_representation


def save_as_w2v(data_path):
    json_file = open(data_path + 'train.json')
    training_data = json.load(json_file)
    json_file.close()
    bad_tokens = ['\'', '\"', '.', ',', '?', '!', '[', ']', ')', '(', '@', '#', '$', '%', '^', '&', '*', '+',
                  '=', '<', '>', ';', ':', '-']
    all_docs = []
    for idx, doc in enumerate(training_data):
        doc_words = tokenize(doc)
        all_docs.append(doc_words)

    SIZE = 100
    model1 = gensim.models.Word2Vec(all_docs, min_count=1, size=SIZE, window=100)

    vcs = np.empty([24000, SIZE])
    for idx, doc in enumerate(training_data):
        doc_words = tokenize(doc)
        n = 0
        for w in doc_words:
            try:
                vcs[idx] += model1[w]
                n += 1
            except KeyError:
                pass
        vcs[idx] /= n

    return vcs


def kmeans_on_w2vec(data_path):
    vcs = save_as_w2v('./data/')
    K = KM('./data/', True, 4, vcs=vcs)
    _, labels = K.k_means(False)
    data_clustering_representation('./data/', vcs=vcs, cl_labels=labels)


# kmeans_on_w2vec('./data/')


