import json
import math
import numpy as np
from evaluation_metrics import evaluate
import time
from utils import make_index_and_save_training_data_as_vector_space, save_validation_data_as_vector_space, tokenize, \
    dict_to_vect


class KNN:
    def __init__(self, data_path, load_train=False, load_val=False, preprocess=None):
        self.data_path = data_path
        self.preprocess = preprocess
        self.methods = ['cosine-similarity', 'euclidean-distance']
        self.train_vcs, self.index, self.categories = make_index_and_save_training_data_as_vector_space(data_path, load=load_train, preprocess=preprocess)
        self.N_docs = len(self.categories)
        self.N_words = len(self.index.keys())
        self.val_vcs, self.labels = save_validation_data_as_vector_space(self.data_path, self.index, self.N_docs,
                                                                         load=load_val, preprocess=preprocess)
        self.N_val_docs = len(self.val_vcs)

    def knn_on_single_doc(self, doc, method, k):
        doc_words = tokenize(doc)
        doc_tf = {}
        for word in doc_words:
            if word in doc_tf.keys():
                doc_tf[word] += 1
            elif word in self.index.keys():
                doc_tf[word] = 1

        for word in doc_tf.keys():
            if word in self.index.keys():
                doc_tf[word] *= math.log10(self.N_docs / len(self.index[word]))

        doc_vector = dict_to_vect(doc_tf, self.index).reshape(1, self.N_words)
        if method == 'cosine-similarity':
            train_vcs_normalized = np.divide(self.train_vcs, np.linalg.norm(self.train_vcs, axis=1)[:, np.newaxis])
            doc_vector_normalized = np.divide(doc_vector, np.linalg.norm(doc_vector, axis=1)[:, np.newaxis])
            result = np.matmul(doc_vector_normalized, train_vcs_normalized.T)
            del train_vcs_normalized
            del doc_vector_normalized
        elif method == 'euclidean-distance':
            x1 = -np.sum(self.train_vcs ** 2, axis=1)[:, np.newaxis].T
            x2 = - np.sum(doc_vector ** 2, axis=1)[:, np.newaxis]
            x3 = 2 * np.matmul(doc_vector, self.train_vcs.T)
            result = x1 + x2 + x3
            del x1
            del x2
            del x3
        nearest_neighbors = np.argsort(result, axis=1)[:, -k:]
        del result
        nearest_neighbors = self.categories[nearest_neighbors]
        prediction = np.bincount(nearest_neighbors[0]).argmax()
        return prediction

    def knn_on_validation_set(self, method='cosine-similarity', k=1):
        if method == 'cosine-similarity':
            train_vcs_normalized = np.divide(self.train_vcs, np.linalg.norm(self.train_vcs, axis=1)[:, np.newaxis])
            val_vcs_normalized = np.divide(self.val_vcs, np.linalg.norm(self.val_vcs, axis=1)[:, np.newaxis])
            result = np.matmul(val_vcs_normalized, train_vcs_normalized.T)
            del train_vcs_normalized
            del val_vcs_normalized
        elif method == 'euclidean-distance':
            x1 = -np.sum(self.train_vcs ** 2, axis=1)[:, np.newaxis].T
            x2 = - np.sum(self.val_vcs ** 2, axis=1)[:, np.newaxis]
            x3 = 2 * np.matmul(self.val_vcs, self.train_vcs.T)
            result = x1 + x2 + x3
            del x1
            del x2
            del x3

        nearest_neighbors = np.argsort(result, axis=1)[:, -k:]
        del result
        print(nearest_neighbors)
        nearest_neighbors = self.categories[nearest_neighbors]
        predictions = [0 for i in range(self.N_val_docs)]
        for i, row in enumerate(nearest_neighbors):
            predictions[i] = np.bincount(row).argmax()
        return predictions, self.labels

    def find_results_of_ks(self):
        results = {m: {} for m in self.methods}
        for m in self.methods:
            for k in [1, 3, 5]:
                t1 = time.time()
                predictions, labels = self.knn_on_validation_set(k=k, method=m)
                results[m][k] = evaluate(predictions, labels)
                t2 = time.time()
                print(t2 - t1, 'secs')
        f = open('knn_with_different_ks.txt', 'w+')
        for m in self.methods:
            for k in results[m].keys():
                print("##############################\nRESULTS FOR M={} K={}\n".format(m, k),
                      results[m][k], file=f)
        f.close()

    def perform_on_specific_setting(self, method, k):
        predictions, labels = self.knn_on_validation_set(k=k, method=method)
        result = evaluate(predictions, labels)
        f = open('knn_with_'+str(self.preprocess)+'.txt', 'w+')
        print("##############################\nRESULTS FOR M={} K={}\n".format(method, k), result, file=f)
        f.close()


# K = KNN('./data/', load_val=False, load_train=False, preprocess='stopwords')
# K.perform_on_specific_setting(method='cosine-similarity', k=5)

# K = KNN('./data/', load_val=False, load_train=False)
# K.find_results_of_ks()
