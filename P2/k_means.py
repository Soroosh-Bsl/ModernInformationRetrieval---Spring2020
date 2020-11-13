import numpy as np
from utils import make_index_and_save_training_data_as_vector_space

class KM:
  def __init__(self, data_path, load=False, n_clusters=4, vcs=None):
    self.data_path = data_path
    self.n_clusters = n_clusters
    if vcs is None:
        self.training_vectors, self.index, _ = make_index_and_save_training_data_as_vector_space(data_path, load)
        self.N_words = len(self.index.keys())
    else:
        self.training_vectors = vcs
        self.N_words = vcs.shape[1]
    self.N_docs = self.training_vectors.shape[0]
    self.centroids = np.empty([4, self.N_words], dtype='float32')
    for i in range(self.n_clusters):
        rand_doc = np.random.randint(self.N_docs)
        docs_list = [(rand_doc+j)%self.N_docs for j in range(10)]
        self.centroids[i] = np.mean(self.training_vectors[docs_list])
    self.prev_labels, self.labels = np.asarray([0 for i in range(self.N_docs)]), np.asarray([1 for i in range(self.N_docs)])
    
  def update_clusters(self):
    x1 = np.sum(self.training_vectors ** 2, axis=1)[:, np.newaxis]
    x2 = np.sum(self.centroids**2, axis=1)[:, np.newaxis].T
    x3 = - 2 * np.matmul(self.training_vectors, self.centroids.T)
    result = x1 + x2 + x3
    self.labels = np.argmin(result, axis=1)
    del x1
    del x2
    del x3
    del result

  def update_centroids(self):
    for c in range(self.n_clusters):
        x = self.training_vectors[self.labels==c].reshape(-1, self.N_words)
        self.centroids[c] = np.mean(x, axis=0)
		
  def k_means(self, save=True):
    it = 1
    while not np.array_equal(self.prev_labels, self.labels):
        self.prev_labels = self.labels
        self.update_clusters()
        self.update_centroids()
        string = "\n#####################\nIteration:{}".format(it)
        it += 1
        for i in range(4):
            string += "\nCENTROID {} = {}".format(i+1, np.bincount(self.labels)[i])
        print(string)

    if save:
        np.save(self.data_path+'labels_of_clustering', np.asarray(self.labels))
        np.save(self.data_path+'centroids', np.asarray(self.centroids))

    return np.asarray(self.centroids), np.asarray(self.labels)
    

# K = KM('./data/', False, 4)
# K.k_means()
