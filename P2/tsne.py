from sklearn.manifold.t_sne import TSNE
import numpy as np
from utils import make_index_and_save_training_data_as_vector_space, save_validation_data_as_vector_space
import matplotlib.pyplot as plt


def data_clustering_representation(data_path, n_samples=100, n_clusters=4, vcs=None, cl_labels=None):
    if vcs is None:
        clustering_labels = np.load(data_path + 'labels_of_clustering.npy')
        training_vectors, index, categories = make_index_and_save_training_data_as_vector_space(data_path, False)
    else:
        clustering_labels = cl_labels
        training_vectors = vcs
        categories = np.load(data_path+'categories.npy', allow_pickle=True)
    labels = []
    selected_ids = []
    colors = ['green', 'red', 'blue', 'black', 'yellow']
    markers = ['-', '*', 'o', '^', '*']
    cs1 = []
    cs2 = []
    ms = []
    for c in range(n_clusters):
        l = 0
        while len(selected_ids) < n_samples * (c + 1):
            if clustering_labels[l] == c:
                selected_ids.append(l)
                labels.append(c + 1)
                cs1.append(colors[categories[l] - 1])
                cs2.append(colors[c])
                ms.append(markers[c])
            l += 1

    new_vcs = TSNE(n_components=2).fit_transform(training_vectors[selected_ids], )
    fig = plt.figure()
    plt.scatter(new_vcs[:, 0], new_vcs[:, 1], c=cs1)
    plt.savefig('./tsne_one.png')
    fig = plt.figure()
    plt.scatter(new_vcs[:, 0], new_vcs[:, 1], c=cs2)
    plt.savefig('./tsne_two.png')
    # plt.show()


# data_clustering_representation('./data/')