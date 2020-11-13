from elasticsearch import Elasticsearch
import numpy as np
from tqdm import tqdm


def adjacency_matrix_of_authors(host_ip='localhost', port_number=9200, load=False):
    if load:
        M = np.load('./M_authors.npy')
        authors_list = list(np.load('./authors_list.npy'))
        return M, authors_list

    es = Elasticsearch([{"host": host_ip, "port": port_number}])
    x = es.count(index='paper_index', doc_type='paper')
    number_of_papers = x['count']
    authors = {}
    for id in tqdm(range(number_of_papers)):
        p = es.get(index='paper_index', doc_type='paper', id=id)
        p = p['_source']['paper']
        related_authors = []
        for ref in p['references']:
            q = '{"query":{"term":{"paper.id":"!"}}}'
            q = q.replace('!', str(ref))
            p_of_ref = es.search(index='paper_index', doc_type='paper', body=q)
            if p_of_ref['hits']['total'] > 0:
                p_of_ref = p_of_ref['hits']['hits'][0]['_source']['paper']
                related_authors += p_of_ref['authors']

        for a in p['authors']:
            if a in authors.keys():
                set_of_a = authors[a]
            else:
                set_of_a = set()
            for ra in related_authors:
                set_of_a.add(ra)
            authors[a] = set_of_a

    n_authors = len(authors.keys())
    M = np.zeros([n_authors, n_authors])
    authors_list = list(authors.keys())
    for i in range(len(authors_list)):
        for ra in authors[authors_list[i]]:
            if ra in authors_list:
                M[i][authors_list.index(ra)] = 1

    np.save('./M_authors', M)
    np.save('./authors_list', np.asarray(authors_list))
    return M, authors_list


def hits(required_number_of_authors, host_ip='localhost', port_number=9200, number_of_iterations=5):
    M, authors_list = adjacency_matrix_of_authors(host_ip=host_ip, port_number=port_number)

    authority = np.ones([M.shape[0]])
    hub = np.ones([M.shape[0]])
    for i in range(number_of_iterations):
        authority = np.zeros(authority.shape)
        for j in range(M.shape[0]):
            incoming_neighbors = M[:][j].reshape(-1)
            authority[j] += np.dot(hub, incoming_neighbors)
        authority /= np.linalg.norm(authority)
        hub = np.zeros(hub.shape)
        for j in range(M.shape[0]):
            outgoing_neighbors = M[j].reshape(-1)
            hub[j] += np.dot(authority, outgoing_neighbors)
        hub /= np.linalg.norm(hub)

    result = [(a, x) for a, x in sorted(zip(authority, authors_list), reverse=True)][:required_number_of_authors]
    print("Top Authors: ")
    for i, (a, x) in enumerate(result):
        print(str(i+1) + '- ' + x + " -Authority: " + str(a))
    return result


# hits(10)
