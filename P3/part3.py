import numpy as np
from elasticsearch import Elasticsearch
from tqdm import tqdm


def make_adjacency_matrix_from_es(host_ip='localhost', port_number=9200, number_of_papers=2000, load=False):
    if load:
        M = np.load('./M.npy')
        return M

    id_to_id = ['' for i in range(number_of_papers)]
    es = Elasticsearch([{"host": host_ip, "port": port_number}])
    if not es.indices.exists(index='paper_index'):
        raise NotImplemented
    M = np.zeros([number_of_papers, number_of_papers])

    for id in tqdm(range(number_of_papers)):
        p = es.get(index='paper_index', doc_type='paper', id=id)
        id_to_id[id] = p['_source']['paper']['id']

    # Using Elasticsearch
    for id in tqdm(range(number_of_papers)):
        p = es.get(index='paper_index', doc_type='paper', id=id)
        id_to_id[id] = p['_source']['paper']['id']
        for scholar_id in p['_source']['paper']['references']:
            if scholar_id in id_to_id:
                M[id][id_to_id.index(scholar_id)] = 1
            # query = '{"query":{"term":{"paper.id":"'+str(scholar_id)+'"}}}'
            # ref = es.search(index='paper_index', doc_type='paper', body=query)
            # if ref['hits']['total'] > 0:
            #     M[id][int(ref['hits']['hits'][0]['_id'])] = 1
    np.save('./M', M)

    return M


def pagerank(M, alpha=0.85, maxerr=0.001):
    N = M.shape[1]
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    v_prev = np.ones(v.shape)
    M_hat = (alpha * M + (1 - alpha) / N)
    while np.max(np.abs(v-v_prev)) >= maxerr:
        v_prev = v
        v = np.matmul(M_hat, v)
        v = v / np.linalg.norm(v, 1)
    return v


def set_pagerank_in_es(host_ip='localhost', port_number=9200, alpha=0.85):
    M = make_adjacency_matrix_from_es(host_ip=host_ip, port_number=port_number)
    pg = pagerank(M, alpha)
    es = Elasticsearch([{"host": host_ip, "port": port_number}], )
    for p in tqdm(range(pg.shape[0])):
        update_query = '{"doc":{"paper": {"page_rank": '+str(pg[p][0])+'}}}'
        _ = es.update(index='paper_index', doc_type='paper', id=p, body=update_query)


# set_pagerank_in_es()
