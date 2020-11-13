from elasticsearch import Elasticsearch
import json
from tqdm import tqdm


def make_index(host_ip='localhost', port_number=9200, json_dir='./scholar.json'):
    es = Elasticsearch([{'host': host_ip, 'port': port_number}])
    if es.indices.exists(index="paper_index"):
        es.indices.delete(index='paper_index', ignore=[400, 404])
    es.indices.create(index='paper_index', ignore=[400, 404], body={})
    with open(json_dir) as f:
        data = json.load(f)
        id = 0
        for p in tqdm(data["papers"]):
            p = '{' + \
                '"paper":' + \
                    json.dumps(p) + \
                '}'
            p = json.loads(p)
            es.index(index='paper_index', doc_type='paper', body=p, id=id)
            id += 1
    return es


def clear_index(host_ip='localhost', port_number=9200,):
    es = Elasticsearch([{'host': host_ip, 'port': port_number}])
    es.indices.delete(index='paper_index', ignore=[400, 404])


def clear_index_docs(host_ip='localhost', port_number=9200, ):
    es = Elasticsearch([{'host': host_ip, 'port': port_number}])
    es.indices.delete(index='paper_index', ignore=[400, 404])
    es.indices.create(index='paper_index', ignore=[400, 404])


