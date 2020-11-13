from elasticsearch import Elasticsearch


def search(terms, weights, host_ip='localhost', port_number=9200, use_page_rank=False, size=10):
    es = Elasticsearch([{"host": host_ip, "port": port_number}])

    if use_page_rank:
        query = {
                    "from": 0,
                    "size": size,
                    "query": {
                        "function_score": {
                            "functions": [
                                {
                                    "script_score": {
                                        "script": "_score + 10000*doc['paper.page_rank'].value"
                                    }
                                }
                            ],
                            "query": {
                                "bool": {
                                    "should": [
                                        {
                                            "match": {
                                                "paper.title": {
                                                    "query": terms[0],
                                                    "boost": weights[0]
                                                }
                                            }
                                        },
                                        {
                                            "match": {
                                                "paper.abstract": {
                                                    "query": terms[1],
                                                    "boost": weights[1]
                                                }
                                            }
                                        },
                                        {
                                            "range": {
                                                "paper.date": {
                                                    "gte": int(terms[2]),
                                                    "boost": weights[2]
                                                }
                                            }
                                        }
                                    ]
                                }
                            }
                        }
                    }
                }
    else:
        query = {
                    "from": 0,
                    "size": size,
                    "query": {
                        "function_score": {
                            "functions": [
                                {
                                    "script_score": {
                                        "script": "_score"
                                    }
                                }
                            ],
                            "query": {
                                "bool": {
                                    "should": [
                                        {
                                            "match": {
                                                "paper.title": {
                                                    "query": terms[0],
                                                    "boost": weights[0]
                                                }
                                            }
                                        },
                                        {
                                            "match": {
                                                "paper.abstract": {
                                                    "query": terms[1],
                                                    "boost": weights[1]
                                                }
                                            }
                                        },
                                        {
                                            "range": {
                                                "paper.date": {
                                                    "gte": int(terms[2]),
                                                    "boost": weights[2]
                                                }
                                            }
                                        }
                                    ]
                                }
                            }
                        }
                    }
                }

    res = es.search(index='paper_index', doc_type='paper', body=query)
    res = res['hits']['hits']
    for i, r in enumerate(res):
        print(r)
        r = r['_source']['paper']
        r_title = 'Title: '+r['title']+'\n'
        r_abstract = 'Abstract: '+r['abstract']+'\n'
        r_date = 'Publication Date: '+str(r['date'])+'\n'
        r_authors = 'Authors: '+str(r['authors'])+'\n'
        r_id = 'SemanticScholarID: '+r['id']+'\n'
        r_refs_ids = 'SemanticScholarIDs of References: '+str(r['references'])+'\n'

        if use_page_rank:
            r_pg = 'PageRank of Paper: '+str(r['page_rank'])+'\n'
            string = str(i+1)+'-\n'+r_title+r_authors+r_abstract+r_date+r_id+r_refs_ids+r_pg+'##########################\n'
        else:
            string = str(i+1)+'-\n'+r_title+r_authors+r_abstract+r_date+r_id+r_refs_ids+'##########################\n'
        print(string)


# search(["Lottery", "lottery", "2018"], [2, 1, 1], use_page_rank=True)
