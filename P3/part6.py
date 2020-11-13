from sklearn.svm import LinearSVC, SVC
import numpy as np
import random
import math


def make_pairs(data_path='./data/'):
    queries = {}
    with open(data_path) as f:
        for line in f:
            relevance = int(line[0:2])
            line = list(line[2:].split(' '))
            qid = None
            q = []
            for item in line:
                try:
                    key, value = item.split(':')
                except:
                    break
                if key == 'qid':
                    qid = value
                if key == 'qid' and value not in queries.keys():
                    queries[value] = [[], [], []]
                if key != 'qid':
                    q.append(float(value))

            assert len(q) == 46
            queries[qid][relevance].append(np.asarray(q))

        pairs = []
        for q in queries.keys():
            tmp = queries[q]
            for high_2 in tmp[2]:
                for high_1 in tmp[1]:
                    pairs.append((high_2, high_1))
            for high_2 in tmp[2]:
                for high_0 in tmp[0]:
                    pairs.append((high_2, high_0))
            for high_1 in tmp[1]:
                for high_0 in tmp[0]:
                    pairs.append((high_1, high_0))

        pairs = np.asarray(pairs)
        return pairs


def build_training_data(data_path='./data/'):
    pairs = make_pairs(data_path=data_path)
    data = []
    for p in pairs:
        data.append(p[0]-p[1])
    data = np.asarray(data)
    targets = np.ones(data.shape[0], dtype=int)
    p = random.choices(range(targets.shape[0]), k=targets.shape[0]//2)
    targets[p] = -1
    data[p] = -data[p]
    return data, targets


def get_data_and_ground_truth_ranking(data_path='./data'):
    queries = {}
    with open(data_path) as f:
        for line in f:
            relevance = int(line[0:2])
            line = list(line[2:].split(' '))
            qid = None
            q = []
            for i, item in enumerate(line):
                try:
                    key, value = item.split(':')
                except:
                    if item == '#docid':
                        doc_id = line[i+2]
                        break
                if key == 'qid':
                    qid = value
                if key == 'qid' and value not in queries.keys():
                    queries[value] = []
                if key != 'qid':
                    q.append(float(value))
            assert len(q) == 46
            queries[qid].append((np.asarray(q), doc_id, relevance))

        for q in queries.keys():
            queries[q].sort(key=lambda t: (t[2], t[1]), reverse=True)

        return queries


def obtain_ranking(svm, data_path='./data'):
    data = get_data_and_ground_truth_ranking(data_path=data_path)
    docs_score = {}
    rankings = {}
    for q in data.keys():
        docs_score[q] = []
        for i in range(len(data[q])):
            data_i = []
            for j in range(len(data[q])):
                if i != j:
                    data_i.append(data[q][i][0] - data[q][j][0])
            data_i = np.asarray(data_i)
            scores = svm.predict(data_i)
            docs_score[q].append((data[q][i][1], sum(scores)))
        docs_score[q].sort(key=lambda t: (t[1], t[0]), reverse=True)
        svm_rank = [x for x, _ in docs_score[q]]
        opt_rank = [y for _, y, _ in data[q]]
        relevances = {y:z for _, y, z in data[q]}
        rankings[q] = (svm_rank, opt_rank, relevances)

    return rankings


def ndcg_5(rank, best_rank, relevances):
    best_dcg = 0
    for i, r in enumerate(best_rank):
        if i > 4:
            break
        rel = relevances[r]
        denom = math.log2(1 + (i+1))
        best_dcg += rel/denom

    dcg = 0
    for i, r in enumerate(rank):
        if i > 4:
            break
        rel = relevances[r]
        denom = math.log2(1 + (i + 1))
        dcg += rel / denom

    if best_dcg == 0:
        return float('nan')

    return dcg/best_dcg


def test(svm, data_path='./data/'):
    rankings = obtain_ranking(svm=svm, data_path=data_path)
    ndcgs = []
    for q in rankings.keys():
        ndcgs.append(ndcg_5(rankings[q][0], rankings[q][1], rankings[q][2]))
    ndcgs = [x for x in ndcgs if not math.isnan(x)]
    return sum(ndcgs)/(len(ndcgs)+1e-20)


def train_svm(data_path='./data/', C=1.):
    training_data, training_targets = build_training_data(data_path=data_path+'/train.txt')
    svm = LinearSVC(C=C, max_iter=1000)
    # svm = SVC(C=C, max_iter=2000)
    svm.fit(training_data, training_targets)
    return test(svm=svm, data_path=data_path+'/vali.txt'), svm


def find_c(data_path='./data/'):
    best_res = 0.
    best_c = None
    best_svm = None
    for c in [.01, .1, 1., 10., 100.]:
        res, svm = train_svm(data_path=data_path, C=c)
        print("Trying for C={} NDCG@5={}".format(c, res))
        if res > best_res:
            best_c = c
            best_res = res
            best_svm = svm
    print("Best C in exponential search=", best_c, "With NDCG5 on Validation Set=", best_res)

    ranges = {.01: (0.001, 0.03, 0.003), .1: (0.001, 0.3, 0.03), 1.: (0.5, 3, 0.25), 10.: (8, 12, 0.5), 100.: (80, 120, 4)}
    start, end, step = ranges[best_c]
    for c in np.arange(start, end, step):
        res, svm = train_svm(data_path=data_path, C=c)
        print("Trying for C={} NDCG@5={}".format(c, res))
        if res > best_res:
            best_c = c
            best_res = res
            best_svm = svm
    print("Best C after search in neighborhood=", best_c, "With NDCG5 on Validation Set=", best_res)
    res = test(svm=best_svm, data_path=data_path+'/test.txt')
    print("With NDCG5 on Test Set=", res)


# find_c()
