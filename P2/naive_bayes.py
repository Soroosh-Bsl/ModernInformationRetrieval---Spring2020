import json
from utils import tokenize, process_by_nltk
from evaluation_metrics import evaluate
from tqdm import tqdm
import math


def analyze_training_data(data_path, preprocess=None):
    json_file = open(data_path + 'train.json')
    training_data = json.load(json_file)
    json_file.close()
    categories = [0 for i in range(4)]
    categories_words = [0 for i in range(4)]
    index = {}
    N = 0
    for idx, doc in enumerate(training_data):
        categories[int(doc['category'])-1] += 1
        doc_words = tokenize(doc)
        if preprocess is not None:
            doc_words = process_by_nltk(doc_words, preprocess)
        for word in doc_words:
            categories_words[int(doc['category'])-1] += 1
            if word not in index.keys():
                l = [0 for i in range(4)]
                l[int(doc['category'])-1] += 1
                index[word] = l
            else:
                index[word][int(doc['category'])-1] += 1
        N += 1
    N_words = len(index.keys())

    return index, categories, categories_words, N_words


def naive_bayes_on_single_doc(doc, index, categories, categories_words, N_words, alpha=1., preprocess=None):
    doc_words = tokenize(doc)
    if preprocess is not None:
        doc_words = process_by_nltk(doc_words, preprocess)
    probabilities = [0 for i in range(4)]

    for i in range(4):
        probabilities[i] += math.log10(categories[i]/sum(categories))

    for word in doc_words:
        if word in index.keys():
            for i in range(4):
                probabilities[i] += math.log10((index[word][i] + alpha) / (categories_words[i] + alpha*N_words))

    max_index = 0
    for i in range(4):
        if probabilities[i] > probabilities[max_index]:
            max_index = i
    return max_index+1


def naive_bayes_on_validation_set(data_path, index, categories, categories_words, N_words, alpha=1., preprocess=None):
    json_file = open(data_path+'validation.json')
    validation_data = json.load(json_file)
    json_file.close()
    predictions = []
    labels = []
    for idx, doc in tqdm(enumerate(validation_data)):
        predictions.append(naive_bayes_on_single_doc(doc, index, categories, categories_words, N_words, alpha, preprocess))
        labels.append(doc['category'])

    return predictions, labels


def find_alphas(preprocess=None):
    index, categories, categories_words, N_words = analyze_training_data('./data/')
    results = {}
    for alpha in [i/10 for i in range(1, 51)]:
        predictions, labels = naive_bayes_on_validation_set('./data/', index, categories, categories_words, N_words, alpha=alpha, preprocess=preprocess)
        results[alpha] = evaluate(predictions, labels)
    if preprocess is not None:
        f = open('naive_bayes_with_different_alphas_'+str(preprocess)+'_.txt', 'w+')
    else:
        f = open('naive_bayes_with_different_alphas.txt', 'w+')
    for alpha in results.keys():
        print("##############################\nRESULTS FOR ALPHA={}\n".format(alpha),
              results[alpha], file=f)
    f.close()


# find_alphas('stopwords')
