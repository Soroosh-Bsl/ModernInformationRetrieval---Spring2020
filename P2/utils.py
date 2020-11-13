import json
import numpy as np
import math
from tqdm import tqdm


def dict_to_vect(vecotr_dict, index):
    ws = list(index.keys())
    new_v = [0 for i in range(len(ws))]
    for idx, w in enumerate(ws):
        if w in vecotr_dict.keys():
            new_v[idx] = vecotr_dict[w]
    new_v = np.asarray(new_v, dtype='float32')
    return new_v


def make_index_and_save_training_data_as_vector_space(data_path, load=False, preprocess=None):
    if load:
        docs_freq = np.load(data_path+'index.npy', allow_pickle=True).item()
        categories = np.load(data_path+'categories.npy', allow_pickle=True)
        docs_vectors = np.load(data_path+'train_np_vectors.npy', allow_pickle=True)
        return docs_vectors, docs_freq, categories

    json_file = open(data_path+'train.json')
    training_data = json.load(json_file)
    json_file.close()
    docs_freq = {}
    docs_tf = []
    categories = []
    N = 0
    bad_tokens = ['\'', '\"', '.', ',', '?', '!', '[', ']', ')', '(', '@', '#', '$', '%', '^', '&', '*', '+',
                  '=', '<', '>', ';', ':', '-']
    for idx, doc in enumerate(training_data):
        doc_words = tokenize(doc)
        if preprocess is not None:
            doc_words = process_by_nltk(doc_words, preprocess)
        categories.append(doc['category'])
        doc_tf = {}
        for word in doc_words:
            if word in bad_tokens:
                continue
            if word in doc_tf.keys():
                doc_tf[word] += 1
            else:
                doc_tf[word] = 1
            if word in docs_freq.keys():
                if idx not in docs_freq[word]:
                    docs_freq[word].append(idx)
            else:
                docs_freq[word] = [idx]
        docs_tf.append(doc_tf)
        N += 1

    for doc_vector in docs_tf:
        for word in doc_vector.keys():
            doc_vector[word] *= math.log10(N/len(docs_freq[word]))

    docs_tf = np.asarray(docs_tf)
    docs_vectors = np.empty([len(categories), len(docs_freq.keys())], dtype='float32')
    for i in tqdm(range(len(docs_tf))):
        docs_vectors[i] = dict_to_vect(docs_tf[i], docs_freq)
        docs_tf[i] = None


    np.save(data_path+'train_np_vectors', docs_vectors)
    np.save(data_path+'index', np.asarray(docs_freq))
    np.save(data_path+'categories', np.asarray(categories))

    return docs_vectors, np.asarray(docs_freq), np.asarray(categories)


def save_validation_data_as_vector_space(data_path, index, N, load=False, preprocess=None):
    if load:
        return np.load(data_path+'val_np_vectors.npy', allow_pickle=True), np.load(data_path+'val_categories.npy', allow_pickle=True)
    json_file = open(data_path + 'validation.json')
    validation_data = json.load(json_file)
    json_file.close()
    categories = np.array([0 for i in range(len(validation_data))])
    vectors = np.empty([len(validation_data), len(index.keys())], dtype='float32')
    for idx, doc in tqdm(enumerate(validation_data)):
        doc_words = tokenize(doc)
        if preprocess is not None:
            doc_words = process_by_nltk(doc_words, preprocess)
        doc_tf = {}
        for word in doc_words:
            if word in doc_tf.keys():
                doc_tf[word] += 1
            elif word in index.keys():
                doc_tf[word] = 1

        for word in doc_tf.keys():
            if word in index.keys():
                doc_tf[word] *= math.log10(N / len(index[word]))
        doc_tf = dict_to_vect(doc_tf, index)
        vectors[idx] = doc_tf
        categories[idx] = doc['category']

    np.save(data_path+'val_np_vectors', vectors)
    np.save(data_path+'val_categories', np.asarray(categories))
    return vectors, categories


def tokenize(doc):
    bad_tokens = ['\'', '\"', '.', ',', '?', '!', '[', ']', ')', '(', '@', '#', '$', '%', '^', '&', '*', '+',
                  '=', '<', '>', ';', ':']
    body_text = str(doc['body']).lower()
    title_text = doc['title'].lower()
    for bt in bad_tokens:
        body_text = body_text.replace(bt, ' ')
        title_text = title_text.replace(bt, ' ')
    body_words = body_text.split()
    title_words = title_text.split()
    doc_words = body_words + title_words
    return doc_words


def process_by_nltk(words, method='stemming'):
    import nltk
    if method == 'stemming':
        stemmer = nltk.stem.LancasterStemmer()
        for i in range(len(words)):
            words[i] = stemmer.stem(words[i])
        return words
    elif method == 'lemmatization':
        lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        for i in range(len(words)):
            words[i] = lemmatizer.lemmatize(words[i])
        return words
    elif method == 'stopwords':
        from nltk.corpus import stopwords
        stopwords_set = set(stopwords.words('english'))
        new_words = []
        for word in words:
            if word not in stopwords_set:
                new_words.append(word)
        return new_words
