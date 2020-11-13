from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import json
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from tqdm import tqdm


def training(data_path, text_clf):
    json_file = open(data_path+'train.json')
    training_data = json.load(json_file)
    json_file.close()
    all_docs_texts = []
    categories = []
    for doc in training_data:
        body_text = str(doc['body']).lower()
        title_text = str(doc['title']).lower()
        doc_text = body_text + title_text
        all_docs_texts.append(doc_text)
        categories.append(doc['category'])

    text_clf.fit(all_docs_texts, categories)


def validation(data_path, *args):
    json_file = open(data_path + 'validation.json')
    validation_data = json.load(json_file)
    json_file.close()
    all_docs_texts = []
    labels = []
    for idx, doc in tqdm(enumerate(validation_data)):
        body_text = doc['body']
        title_text = doc['title']
        doc_text = body_text + title_text
        all_docs_texts.append(doc_text)
        labels.append(doc['category'])

    if len(args) == 1:
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LinearSVC(C=args[0]))
        ])
    elif len(args) == 2:
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', RandomForestClassifier(n_estimators=args[0], max_depth=args[1]))
        ])

    training(data_path, text_clf)
    predictions = text_clf.predict(all_docs_texts)
    return predictions, labels