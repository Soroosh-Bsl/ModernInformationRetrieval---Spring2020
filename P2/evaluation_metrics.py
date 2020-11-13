import numpy as np


def cosine_sim(a, b):
    dot_product = 0
    a_l = 0
    b_l = 0
    for w in a.keys():
        if w in b.keys():
            dot_product += a[w] * b[w]
        a_l += a[w]*a[w]
    for w in b.keys():
        b_l += b[w]*b[w]
    a_l = np.sqrt(a_l)
    b_l = np.sqrt(b_l)
    l = a_l*b_l
    return dot_product/l


def euclidean_dist(a, b):
    dist = 0
    for w in a.keys():
        if w in b.keys():
            dist += (a[w]-b[w])**2
        else:
            dist += a[w]**2

    for w in b.keys():
        if w not in a.keys():
            dist += b[w]**2
    return dist


def precision_metric(predictions, labels, category):
    tp = 0
    fp = 0

    for i, p in enumerate(predictions):
        if p != category:
            continue
        elif p == labels[i]:
            tp += 1
        elif p != labels[i]:
            fp += 1
    return tp/(tp+fp+1e-20)


def recall_metric(predictions, labels, category):
    tp = 0
    fn = 0
    for i, l in enumerate(labels):
        if l != category:
            continue
        elif l == predictions[i]:
            tp += 1
        elif l != predictions[i]:
            fn += 1
    return tp/(tp+fn+1e-20)


def accuracy_metric(predictions, labels):
    true = 0
    for i, p in enumerate(predictions):
        if p == labels[i]:
            true += 1
    return true/len(predictions)


def confusion_matrix_metric(predictions, labels):
    table = [[0 for j in range(4)] for i in range(4)]
    for i, p in enumerate(predictions):
        table[p-1][labels[i]-1] += 1
    return table


def calc_f(predictions, labels, category, beta=1):
    precision = precision_metric(predictions, labels, category)
    recall = recall_metric(predictions, labels, category)
    return ((1+beta**2)*precision+recall)/((beta**2)*precision+recall+1e-20)


def macro_averaged_f_metric(predictions, labels, beta=1):
    sum_f1s = 0
    for c in range(1, 5):
        sum_f1s += calc_f(predictions, labels, c, beta=beta)
    return sum_f1s/4


def evaluate(predictions, labels):
    precisions = []
    recalls = []
    for i in range(1, 5):
        precisions.append(precision_metric(predictions, labels, i))
        recalls.append(recall_metric(predictions, labels, i))
    acc = accuracy_metric(predictions, labels)
    confusion_matrix = confusion_matrix_metric(predictions, labels)
    macro_averaged_f = macro_averaged_f_metric(predictions, labels, beta=1)

    processed_results = ""
    for i in range(1, 5):
        processed_results += "Precision For Class{} = {}\n".format(i, precisions[i-1])
    processed_results += '-------------------------------------------------\n'
    for i in range(1, 5):
        processed_results += "Recall For Class{} = {}\n".format(i, recalls[i - 1])
    processed_results += '-------------------------------------------------\n'
    processed_results += "Accuracy = {}\n".format(acc)
    processed_results += '-------------------------------------------------\n'
    processed_results += "Confusion Matrix:\n"
    processed_results += "     "
    for j in range(4):
        processed_results += "{}    ".format(j+1)
    processed_results += "\n-------------------------\n"
    for i in range(4):
        processed_results += "{}|   ".format(i+1)
        for j in range(4):
            processed_results += "{}    ".format(confusion_matrix[i][j])
        processed_results += "\n"
    processed_results += '-------------------------------------------------\n'
    processed_results += "Macro Averaged F1 Score = {}\n".format(macro_averaged_f)

    return processed_results
