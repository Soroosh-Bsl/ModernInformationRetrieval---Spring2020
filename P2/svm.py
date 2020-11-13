from evaluation_metrics import evaluate
from svm_and_random_forest_commons import validation


def search_best_c(data_path):
    results = {}
    for C in [i/10 for i in range(1, 21)]:
        predictions, labels = validation(data_path, C)
        results[C] = evaluate(predictions, labels)
    f = open('SVM_with_different_Cs.txt', 'w+')
    for C in results.keys():
        print("##############################\nRESULTS FOR C={}\n".format(C),
              results[C], file=f)
    f.close()


# search_best_c('./data/')
