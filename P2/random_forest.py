from evaluation_metrics import evaluate
from svm_and_random_forest_commons import validation


def search_best_n_est_and_max_depth(data_path):
    results = {}
    for num_estimators in [i for i in range(20, 200, 20)]:
        for max_depth in [i for i in range(100, 500, 50)]:
            predictions, labels = validation(data_path, num_estimators, max_depth)
            results[(num_estimators, max_depth)] = evaluate(predictions, labels)
    f = open('random_forest_with_different_parameters.txt', 'w+')
    for k in results.keys():
        print("##############################\nRESULTS FOR (N_EST, MAX_DEPTH)={}\n".format(k),
              results[k], file=f)
    f.close()


# search_best_n_est_and_max_depth('./data/')
