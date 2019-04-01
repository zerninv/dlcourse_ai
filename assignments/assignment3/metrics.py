def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for pred, real in zip(prediction, ground_truth):
        true_positive += 1 if pred and real else 0
        true_negative += 1 if not (pred or real) else 0
        false_positive += 1 if pred and not real else 0
        false_negative += 1 if not pred and real else 0

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    accuracy = (true_positive + true_negative) / ground_truth.shape[0]
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    return sum(ground_truth == prediction) / ground_truth.shape[0]