""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import argparse
import json

from bertserini.utils.utils import normalize_answer, init_logger

logger = init_logger("evluation")


def cover_score(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    return ground_truth in prediction


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def precision_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision


def recall_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return recall

def overlap_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    else:
        return 1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def metric_max_recall(metric_fn, prediction, ground_truths):
    score_recall = []
    for ground_truth in ground_truths:
        score_ground_truth = []
        for predict in prediction:
            score = metric_fn(predict, ground_truth)
            score_ground_truth.append(score)
        score_recall.append(score_ground_truth)
    # print(score_recall) TODO: have empty score?
    try:
        return max(max(score_recall))
    except ValueError:
        return 0


def evaluate(dataset, predictions):
    sentence_cover = precision = cover = sentence_recall = recall = f1 = exact_match = total = overlap = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + str(qa['id']) + ' will receive score 0.'
                    logger.error(message)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = [predictions[qa['id']]]
                #prediction_sentence = predictions[qa['id']]['sentences']
                cover += metric_max_recall(cover_score, prediction, ground_truths)
                exact_match += metric_max_recall(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_recall(
                    f1_score, prediction, ground_truths)
                overlap += metric_max_recall(
                    overlap_score, prediction, ground_truths)
                precision += metric_max_recall(
                    precision_score, prediction, ground_truths)
                recall += metric_max_recall(
                    recall_score, prediction, ground_truths)
                #sentence_recall += metric_max_recall(recall_score, prediction_sentence, ground_truths)
                #sentence_cover += metric_max_recall(cover_score, prediction_sentence, ground_truths)
    logger.info("total: {}".format(total))
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    recall = 100.0 * recall / total
    overlap = 100.0 * overlap / total
    cover = 100.0 * cover / total
    precision = 100.0 * precision / total
    #sentence_recall = 100.0 * sentence_recall / total
    #sentence_cover = 100.0 * sentence_cover / total

    return {'exact_match': exact_match, 'f1': f1, "recall": recall, 
            #"sentence_recall": sentence_recall, "sentence_cover": sentence_cover,
            "precision": precision, "cover": cover, "overlap": overlap}


def squad_v1_eval(dataset_filename, prediction_filename):
    expected_version = '1.1'
    with open(dataset_filename) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != expected_version:
            logger.error('Evaluation expects v-{}, but got dataset with v-{}'.format(
                expected_version, dataset_json['version']))
        dataset = dataset_json['data']
    with open(prediction_filename) as prediction_file:
        predictions = json.load(prediction_file)
    ans = evaluate(dataset, predictions)
    return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file', type=int,
                        help='number of top k paragraphs into bert')
    parser.add_argument('prediction_file', type=str,
                        help='Path to index file')
    args = parser.parse_args()
    squad_v1_eval(args.dataset_file, args.prediction_file)
