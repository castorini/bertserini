import json
import argparse
import os, time, random

# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
from tqdm import trange
from tqdm import tqdm
import numpy as np

from utils import choose_best_answer, weighted_score, normalize_answer, normalize_text, get_type

from eval.evaluate_v1 import squad_v1_eval as squad_evaluation
#from eval.trivia_eval import evaluation as trivia_evaluation
from eval.evaluate_v1_drcd import evaluation as drcd_evaluation
from eval.evaluate_v1_cmrc import evaluate as cmrc_evaluation
#from eval.evaluate_v1_special import evaluation as special_evaluation

def get_score_with_results(eval_data, predictions, mu, dataset):
    answers = {}
    score = {}

    for predict_id, predict in enumerate(predictions):

        try:
            if dataset == "trivia":
                id_ = predict[0]['id'].split("--")[0]
            else:
                id_ = predict[0]['id']
        except IndexError as e:
            pass

        if not predict:
            continue

        best_answer = choose_best_answer(
            predict,
            weighted_score,
            1 - mu, mu)

        answers[id_] = best_answer['answer'].replace("##", "")

        score[id_] = best_answer["total_score"]

    json.dump(answers, open("tmp.answer", 'w'))
    json.dump(score, open("tmp.score", 'w'))

    if dataset == "squad":
        eval_result = squad_evaluation(eval_data, "tmp.answer")
    elif dataset == "cmrc":
        eval_result = cmrc_evaluation(eval_data, "tmp.answer")
        eval_result = {"f1_score": eval_result[0],
                        "exact_match": eval_result[1],
                        "total_count": eval_result[2],
                        "skip_count": eval_result[3]}
    elif args.dataset == "trivia":
        eval_result = trivia_evaluation(eval_data, "tmp.answer")
    elif args.dataset == "drcd":
        eval_result = drcd_evaluation(eval_data, "tmp.answer")
    elif args.dataset == "special":
        eval_result = special_evaluation(eval_data, "tmp.answer")
    else:
        eval_result = squad_evaluation(eval_data, "tmp.answer")

    print("mu:{}, result:{}".format(mu, eval_result))
    return eval_result, answers

def get_best_mu_with_scores(eval_data, predictions, mu_range, dataset, output_path):
    score_test = {}
    best_mu = 0
    best_em = 0
    for mu in mu_range:
        eval_result, answers = get_score_with_results(eval_data, predictions, mu, dataset)
        score_test[mu] = eval_result
        if eval_result["exact_match"] > best_em:
            best_mu = mu
            best_em = eval_result['exact_match']
            json.dump(answers, open(output_path + "/prediction.json", 'w'))

    json.dump(score_test, open(output_path + "/score.json", 'w'))
    return best_mu, score_test[best_mu]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', type=str, default="",
                        help='Path to question data')
    parser.add_argument('--search_file', type=str, default="",
                        help='Path to bert output')
    parser.add_argument("--para_num", type=int, default=100,
                        help="top k paragraphs to eval")
    parser.add_argument("--dataset", type=str, default="squad",
                        help="")
    parser.add_argument("--output_path", type=str, default="",
                        help="")
    parser.add_argument("--mu_min", type=float, default=0)
    parser.add_argument("--mu_max", type=float, default=1)
    parser.add_argument("--mu_interval", type=float, default=0.1)
    args = parser.parse_args()
    print(args)

    predictions = json.load(open(args.search_file, 'r'))

    answers = {}
    cover = 0
    print("Total predictions:", len(predictions))

    predictions = [p[:args.para_num] for p in predictions]

    print(get_best_mu_with_scores(args.eval_data, predictions,
                                  np.arange(args.mu_min, args.mu_max, args.mu_interval),
                                  args.dataset, args.output_path))
