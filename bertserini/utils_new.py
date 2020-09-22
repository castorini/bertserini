import json
from .base import Question, Answer


def get_best_answer(candidates, weight=0.5):
    for ans in candidates:
        ans.aggregate_score(weight)
    return candidates.sorted(key=lambda x: x.total_score, reverse=True)[0]


def extract_squad_questions(squad_filename, language="en"):
    data = json.load(open(squad_filename, 'r'))
    data = data["data"]
    questions = []
    for article in data:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                id_ = qa["id"]
                question = qa["question"]
                questions.append(Question(question, id_, language))
    return questions

