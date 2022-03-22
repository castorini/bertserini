import json
from hanziconv import HanziConv

from bertserini.reader.base import Question
from bertserini.utils.utils import strip_accents


def get_best_answer(candidates, weight=0.5):
    for ans in candidates:
        ans.aggregate_score(weight)
    return sorted(candidates, key=lambda x: x.total_score, reverse=True)[0]


def extract_squad_questions(squad_filename, do_strip_accents=False, language="en"):
    data = json.load(open(squad_filename, 'r'))
    data = data["data"]
    questions = []
    for article in data:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                id_ = qa["id"]
                question = qa["question"]
                if do_strip_accents:
                    question = strip_accents(question)
                if language == "zh":
                    HanziConv.toSimplified(question)
                questions.append(Question(question, id_, language))
    return questions