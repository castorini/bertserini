import unicodedata
import string
import logging
import json
import re
import zhon
import numpy as np


def strip_accents(text):
    return "".join(char for char in unicodedata.normalize('NFKD', text)
                   if unicodedata.category(char) != 'Mn')


def choose_best_answer(final_answers, score_computer,
                       paragraph_score_weight, phrase_score_weight, mode="origin"):
    scored_answers = get_voted_answers(final_answers, score_computer,
                                       paragraph_score_weight, phrase_score_weight, mode)
    sorted_answers = sorted(scored_answers, key=lambda k: k['total_score'], reverse=True)

    return sorted_answers[0]


def weighted_score(paragraph_score, phrase_score, paragraph_weight=0.5, phrase_weight=0.5):
    return paragraph_score * paragraph_weight + phrase_score * phrase_weight


def get_type(sent):
    ts = ['Who', 'Why', 'What', 'Which', 'When', 'How', 'Where']
    tp = 'others'

    for token in sent.split():
        for t in ts:
            if token.lower().startswith(t.lower()):
                tp = t
                return tp
    return tp


def get_voted_answers(answerlist, score_computer, paragraph_score_weight, phrase_score_weight, mode="origin"):
    if mode == "origin":
        return get_scored_answers(answerlist, score_computer, paragraph_score_weight, phrase_score_weight)
    elif mode == "ext_origin":
        return get_scored_answers(answerlist[0], score_computer, paragraph_score_weight, phrase_score_weight)
    else:
        answer_dict = {}
        # base_ans = get_scored_answers(answerlist[0], score_computer, paragraph_score_weight, phrase_score_weight)
        # print(base_ans)
        answerlist = answerlist
        # answerlist = [answerlist[0]]
        answers = get_scored_answers(answerlist[0], score_computer, paragraph_score_weight, phrase_score_weight)
        for ans in answers:
            answer_text = ans['answer']
            answer_sentence = ans['sentence']
            answer_score = ans['total_score']
            if answer_text not in answer_dict:
                answer_dict[answer_text] = {
                    "count": 1,
                    "total_scores": [1 * answer_score],
                    "sentences": [answer_sentence],
                    "answer_text": [answer_text]
                }
            else:
                answer_dict[answer_text]['count'] += 1
                answer_dict[answer_text]['total_scores'].append(1 * answer_score)
                answer_dict[answer_text]['sentences'].append(answer_sentence)
                answer_dict[answer_text]['answer_text'].append(answer_text)

        combined_answers = []
        for ans in answer_dict:
            new_answer = {
                "answer": answer_dict[ans]['answer_text'][0],
                # "count": answer_dict[ans]['count'],
                # "total_score": answer_dict[ans]['count'], 
                "total_score": np.sum(answer_dict[ans]['total_scores']),
                "sentence": answer_dict[ans]['sentences'],
                "best_score": np.max(answer_dict[ans]['total_scores'])
            }
            combined_answers.append(new_answer)

        return combined_answers


def get_scored_answers(final_answers, score_computer, paragraph_score_weight, phrase_score_weight):
    scored_answers = []
    for ans_id, ans in enumerate(final_answers):
        # print(ans)
        paragraph_score = ans['paragraph_score']
        phrase_score = ans['phrase_score']
        total_score = score_computer(paragraph_score, phrase_score, paragraph_score_weight, phrase_score_weight)
        new_answer = ans
        new_answer['total_score'] = total_score
        scored_answers.append(new_answer)
        # break
    return scored_answers


def convert_squad_to_list(squad_filename):
    data = json.load(open(squad_filename, 'r'))
    data = data["data"]
    converted_data = []
    for article in data:
        for paragraph in article["paragraphs"]:
            text = paragraph["context"]
            for qa in paragraph["qas"]:
                id_ = qa["id"]
                question = qa["question"]
                answers = qa["answers"]
                converted_data.append({"id": id_, "question": question, "answers": answers, "context": text})
    return converted_data


def init_logger(bot):
    # create logger with 'spam_application'
    # bot = 'server_cn' if args.chinese else 'server_en'
    logger = logging.getLogger('{} log'.format(bot))
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('{}.log'.format(bot))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def split_title(paragraph):
    sents = paragraph.split(".")
    text = ".".join(sents[1:]).strip()
    title = sents[0].strip()
    return title, text


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_text(s):
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return remove_punc(lower(s))


def normalize_chinese_text(s):
    def remove_punc(text):
        exclude = set(zhon.hanzi.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return remove_punc(s)
