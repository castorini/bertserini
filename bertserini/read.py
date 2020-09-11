import json
from tqdm import trange, tqdm

from bert_reader import BertReader
from bertserini.args import *
from bertserini.utils import strip_accents


if __name__ == "__main__":
    QAs = convert_squad_to_list("./data/squad_v1.1/dev-v1.1.json")

    bert_reader = BertReader(args)
    all_results = []
    for question_id in trange(len(QAs)):
        question = strip_accents(QAs[question_id]["question"])
        paragraph_texts = [QAs[question_id]["context"]]
        id_ = QAs[question_id]["id"]

        paragraph_scores = [100]

        final_answers = bert_reader.predict(id_, question, paragraph_texts, paragraph_scores)
        print(question, final_answers)

        all_results.append(final_answers)
    json.dump(all_results, open("pytorch_bert_squad.json", 'w'))
