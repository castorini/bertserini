import json
import time
import unicodedata
from tqdm import trange, tqdm

from hanziconv import HanziConv

from run_squad import BertReader
#from retriever.anserini_retriever import anserini_retriever, build_searcher
from retriever.pyserini_retriever import anserini_retriever, build_searcher
from utils import (convert_squad_to_list, normalize_text, init_logger, strip_accents)

from args import *

if __name__ == "__main__":
    """
        Connect anserini with bert.
        Question from SQuAD1.0-dev
        search paragraph using Anserini (Top 1)
        extract phrase using Bert (SQuAD1.0 pretrained version)
    """
    #logger = init_logger("bert_search")

    QAs = convert_squad_to_list(args.predict_file)

    bert_reader = BertReader(args)
    ansrini_searcher = build_searcher(args.k1, args.b, args.index_path, args.rm3, chinese=args.chinese)

    count_hit = [0] * (args.para_num)
    count_total = [0] * (args.para_num)

    all_results = []

    for question_id in trange(len(QAs)):
        start_time = time.time()
        question = strip_accents(QAs[question_id]['question']) # convert Latin into English
        if args.chinese:
            if args.toSimplified:
                question = HanziConv.toSimplified(question)
            paragraphs = anserini_retriever(question, ansrini_searcher, args.para_num)
        else:
            paragraphs = anserini_retriever(question, ansrini_searcher, args.para_num)
        if len(paragraphs) == 0:
            continue
        paragraph_texts = []
        paragraph_scores = []
        hit_flag = False
        for paragraph_id, paragraph in enumerate(paragraphs):
            paragraph_texts.append(paragraph['text'])
            paragraph_scores.append(paragraph['paragraph_score'])
            count_total[paragraph_id] += 1
            if hit_flag:
                count_hit[paragraph_id] += 1
                continue
            for k in range(len(QAs[question_id]['answers'])):
                if normalize_text(QAs[question_id]['answers'][k]["text"]) in normalize_text(paragraph['text']):
                    count_hit[paragraph_id] += 1
                    hit_flag = True
                    break
        
        final_answers = bert_reader.predict(QAs[question_id]['id'], question, paragraph_texts, paragraph_scores)

        all_results.append(final_answers)
        print(final_answers)

    json.dump(all_results, open(args.output_fn, 'w'))

    #logger.info("=======================================")
    #logger.info("count_total {} count_hit {}".format(count_total, count_hit))
    json.dump([count_total, count_hit], open("count_{}.json".format(args.output_fn), "w"))
    #logger.info("=======================================")

