import json
import time
import unicodedata

from run_squad_new import BertReader
from retriever.anserini_retriever import anserini_retriever, build_searcher
#from retriever.pyserini_retriever import anserini_retriever, build_searcher
from utils import (convert_squad_to_list, normalize_text, strip_accents, choose_best_answer, weighted_score)

from args import *

if __name__ == "__main__":

    bert_reader = BertReader(args)
    ansrini_searcher = build_searcher(args.k1, args.b, args.index_path, args.rm3, chinese=args.chinese)

    while True:
        print("Please input your question[use empty line to exit]:")
        question = input()
        if len(question.strip()) == 0:
            break
        if args.chinese:
            paragraphs = anserini_retriever(question.encode("utf-8"), ansrini_searcher, args.para_num)
        else:
            paragraphs = anserini_retriever(question, ansrini_searcher, args.para_num)
        if len(paragraphs) == 0:
            print("No related Wiki passage found")
        paragraph_texts = []
        paragraph_scores = []
        for paragraph_id, paragraph in enumerate(paragraphs):
            paragraph_texts.append(paragraph['text'])
            paragraph_scores.append(paragraph['paragraph_score'])
        #print(paragraph_texts[:3])
        
        final_answers = bert_reader.predict(0, question, paragraph_texts, paragraph_scores)
        mu = 0.45
        best_answer = choose_best_answer(final_answers, weighted_score, 1-mu, mu)
        #print(final_answers)
        #print(best_answer)
        #{'id': 0, 'answer': '1982', 'phrase_score': 14.186316013336182, 'paragraph_score': 9.100600242614746, 'total_score': 11.389172339439394}
        print("Answer:{}\tTotal Score:{:.2f}\tParagraph Score:{:.2f}\tPhrase Score:{:.2f}".format(
            best_answer["answer"], best_answer["total_score"], best_answer["paragraph_score"], best_answer["phrase_score"]))

