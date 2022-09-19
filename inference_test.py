from bertserini.reader.base import Question, Context
from bertserini.reader.bert_reader import BERT
from bertserini.reader.dpr_reader import DPR
from bertserini.reader.t5_reader import T5
from bertserini.utils.utils import get_best_answer
from bertserini.experiments.args import *
from bertserini.retriever.pyserini_retriever import retriever, build_searcher

ENG_reader = "T5"
do_local_test = True
do_bm25_test = True
do_dpr_test = False
do_chinese_test = False

if ENG_reader == "BERT":
    args.model_name_or_path = "rsvp-ai/bertserini-bert-base-squad"
    args.tokenizer_name = "rsvp-ai/bertserini-bert-base-squad"
    bert_reader = BERT(args)

elif ENG_reader == "DPR":
    args.model_name_or_path = "facebook/dpr-reader-multiset-base"
    args.tokenizer_name = "facebook/dpr-reader-multiset-base"
    bert_reader = DPR(args)

elif ENG_reader == "T5":
    args.model_name_or_path = "/data/aileen/workspace/t5_test2/models/gpu/checkpoint-10500"
    args.tokenizer_name = "t5-base"
    bert_reader = T5(args)

# question = Question("Why did Mark Twain call the 19th century the glied age?")
question = Question("What is the capital city of China?")

print(question.text)

if do_local_test:
    print("######################### Testing Local Context #########################")
    contexts = [Context('The "Gilded Age" was a term that Mark Twain used to describe the period of the late 19th century when there had been a dramatic expansion of American wealth and prosperity.'),
                Context('The "Gilded Age"')]
    candidates = bert_reader.predict(question, contexts)
    answer = get_best_answer(candidates, 1.0)
    print("Answer:", answer.text)

if do_bm25_test:
    print("######################### Testing BM25 Context #########################")
    args.index_path = "./indexes/lucene-index.enwiki-20180701-paragraphs"
    searcher = build_searcher(args)
    contexts = retriever(question, searcher, 10)
    candidates = bert_reader.predict(question, contexts)
    print(candidates)
    answer = get_best_answer(candidates, 0.45)
    print("Answer:", answer.text) # todo: no context returned. is the context included? maybe update to another question
    # print("BM25 Test Passed")

if do_dpr_test:
    print("######################### Testing DPR Context #########################")
    args.retriever = "dpr"
    args.encoder = "facebook/dpr-question_encoder-multiset-base"
    args.query_tokenizer_name = "facebook/dpr-question_encoder-multiset-base"
    args.index_path = "../pyserini/dpr-ctx_encoder-multiset-base.ik-nlp-22_slp" # todo: replicate dpr on wiki and release dpr-indexes
    args.device = "cuda:cpu"
    args.sparse_index = "../anserini/lucene-index.ik-nlp-22"
    searcher = build_searcher(args)
    contexts = retriever(question, searcher, 10)
    candidates = bert_reader.predict(question, contexts)
    answer = get_best_answer(candidates, 0.45)
    print("Answer:", answer.text)
    print("DPR Test Passed")

if do_chinese_test:
    print("######################### Testing BM25 Chinese #########################")
    args.model_name_or_path = "rsvp-ai/bertserini-bert-base-cmrc"
    args.tokenizer_name = "rsvp-ai/bertserini-bert-base-cmrc"
    bert_reader = BERT(args)
    args.index_path = "./indexes/lucene-index.zhwiki-20181201-paragraphs"
    args.language = "zh"
    args.retriever = "bm25"
    question = Question("《战国无双3》是由哪两个公司合作开发的？")
    searcher = build_searcher(args)
    contexts = retriever(question, searcher, 10)
    candidates = bert_reader.predict(question, contexts)
    answer = get_best_answer(candidates, 0.45)
    print("Answer:", answer.text)
    print("BM25 Chinese Test Passed")
