from bertserini.reader.base import Question, Context
from bertserini.reader.bert_reader import BERT
from bertserini.utils.utils_new import get_best_answer
from bertserini.experiments.args import *
from bertserini.retriever.pyserini_retriever import retriever, build_searcher

do_english_test = True
do_local_test = False
do_bm25_test = False
do_dpr_test = False
do_chinese_test = True

if do_english_test:
    args.model_name_or_path = "rsvp-ai/bertserini-bert-base-squad"
    args.tokenizer_name = "rsvp-ai/bertserini-bert-base-squad"
    bert_reader = BERT(args)
    print("Question: Why did Mark Twain call the 19th century the glied age?")

if do_local_test:
    print("######################### Testing Local Context #########################")
    question = Question("Why did Mark Twain call the 19th century the glied age?")
    contexts = [Context('The "Gilded Age" was a term that Mark Twain used to describe the period of the late 19th century when there had been a dramatic expansion of American wealth and prosperity.')]
    candidates = bert_reader.predict(question, contexts)
    answer = get_best_answer(candidates, 1.0)
    print("Answer:", answer.text)
    print("Local Context Test Passed")

if do_bm25_test:
    print("######################### Testing BM25 Context #########################")
    args.index_path = "/data/y247xie/01_exps/anserini/lucene-index.ik-nlp-22"
    searcher = build_searcher(args)
    contexts = retriever(question, searcher, 10)
    candidates = bert_reader.predict(question, contexts)
    answer = get_best_answer(candidates, 0.45)
    print("Answer:", answer.text)
    print("BM25 Test Passed")

if do_dpr_test:
    print("######################### Testing DPR Context #########################")
    args.retriever = "dpr"
    args.encoder = "facebook/dpr-question_encoder-multiset-base"
    args.query_tokenizer_name = "facebook/dpr-question_encoder-multiset-base"
    args.index_path = "/data/y247xie/01_exps/pyserini/dpr-ctx_encoder-multiset-base.ik-nlp-22_slp"
    args.device = "cuda:0"
    args.sparse_index = "/data/y247xie/01_exps/anserini/lucene-index.ik-nlp-22"
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
    question = Question("《战国无双3》是由哪两个公司合作开发的？")
    searcher = build_searcher(args)
    contexts = retriever(question, searcher, 10)
    candidates = bert_reader.predict(question, contexts)
    answer = get_best_answer(candidates, 0.45)
    print("Answer:", answer.text)
    print("BM25 Chinese Test Passed")
