from bertserini.reader.base import Question, Context
from bertserini.reader.bert_reader import BERT
#from pygaggle.qa.dpr_reader import DprReader
from bertserini.reader.dpr_reader import DprReader, DprSelection
from bertserini.utils.utils_new import get_best_answer
from bertserini.experiments.args import *
from bertserini.retriever.pyserini_retriever import retriever, build_searcher

do_english_test = True
do_local_test = True
do_bm25_test = False
do_dpr_test = False
do_chinese_test = False

if do_english_test:
    #args.model_name_or_path = "rsvp-ai/bertserini-bert-base-squad"
    #args.tokenizer_name = "rsvp-ai/bertserini-bert-base-squad"
    args.model_name_or_path = "facebook/dpr-reader-single-nq-base"
    args.tokenizer_name = "facebook/dpr-reader-single-nq-base"
    #bert_reader = BERT(args)
    bert_reader = DprReader(
        args.model_name_or_path, 
        args.tokenizer_name, 
        [DprSelection()], 
        1, # num_spans,
        100, #max_answer_length
        10, # max number of answer spans per passage
        4, #batch size
        "cuda:0")
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
    searcher_config = {
        "k1": 0.9,
        "b": 0.4,
        "index": "/data/y247xie/01_exps/anserini/lucene-index.ik-nlp-22",
        "language": "en",
    }
    searcher = build_searcher("bm25", searcher_config)
    contexts = retriever(question, searcher, 10)
    candidates = bert_reader.predict(question, contexts)
    answer = get_best_answer(candidates, 0.45)
    print("Answer:", answer.text)
    print("BM25 Test Passed")

if do_dpr_test:
    print("######################### Testing DPR Context #########################")
    searcher_config = {
        "encoder": "facebook/dpr-question_encoder-single-nq-base",
        "tokenizer_name": "facebook/dpr-question_encoder-single-nq-base",
        "index": "/data/y247xie/01_exps/pyserini/dpr-ctx_encoder-multiset-base.ik-nlp-22_slp",
        #"index": "/data/y247xie/01_exps/pyserini/dpr-ctx_encoder-multiset-base.mesh-2021-abs",
        "device": "cuda:0",
    #    "sparse_index": "/data/y247xie/01_exps/anserini/lucene-index.mesh-2021-abs",
        "sparse_index": "/data/y247xie/01_exps/anserini/lucene-index.ik-nlp-22",
    }
    searcher = build_searcher("dpr", searcher_config)
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
    searcher_config = {
        "k1": 0.9,
        "b": 0.4,
        "index": "./indexes/lucene-index.zhwiki-20181201-paragraphs",
        "language": "zh",
    }
    question = Question("《战国无双3》是由哪两个公司合作开发的？")
    searcher = build_searcher("bm25", searcher_config)
    contexts = retriever(question, searcher, 10)
    candidates = bert_reader.predict(question, contexts)
    answer = get_best_answer(candidates, 0.45)
    print("Answer:", answer.text)
    print("BM25 Chinese Test Passed")
