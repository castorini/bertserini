import os
import numpy as np

from pyserini.search import SimpleSearcher

from utils import init_logger, strip_accents, normalize_text
logger = init_logger("anserini_retriever")

def build_searcher(k1=0.9, b=0.4, index_path="index/lucene-index.wiki_paragraph_drqa.pos+docvectors", segmented=False, rm3=False, chinese=False):
    searcher = SimpleSearcher(index_path)
    searcher.set_bm25(k1, b)
    if chinese:
        searcher.object.setLanguage("zh")
        print("########### we are usinig Chinese retriever ##########")
    return searcher

def anserini_retriever(question, searcher, para_num=20, tag=""):
    try:
        hits = searcher.search(question, k=para_num)
    except ValueError as e:
        logger.error("Search failure: {}, {}".format(question, e))
        return []

    paragraphs = []

    for paragraph in hits:
        if ("||" in paragraph.raw) or ("/><" in paragraph.raw) or \
           ("|----|" in paragraph.raw) or ("#fffff" in paragraph.raw):
            continue
        else:
            paragraph_dict = {'text': paragraph.raw,
                              'paragraph_score': paragraph.score,
                              'docid': paragraph.docid}
                              #"tag": paragraph.tag}
            paragraphs.append(paragraph_dict)

    return paragraphs
