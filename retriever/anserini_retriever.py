import os
import numpy as np

import jnius_config
jnius_config.set_classpath("lib/anserini-0.6.0-SNAPSHOT-fatjar.jar")

from jnius import autoclass
JString = autoclass('java.lang.String')
JSearcher = autoclass('io.anserini.search.SimpleSearcher')

from utils import init_logger, strip_accents, normalize_text
#logger = init_logger("anserini_retriever")

def build_searcher(k1=0.9, b=0.4, index_path="index/lucene-index.wiki_paragraph_drqa.pos+docvectors", segmented=False, rm3=False, chinese=False):
    searcher = JSearcher(JString(index_path))
    searcher.setBM25Similarity(k1, b)
    if not rm3:
        searcher.setSearchChinese(chinese)
        searcher.setDefaultReranker()
    else:
        searcher.setRM3Reranker()
    return searcher


def anserini_retriever(question, searcher, para_num=20, tag=""):
    try:
        #hits = searcher.search(JString(question), para_num, JString(tag))
        hits = searcher.search(JString(question.encode("utf-8")), para_num)
    except ValueError as e:
        #logger.error("Search failure: {}, {}".format(question, e))
        print("Search failure: {}, {}".format(question, e))
        return []

    paragraphs = []

    for paragraph in hits:
        if ("||" in paragraph.content) or ("/><" in paragraph.content) or \
           ("|----|" in paragraph.content) or ("#fffff" in paragraph.content):
            continue
        else:
            paragraph_dict = {'text': paragraph.content,
                              'paragraph_score': paragraph.score,
                              'docid': paragraph.docid}
                              #"tag": paragraph.tag}
            paragraphs.append(paragraph_dict)

    return paragraphs
