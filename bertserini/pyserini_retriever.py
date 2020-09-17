from pyserini.search import SimpleSearcher
from bertserini.utils import init_logger

logger = init_logger("retriever")


def build_searcher(k1=0.9, b=0.4, index_path="index/lucene-index.wiki_paragraph_drqa.pos+docvectors", chinese=False):
    searcher = SimpleSearcher(index_path)
    searcher.set_bm25(k1, b)
    if chinese:
        searcher.object.setLanguage("zh")
        print("########### we are usinig Chinese retriever ##########")
    return searcher


def retriever(question, searcher, para_num=20):
    try:
        hits = searcher.search(question, k=para_num)
    except ValueError as e:
        logger.error("Search failure: {}, {}".format(question, e))
        return []

    paragraphs = []
    for hit in hits:
        doc_id = hit.docid
        score = hit.score
        text = hit.contents

        if ("||" in text) or ("/><" in text) or \
           ("|----|" in text) or ("#fffff" in text):
            continue
        else:
            paragraph_dict = {'text': text,
                              'paragraph_score': score,
                              'docid': doc_id}
            paragraphs.append(paragraph_dict)

    return paragraphs
