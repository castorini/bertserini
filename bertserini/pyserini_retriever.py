from pyserini.search import SimpleSearcher
from .utils import init_logger
from .base import hits_to_contexts

logger = init_logger("retriever")


def build_searcher(index_path, k1=0.9, b=0.4, language="en"):
    searcher = SimpleSearcher(index_path)
    searcher.set_bm25(k1, b)
    searcher.object.setLanguage(language)
    return searcher


def retriever(question, searcher, para_num=20):
    try:
        hits = searcher.search(question, k=para_num)
    except ValueError as e:
        logger.error("Search failure: {}, {}".format(question, e))
        return []

    """
    paragraphs = []
    for hit in hits:
        doc_id = hit.docid
        score = hit.score
        text = hit.raw

        if ("||" in text) or ("/><" in text) or \
           ("|----|" in text) or ("#fffff" in text):
            continue
        else:
            paragraph_dict = {'text': text,
                              'paragraph_score': score,
                              'docid': doc_id}
            paragraphs.append(paragraph_dict)
            """

    return hits_to_contexts(hits)
