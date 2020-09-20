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
    language = question.language
    try:
        if language == "zh":
            hits = searcher.search(question.text.encode("utf-8"), k=para_num)
        else:
            hits = searcher.search(question.text, k=para_num)
    except ValueError as e:
        logger.error("Search failure: {}, {}".format(question.text, e))
        return []
    return hits_to_contexts(hits, language)
