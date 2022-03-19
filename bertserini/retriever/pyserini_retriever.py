from typing import List
import json

from pyserini.search.lucene import LuceneSearcher, JLuceneSearcherResult
from bertserini.utils.utils import init_logger
from bertserini.reader.base import Context

logger = init_logger("retriever")


def build_searcher(index_path, k1=0.9, b=0.4, language="en"):
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(k1, b)
    searcher.object.setLanguage(language)
    return searcher

def build_searcher_from_prebuilt_index(index_name, k1=0.9, b=0.4, language="en"):
    searcher = LuceneSearcher.from_prebuilt_index(index_name)
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


def hits_to_contexts(hits: List[JLuceneSearcherResult], language="en", field='raw', blacklist=[]) -> List[Context]:
    """
        Converts hits from Pyserini into a list of texts.
        Parameters
        ----------
        hits : List[JLuceneSearcherResult]
            The hits.
        field : str
            Field to use.
        language : str
            Language of corpus
        blacklist : List[str]
            strings that should not contained
        Returns
        -------
        List[Text]
            List of texts.
     """
    contexts = []
    for i in range(0, len(hits)):
        t = hits[i].raw if field == 'raw' else hits[i].contents
        try: # the previous chinese index stores the contents as "raw", while the english index stores the json string.
            t = json.loads(t)["contents"]
        except:
            pass
        for s in blacklist:
            if s in t:
                continue
        metadata = {'raw': hits[i].raw, 'docid': hits[i].docid}
        contexts.append(Context(t, language, metadata, hits[i].score))
    return contexts
