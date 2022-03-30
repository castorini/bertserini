from typing import List
import json

from pyserini.search import FaissSearcher, DprQueryEncoder
from pyserini.search.lucene import LuceneSearcher, JLuceneSearcherResult
from bertserini.utils.utils import init_logger
from bertserini.reader.base import Context

logger = init_logger("retriever")


def build_searcher(args):
    if args.retriever == "bm25":
        searcher = LuceneSearcher(args.index_path)
        searcher.set_bm25(args.k1, args.b)
        searcher.object.setLanguage(args.language)
    elif args.retriever == "dpr":
        query_encoder = DprQueryEncoder(
            encoder_dir=args.encoder,
            tokenizer_name=args.query_tokenizer_name,
            device=args.device)
        searcher = FaissSearcher(args.index_path, query_encoder)
        ssearcher = LuceneSearcher(args.sparse_index)
        searcher.ssearcher = ssearcher
    else:
        raise Exception("Non-Defined Retriever:", args.retriever)
    return searcher

def build_searcher_from_prebuilt_index(args):
    if args.retriever == "bm25":
        searcher = LuceneSearcher.from_prebuilt_index(args.index_path)
        searcher.set_bm25(args.k1, args.b)
        searcher.object.setLanguage(args.language)
    else:
        raise Exception("Not implemented regriever from prebuilt index:", args.retirever)
    return searcher


def retriever(question, searcher, para_num=20):
    language = question.language
    if type(searcher) == FaissSearcher:
        results = searcher.search(question.text, k=para_num)
        hits = []
        for r in results:
            hit = searcher.doc(r.docid).get("raw")
            hits.append((hit, r.score))
    else:
        try:
            if language == "zh":
                hits = searcher.search(question.text.encode("utf-8"), k=para_num)
            else:
                hits = searcher.search(question.text, k=para_num)
        except ValueError as e:
            logger.error("Search failure: {}, {}".format(question.text, e))
            return []
        hits = [(h.raw, h.score) for h in hits]
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
        hit, score = hits[i]
        try: # the previous chinese index stores the contents as "raw", while the english index stores the json string.
            t = json.loads(hit)["contents"]
        except:
            t = hit
        for s in blacklist:
            if s in t:
                continue
        metadata = {}
        contexts.append(Context(t, language, metadata, score))
    return contexts
