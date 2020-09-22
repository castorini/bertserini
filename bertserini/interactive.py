from bertserini.reader.base import Question
from bertserini.reader.bert_reader import BERT
from bertserini.retriever.pyserini_retriever import retriever, build_searcher
from bertserini.utils.utils_new import get_best_answer

if __name__ == "__main__":

    bert_reader = BERT("rsvp-ai/bertserini-bert-base-cmrc", "rsvp-ai/bertserini-bert-base-cmrc")
    searcher = build_searcher("index/lucene-index.wiki_zh_paragraph_with_title_0.6.0.pos+docvectors")

    while True:
        print("Please input your question[use empty line to exit]:")
        question = Question(input(), "zh")
        contexts = retriever(question, searcher, 10)
        candidates = bert_reader.predict(question, contexts)
        answer = get_best_answer(candidates, 0.45)
        print(answer.text)



