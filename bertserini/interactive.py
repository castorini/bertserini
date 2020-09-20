from .base import Question
from .readers import BERT
from .pyserini_retriever import retriever, build_searcher

if __name__ == "__main__":

    bert_reader = BERT("rsvp-ai/bertserini-bert-base-cmrc", "rsvp-ai/bertserini-bert-base-cmrc")
    searcher = build_searcher("index/lucene-index.wiki_zh_paragraph_with_title_0.6.0.pos+docvectors")

    while True:
        print("Please input your question[use empty line to exit]:")
        question = Question(input(), "zh")
        contexts = retriever(question, searcher, 10)
        answers = bert_reader.predict(question, contexts)
        for ans in answers:
            ans.aggregate_score(0.45)
        answers.sort(key=lambda x: x.total_score, reverse=True)
        print(answers[0].text)



