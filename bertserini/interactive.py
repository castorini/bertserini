from .base import Question
from .readers import BERT
from .pyserini_retriever import retriever, build_searcher

if __name__ == "__main__":

    bert_reader = BERT("rsvp-ai/bertserini-bert-base-squad", "rsvp-ai/bertserini-bert-base-squad")
    searcher = build_searcher("index/lucene-index.enwiki-20180701-paragraphs")

    while True:
        print("Please input your question[use empty line to exit]:")
        question = Question(input())
        contexts = retriever(question.text, searcher, 10)
        answers = bert_reader.predict(question, contexts)
        for ans in answers:
            ans.aggregate_score(0.45)
        answers.sort(key=lambda x: x.total_score, reverse=True)
        print(answers[0].text)



