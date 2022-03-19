
from bertserini.reader.base import Question, Context
from bertserini.reader.bert_reader import BERT
from bertserini.utils.utils_new import get_best_answer

model_name = "rsvp-ai/bertserini-bert-base-squad"
tokenizer_name = "rsvp-ai/bertserini-bert-base-squad"
bert_reader = BERT(model_name, tokenizer_name)

question = Question("Why did Mark Twain call the 19th century the glied age?")

contexts = [Context('The "Gilded Age" was a term that Mark Twain used to describe the period of the late 19th century when there had been a dramatic expansion of American wealth and prosperity.')]

candidates = bert_reader.predict(question, contexts)
answer = get_best_answer(candidates, 0.45)
print(answer.text)
print("local context passed")

from bertserini.retriever.pyserini_retriever import retriever, build_searcher
searcher = build_searcher("indexes/lucene-index.enwiki-20180701")
contexts = retriever(question, searcher, 10)
candidates = bert_reader.predict(question, contexts)
answer = get_best_answer(candidates, 0.45)
print(answer.text)
print("e2e context passed")

