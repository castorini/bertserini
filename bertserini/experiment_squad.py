import json
from tqdm import tqdm
from .readers import BERT
from .pyserini_retriever import retriever, build_searcher
from .utils_new import extract_squad_questions

if __name__ == "__main__":

    questions = extract_squad_questions("data/dev-v1.1.json")
    bert_reader = BERT("rsvp-ai/bertserini-bert-base-squad", "rsvp-ai/bertserini-bert-base-squad")
    searcher = build_searcher("index/lucene-index.enwiki-20180701-paragraphs")

    all_answer = []
    for question in tqdm(questions):
        contexts = retriever(question, searcher, 10)
        final_answers = bert_reader.predict(question, contexts)
        final_answers_lst = []
        for ans in final_answers:
            final_answers_lst.append(
                {"id": question.id,
                 "answer": ans.text,
                 "phrase_score": ans.score,
                 "paragraph_score": ans.ctx_score,
                 }
            )
        all_answer.append(final_answers_lst)
    json.dump(all_answer, open("result_bert_base.json", 'w'))
