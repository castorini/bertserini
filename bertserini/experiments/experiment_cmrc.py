import json
from tqdm import tqdm
from bertserini.reader.bert_reader import BERT
from bertserini.retriever.pyserini_retriever import retriever, build_searcher
from bertserini.utils.utils_new import extract_squad_questions

if __name__ == "__main__":

    questions = extract_squad_questions("data/cmrc_dev_squad.json")
    bert_reader = BERT("rsvp-ai/bertserini-bert-base-cmrc", "rsvp-ai/bertserini-bert-base-cmrc")
    searcher = build_searcher("index/lucene-index.wiki_zh_paragraph_with_title_0.6.0.pos+docvectors", language="zh")

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
    json.dump(all_answer, open("result_cmrc.json", 'w'), indent=4)
