from typing import List

import torch
from torch.utils.data import DataLoader, SequentialSampler

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, SquadExample
from transformers.data.processors.squad import SquadResult, squad_convert_examples_to_features
from transformers.models.dpr import DPRReader, DPRReaderTokenizer, DPRQuestionEncoderTokenizer

from bertserini.reader.base import Reader, Question, Context, Answer
from bertserini.utils.utils_squad_metrics import compute_predictions_logits
from bertserini.train.run_squad import to_list

def craft_squad_examples(question: Question, contexts: List[Context]) -> List[SquadExample]:
    examples = []
    for idx, ctx in enumerate(contexts):
        examples.append(
            SquadExample(
                qas_id=idx,
                question_text=question.text,
                context_text=ctx.text,
                answer_text=None,
                start_position_character=None,
                title="",
                is_impossible=False,
                answers=[],
            )
        )
    return examples


class DPR(Reader):
    def __init__(self, args):
        self.model_args = args
        if self.model_args.tokenizer_name is None:
            self.model_args.tokenizer_name = self.model_args.model_name_or_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DPRReader.from_pretrained(self.model_args.model_name_or_path).to(self.device).eval()
        self.tokenizer = DPRReaderTokenizer.from_pretrained(self.model_args.tokenizer_name, do_lower_case=True, use_fast=False)
        self.args = {
            "max_seq_length": 384,
            "doc_stride": 128,
            "max_query_length": 64,
            "threads": 1,
            "tqdm_enabled": False,
            "n_best_size": 20,
            "max_answer_length": 30,
            "do_lower_case": True,
            "output_prediction_file": False,
            "output_nbest_file": self.model_args.output_nbest_file,
            "output_null_log_odds_file": None,
            "verbose_logging": False,
            "version_2_with_negative": True,
            "null_score_diff_threshold": 0,
        }

    def update_args(self, args_to_change):
        for key in args_to_change:
            self.args[key] = args_to_change[key]

    def predict(self, question: Question, contexts: List[Context]) -> List[Answer]:
        examples = craft_squad_examples(question, contexts)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.args["max_seq_length"],
            doc_stride=self.args["doc_stride"],
            max_query_length=self.args["max_query_length"],
            is_training=False,
            return_dataset="pt",
            threads=self.args["threads"],
            tqdm_enabled=self.args["tqdm_enabled"]
        )

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.model_args.eval_batch_size)

        all_results = []

        for batch in eval_dataloader:
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                }
                feature_indices = batch[3]
                outputs = self.model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [outputs[oname][i] for oname in outputs]
                start_logits = outputs.start_logits[i]
                end_logits = outputs.end_logits[i]
                try:
                    start_logits = start_logits.item()
                    end_logits = end_logits.item()
                except:
                    pass
                
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        answers, nbest = compute_predictions_logits(
            all_examples=examples,
            all_features=features,
            all_results=all_results,
            n_best_size=self.args["n_best_size"],
            max_answer_length=self.args["max_answer_length"],
            do_lower_case=self.args["do_lower_case"],
            output_prediction_file=self.args["output_prediction_file"],
            output_nbest_file=self.args["output_nbest_file"],
            output_null_log_odds_file=self.args["output_null_log_odds_file"],
            verbose_logging=self.args["verbose_logging"],
            version_2_with_negative=self.args["version_2_with_negative"],
            null_score_diff_threshold=self.args["null_score_diff_threshold"],
            tokenizer=self.tokenizer,
            language=question.language
        )

        all_answers = []
        for idx, ans in enumerate(nbest):
            all_answers.append(Answer(
                text=nbest[ans][0]["text"],
                score=nbest[ans][0]["start_logit"] + nbest[ans][0]["end_logit"],
                ctx_score=contexts[idx].score,
                language=question.language
            ))
        return all_answers

