from typing import List

import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator, EvalPrediction
from datasets import Dataset
import numpy as np

from bertserini.utils.squad import SquadExample
from bertserini.utils.utils_qa import postprocess_qa_predictions
from bertserini.reader.base import Reader, Question, Context, Answer

__all__ = ['BERT']

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


class BERT(Reader):
    def __init__(self, args):
        self.model_args = args
        if self.model_args.tokenizer_name is None:
            self.model_args.tokenizer_name = self.model_args.model_name_or_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_args.model_name_or_path).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.tokenizer_name, do_lower_case=True)
        self.args = {
            "max_seq_length": 384,
            "doc_stride": 128,
            "max_query_length": 64,
            "threads": 1,
            "tqdm_enabled": False,
            "n_best_size": 20,
            "max_answer_length": 384,
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

        def prepare_validation_features(examples):
            question_column_name = "question"
            context_column_name = "context"
            # answer_column_name = "answers" if "answers" in column_names else column_names[2]
            # Some of the questions have lots of whitespace on the left, which is not useful and will make the
            # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
            # left whitespace
            examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            tokenized_examples = self.tokenizer(
                examples[question_column_name if self.args["pad_on_right"] else context_column_name],
                examples[context_column_name if self.args["pad_on_right"] else question_column_name],
                truncation="only_second" if self.args["pad_on_right"] else "only_first",
                max_length=self.args["max_seq_length"],
                stride=self.args["doc_stride"],
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                verbose=False,
                padding="max_length",
            )

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id and we will store the offset mappings.
            tokenized_examples["example_id"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if self.args["pad_on_right"] else 0

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])

                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]
            # print(tokenized_examples)
            return tokenized_examples

        def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
            """
            Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

            Args:
                start_or_end_logits(:obj:`tensor`):
                    This is the output predictions of the model. We can only enter either start or end logits.
                eval_dataset: Evaluation dataset
                max_len(:obj:`int`):
                    The maximum length of the output tensor. ( See the model.eval() part for more details )
            """

            step = 0
            # create a numpy array and fill it with -100.
            logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
            # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
            for i, output_logit in enumerate(start_or_end_logits):  # populate columns
                # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
                # And after every iteration we have to change the step

                batch_size = output_logit.shape[0]
                cols = output_logit.shape[1]

                if step + batch_size < len(dataset):
                    logits_concat[step: step + batch_size, :cols] = output_logit
                else:
                    logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

                step += batch_size

            return logits_concat

        def post_processing_function(examples, features, predictions, stage="eval"):
            # Post-processing: we match the start logits and end logits to answers in the original context.
            _, all_nbest_json = postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                version_2_with_negative=self.args["version_2_with_negative"],
                n_best_size=self.args["n_best_size"],
                max_answer_length=self.args["max_answer_length"],
                null_score_diff_threshold=self.args["null_score_diff_threshold"],
                output_dir="./tmp/",
                # output_dir=self.args["output_dir"],
                # log_level=log_level,
                prefix=stage,
            )
            # print(predictions)
            # print(all_nbest_json)
            # Format the result to the format the metric expects.
            # if self.args["version_2_with_negative"]:
            #     formatted_predictions = [
            #         {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            #     ]
            # else:
            #     formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

            # references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
            # return EvalPrediction(predictions=formatted_predictions)#, label_ids=references)
            return all_nbest_json

        inputs = {"question": [], "context": [], "id": []}
        for i, ctx in enumerate(contexts):
            inputs["question"].append(question.text)
            inputs["context"].append(contexts[i].text)
            inputs["id"].append(i)
        eval_examples = Dataset.from_dict(inputs)
        column_names = eval_examples.column_names

        eval_dataset = eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
        )

        eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])

        eval_dataloader = DataLoader(
            eval_dataset_for_model,
            collate_fn=default_data_collator,
            batch_size=self.model_args.eval_batch_size,
        )
        self.model.eval()
        all_start_logits = []
        all_end_logits = []
        for batch in eval_dataloader:
            for k in batch:
                batch[k] = batch[k].to(self.device)
            with torch.no_grad():
                outputs = self.model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                all_start_logits.append(start_logits.cpu().numpy())
                all_end_logits.append(end_logits.cpu().numpy())

        start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, self.args["max_answer_length"])
        end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset,  self.args["max_answer_length"])

        del all_start_logits
        del all_end_logits

        outputs_numpy = (start_logits_concat, end_logits_concat)

        all_nbest_json = post_processing_function(eval_examples, eval_dataset, outputs_numpy)

        all_answers = []
        for idx, ans in enumerate(all_nbest_json):
            all_answers.append(Answer(
                text=all_nbest_json[ans][0]["text"],
                score=all_nbest_json[ans][0]["probability"],
                # score=all_nbest_json[ans][0]["start_logit"] + all_nbest_json[ans][0]["end_logit"],
                ctx_score=contexts[idx].score,
                language=question.language
            ))
        return all_answers

