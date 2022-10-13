from typing import List

import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, EvalPrediction
import datasets
from datasets import Dataset
import numpy as np
from typing import List, Optional, Tuple

from bertserini.reader.base import Reader, Question, Context, Answer

from datasets.utils import logging

__all__ = ['T5']
class T5(Reader):
    def __init__(self, args):
        self.model_args = args
        if self.model_args.tokenizer_name is None:
            self.model_args.tokenizer_name = self.model_args.model_name_or_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_args.model_name_or_path).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.tokenizer_name, do_lower_case=True)
        self.question_column = 'question'
        self.context_column = 'context'
        self.answer_column = 'answers'
        '''
          --per_device_train_batch_size 4 \
          --per_device_eval_batch_size 1 \
          --output_dir ./models/s2s_squad2_0train/ \
          --eval_accumulation_steps 1 \
          --predict_with_generate \
        '''
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
            "ignore_pad_token_for_loss": True
        }

    def update_args(self, args_to_change):
        for key in args_to_change:
            self.args[key] = args_to_change[key]

    def predict(self, question: Question, contexts: List[Context]) -> List[Answer]:
        logging.disable_progress_bar()

        def preprocess_squad_batch(
                examples,
                question_column: str,
                context_column: str,
                answer_column: str,
        ) -> Tuple[List[str], List[str]]:
            questions = examples[question_column]
            contexts = examples[context_column]
            answers = examples.get(answer_column,[])

            def generate_input(_question, _context):
                return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])

            inputs = [generate_input(question, context) for question, context in zip(questions, contexts)]
            targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
            return inputs, targets

        def preprocess_function(examples):
            inputs, targets = preprocess_squad_batch(examples, self.question_column, self.context_column, self.answer_column)

            model_inputs = self.tokenizer(inputs, max_length=self.args["max_seq_length"], padding='max_length', truncation=True)
            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=self.args['max_answer_length'], padding='max_length', truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if self.args['ignore_pad_token_for_loss']:
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Validation preprocessing
        def preprocess_validation_function(examples):
            inputs, targets = preprocess_squad_batch(examples, self.question_column, self.context_column, self.answer_column)

            model_inputs = self.tokenizer(
                inputs,
                max_length=self.args["max_seq_length"],
                padding='max_length',
                truncation=True,
                return_offsets_mapping=True,
            )

            if targets:
                # Setup the tokenizer for targets
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(targets, max_length=self.args['max_answer_length'], padding='max_length', truncation=True)

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            # sample_mapping = model_inputs.pop("overflow_to_sample_mapping")
            sample_mapping = list(range(len(model_inputs["input_ids"])))

            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id and we will store the offset mappings.
            model_inputs["example_id"] = []

            for i in range(len(model_inputs["input_ids"])):
                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                model_inputs["example_id"].append(examples["id"][sample_index])

            if targets:
                # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
                # padding in the loss.
                if self.args['ignore_pad_token_for_loss']:
                    labels["input_ids"] = [
                        [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                    ]

                model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        def post_processing_function(examples: datasets.Dataset, features: datasets.Dataset, outputs, stage="eval"):
            # Decode the predicted tokens.
            decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Build a map example to its corresponding features.
            example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
            feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
            predictions = {}
            # Let's loop over all the examples!
            for example_index, example in enumerate(examples):
                # This is the index of the feature associated to the current example.
                feature_index = feature_per_example[example_index]
                predictions[example["id"]] = decoded_preds[feature_index]

            # Format the result to the format the metric expects.
            if self.args['version_2_with_negative']:
                formatted_predictions = [
                    {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
                ]
            else:
                formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

            # references = [{"id": ex["id"], "answers": ex[self.answer_column]} for ex in examples]
            # return EvalPrediction(predictions=formatted_predictions, label_ids=references)
            return formatted_predictions



        inputs = {"question": [], "context": [], "id": []}
        for i, ctx in enumerate(contexts):
            inputs["question"].append(question.text)
            inputs["context"].append(contexts[i].text)
            inputs["id"].append(i)
        print(inputs)
        eval_examples = Dataset.from_dict(inputs)
        column_names = eval_examples.column_names
        eval_dataset = eval_examples.map(
            preprocess_validation_function,
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
        raw_predict = []
        for batch in eval_dataloader:
            for k in batch:
                batch[k] = batch[k].to(self.device)
            outs = self.model.generate(input_ids=batch['input_ids'],
                                  attention_mask=batch['attention_mask'],
                                  max_length=16,
                                  early_stopping=True)
            raw_predict.extend(outs)
        all_nbest_json = post_processing_function(eval_examples, eval_dataset, raw_predict)

        all_answers = []
        for item in all_nbest_json:
            all_answers.append(Answer(
                text=item["prediction_text"],
                score=0.0,
                # score=all_nbest_json[ans][0]["start_logit"] + all_nbest_json[ans][0]["end_logit"],
                ctx_score=contexts[item['id']].score,
                language=question.language
            ))
        return all_answers

