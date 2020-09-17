import os
import logging
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers.data.processors.squad import SquadResult
from bertserini.run_squad import to_list
from bertserini.utils_squad import compute_predictions_log_probs, compute_predictions_logits, SquadExample
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    squad_convert_examples_to_features,
)


logger = logging.getLogger(__name__)


class MySquadExample(SquadExample):
    def __init__(self,
                 qas_id,
                 question_text,
                 context_text,
                 answer_text,
                 start_position_character,
                 title,
                 answers=[],
                 is_impossible=False,
                 paragraph_score=0,
                 chinese=False,
                 tokenizer=None):
        super(MySquadExample, self).__init__(
            qas_id,
            question_text,
            context_text,
            answer_text,
            start_position_character,
            title,
            answers,
            is_impossible,
            chinese,
            tokenizer,
        )
        self.paragraph_score = paragraph_score


def create_inference_examples(query, paragraphs, paragraph_scores, chinese=False, tokenizer=None):
    examples = []
    for (id, paragraph) in enumerate(paragraphs):
        example = MySquadExample(
            qas_id=id,
            question_text=query,
            context_text=paragraph,
            answer_text=None,
            start_position_character=None,
            title="",
            is_impossible=False,
            answers=[],
            paragraph_score=paragraph_scores[id],
            chinese=chinese,
            tokenizer=tokenizer,
        )
        id += 1
        examples.append(example)

    return examples


class BertReader:
    def __init__(self, args):
        super(BertReader, self).__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        checkpoint = self.args.model_name_or_path

        logger.info("Evaluate the following checkpoints: %s", checkpoint)

        # Reload the model
        global_step = ""
        self.model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
        self.model = self.model.to(args.device)

        self.model.eval()

    def predict(self, id_, question, paragraph_texts, paragraph_scores):
        # dataset, examples, features = load_and_cache_examples(self.args, self.tokenizer, evaluate=True, output_examples=True)

        # processor = SquadV2Processor() if self.args.version_2_with_negative else SquadV1Processor()
        # todo convert to single query examples
        examples = create_inference_examples(
            question,
            paragraph_texts,
            paragraph_scores,
            chinese=self.args.chinese,
            tokenizer=self.tokenizer)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.args.max_seq_length,
            doc_stride=self.args.doc_stride,
            max_query_length=self.args.max_query_length,
            is_training=False,
            return_dataset="pt",
            threads=self.args.threads,
            tqdm_enabled=False
        )

        # if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        #     os.makedirs(args.output_dir)

        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # multi-gpu evaluate
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        # Eval!
        # logger.info("***** Running evaluation {} *****".format(prefix))
        # logger.info("  Num examples = %d", len(dataset))
        # logger.info("  Batch size = %d", args.eval_batch_size)

        all_results = []
        # start_time = timeit.default_timer()

        for batch in eval_dataloader:
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                # if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                #     del inputs["token_type_ids"]

                feature_indices = batch[3]

                # XLNet and XLM use more arguments for their predictions
                # if args.model_type in ["xlnet", "xlm"]:
                #     inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                #     # for lang_id-sensitive xlm models
                #     if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                #         inputs.update(
                #             {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                #         )

                outputs = self.model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]

                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )

                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        # Compute predictions
        prefix = ""
        output_prediction_file = os.path.join(self.args.output_dir, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(self.args.output_dir, "nbest_predictions_{}.json".format(prefix))

        if self.args.version_2_with_negative:
            output_null_log_odds_file = os.path.join(self.args.output_dir, "null_odds_{}.json".format(prefix))
        else:
            output_null_log_odds_file = None

        # XLNet and XLM use a more complex post-processing procedure
        if self.args.model_type in ["xlnet", "xlm"]:
            start_n_top = self.model.config.start_n_top if hasattr(self.model,
                                                                   "config") else self.model.module.config.start_n_top
            end_n_top = self.model.config.end_n_top if hasattr(self.model,
                                                               "config") else self.model.module.config.end_n_top

            answers, nbest_answers = compute_predictions_log_probs(
                examples,
                features,
                all_results,
                self.args.n_best_size,
                self.args.max_answer_length,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                start_n_top,
                end_n_top,
                self.args.version_2_with_negative,
                self.tokenizer,
                self.args.verbose_logging,
                self.args.chinese
            )
        else:
            answers, nbest_answers = compute_predictions_logits(
                examples,
                features,
                all_results,
                self.args.n_best_size,
                self.args.max_answer_length,
                self.args.do_lower_case,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                self.args.verbose_logging,
                self.args.version_2_with_negative,
                self.args.null_score_diff_threshold,
                self.tokenizer,
                self.args.chinese
            )

        all_answers = []
        for answer_id, ans in enumerate(answers):
            ans_dict = {"id": id_,
                        "answer": answers[ans][0],
                        "phrase_score": answers[ans][1],
                        "paragraph_score": paragraph_scores[answer_id],
                        }
            all_answers.append(ans_dict)
        return all_answers
