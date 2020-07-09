import torch
from tqdm import tqdm
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from utils_squad import (para_questions_to_examples, convert_examples_to_features,
                         RawResult, form_answer, RawResultExtended)
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer)
from pytorch_transformers import BertForQuestionAnswering
from args import *
from utils import strip_accents
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())
MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
}
def to_list(tensor):
    return tensor.detach().cpu().tolist()

class BertReader:
    def __init__(self, args):
        super(BertReader, self).__init__()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        self.tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        self.model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        self.model = self.model.to(args.device)
        self.model.eval()
        self.args = args

    def predict(self, id_, question, paragraph_texts, paragraph_scores):
        examples = para_questions_to_examples(id_, question, paragraph_texts, paragraph_scores)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=self.tokenizer,
                                                max_seq_length=self.args.max_seq_length,
                                                doc_stride=self.args.doc_stride,
                                                max_n_answers=1,
                                                max_query_length=self.args.max_query_length,
                                                is_training=False)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)

        # eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        # todo: don't know what's the difference between samplers(yqxie)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        all_results = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': None if self.args.model_type == 'xlm' else batch[2]  # XLM don't use segment_ids
                          }
                example_indices = batch[3]
                if self.args.model_type in ['xlnet', 'xlm']:
                    inputs.update({'cls_index': batch[4],
                                   'p_mask':    batch[5]})
                outputs = self.model(**inputs)
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                if self.args.model_type in ['xlnet', 'xlm']:
                    # XLNet uses a more complex post-processing procedure
                    result = RawResultExtended(unique_id            = unique_id,
                                               start_top_log_probs  = to_list(outputs[0][i]),
                                               start_top_index      = to_list(outputs[1][i]),
                                               end_top_log_probs    = to_list(outputs[2][i]),
                                               end_top_index        = to_list(outputs[3][i]),
                                               cls_logits           = to_list(outputs[4][i]))
                else:
                    result = RawResult(unique_id    = unique_id,
                                       start_logits = to_list(outputs[0][i]),
                                       end_logits   = to_list(outputs[1][i]))
                all_results.append(result)

        answers, nbest_answers = form_answer(examples, features,
                                             all_results, self.args.n_best_size,
                                             self.args.max_answer_length, self.args.do_lower_case, "",
                                             "", "", self.args.verbose_logging,
                                             self.args.version_2_with_negative,
                                             self.args.null_score_diff_threshold,
                                             chinese=self.args.chinese)

        all_answers = []
        for answer_id, ans in enumerate(answers):
            ans_dict = {"id": id_,
                        "answer": answers[ans][0],
                        "phrase_score": answers[ans][1],
                        "sentence": answers[ans][2],
                        "paragraph_score": paragraph_scores[answer_id],
                        }
            all_answers.append(ans_dict)
        return all_answers

if __name__ == "__main__":

    bert_reader = BertReader(args)
    all_results = []

    #question = strip_accents("who is kevin o'leary?") # convert Latin into English
    question = "台灣第一座採用花崗石建造的洋式燈塔於何時建立？"
    #paragraph_texts = ["Terence Thomas Kevin O'Leary (born 9 July 1954) is a Canadian businessman, author and television personality. He co-founded O'Leary Funds and SoftKey. ... In 2017, he campaigned to be the leader of the Conservative Party of Canada."]
    paragraph_texts = ["東犬燈塔創建於西元1872年，東犬燈塔為台灣第一批採用花崗石建造的洋式燈塔，樓高四層，由塔身、塔燈及塔三部分組成，長年以來保存相當完好，其空間構造甚具特色，是研究清代洋式燈塔的重要實例。東犬燈塔，塔座係由花崗石砌造，空心圓筒形平面，座身向上微收，內做螺旋石梯，盤旋直上。座頂疊澀出檐，上設工作平台，外圍鑄鐵欄杆，塔頂安圓拱形鑄鐵燈罩以及圓標。東犬燈塔由地面至風標的總高為六十四英尺，即一九點五公尺。其基地呈長方形，長約一八四公尺，寬約六一．五公尺，面積約一二三一六平方公尺。但民國六十七年國防部因戰備需要，將西南角圍牆拆除約五十公尺，故目前面積約一○五四九平方公尺。"]
    paragraph_scores = [100]
    
    final_answers = bert_reader.predict("000", question, paragraph_texts, paragraph_scores)
    print(question, final_answers)

    all_results.append(final_answers)
