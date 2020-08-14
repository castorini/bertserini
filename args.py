import argparse
import pprint
import torch
# from transformers import (WEIGHTS_NAME, BertConfig,
#                                   BertTokenizer,
#                                   XLMConfig, XLMForQuestionAnswering,
#                                   XLMTokenizer, XLNetConfig,
#                                   XLNetForQuestionAnswering,
#                                   XLNetTokenizer)
# from transformers import BertForQuestionAnswering

# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
#                   for conf in (BertConfig, XLNetConfig, XLMConfig)), ())
# MODEL_CLASSES = {
#     'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
#     'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
#     'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
# }

parser = argparse.ArgumentParser()

## Other parameters

parser.add_argument("--eval_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--no_cuda", action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--chinese', action='store_true', help="Chinese")
parser.add_argument('--toSimplified', action='store_true', help="to simplified chinese")
parser.add_argument('--k1', type=float, default=0.9,
                    help='bm25 parameter')
parser.add_argument('--b', type=float, default=0.4,
                    help='bm25 parameter')
parser.add_argument('--rm3', action="store_true",
                    help='wether use rm3 ranker')
parser.add_argument('--index_path', type=str,
                    help='Path to index file')
parser.add_argument('--para_num', type=int,
                    help='number of top k paragraphs into bert')
parser.add_argument('--linking', action="store_true", default=False)
parser.add_argument('--link_doc_score', type=float, default=30)
parser.add_argument('--output_fn', type=str, default="output.json",
                    help='output file name')

parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    required=True,
    #help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
)
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)
parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help="The output directory where the model checkpoints and predictions will be written.",
)

# Other parameters
parser.add_argument(
    "--data_dir",
    default=None,
    type=str,
    help="The input data dir. Should contain the .json files for the task."
         + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
)
parser.add_argument(
    "--train_file",
    default=None,
    type=str,
    help="The input training file. If a data dir is specified, will look for the file there"
         + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
)
parser.add_argument(
    "--predict_file",
    default=None,
    type=str,
    help="The input evaluation file. If a data dir is specified, will look for the file there"
         + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
)
parser.add_argument(
    "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
)
parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)

parser.add_argument(
    "--version_2_with_negative",
    action="store_true",
    help="If true, the SQuAD examples contain some that do not have an answer.",
)
parser.add_argument(
    "--null_score_diff_threshold",
    type=float,
    default=0.0,
    help="If null_score - best_non_null is greater than the threshold predict null.",
)

parser.add_argument(
    "--max_seq_length",
    default=384,
    type=int,
    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
         "longer than this will be truncated, and sequences shorter than this will be padded.",
)
parser.add_argument(
    "--doc_stride",
    default=128,
    type=int,
    help="When splitting up a long document into chunks, how much stride to take between chunks.",
)
parser.add_argument(
    "--max_query_length",
    default=64,
    type=int,
    help="The maximum number of tokens for the question. Questions longer than this will "
         "be truncated to this length.",
)
parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
parser.add_argument(
    "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
)
parser.add_argument(
    "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
)

parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument(
    "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
)
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument(
    "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
)
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument(
    "--n_best_size",
    default=20,
    type=int,
    help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
)
parser.add_argument(
    "--max_answer_length",
    default=30,
    type=int,
    help="The maximum length of an answer that can be generated. This is needed because the start "
         "and end predictions are not conditioned on one another.",
)
parser.add_argument(
    "--verbose_logging",
    action="store_true",
    help="If true, all of the warnings related to data processing will be printed. "
         "A number of warnings are expected for a normal SQuAD evaluation.",
)
parser.add_argument(
    "--lang_id",
    default=0,
    type=int,
    help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
)

parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
)
parser.add_argument(
    "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
)
parser.add_argument(
    "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
)
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
         "See details at https://nvidia.github.io/apex/amp.html",
)
parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

args = parser.parse_args()

pprint.pprint(vars(args))

if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = 1
args.device = device

