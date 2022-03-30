import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--device",
    default="cpu",
    type=str,
    help="Device to run query encoder, cpu or [cuda:0, cuda:1, ...]",
)
parser.add_argument(
    "--dataset_path",
    default=None,
    type=str,
    help="Path to the [dev, test] dataset",
)
parser.add_argument(
    "--retriever",
    default="bm25",
    type=str,
    help="define the indexer type",
)
parser.add_argument(
    "--k1",
    default=0.9,
    type=float,
    help="k1, parameter for bm25 retriever",
)
parser.add_argument(
    "--b",
    default=0.4,
    type=float,
    help="b, parameter for bm25 retriever",
)
parser.add_argument(
    "--encoder",
    default="facebook/dpr-question_encoder-multiset-base",
    type=str,
    help="dpr encoder path or name",
)
parser.add_argument(
    "--query_tokenizer_name",
    default=None,
    type=str,
    help="tokenizer for dpr encoder",
)
parser.add_argument(
    "--index_path",
    default=None,
    type=str,
    help="Path to the indexes of contexts",
)
parser.add_argument(
    "--sparse_index",
    default=None,
    type=str,
    help="Path to the indexes of sarse tokenizer, required when using dense index, in order to retrieve the raw document",
)
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)
parser.add_argument(
    "--tokenizer_name",
    default=None,
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--output",
    default=None,
    type=str,
    help="The output file where the runs results will be written to",
)
parser.add_argument(
    "--output_nbest_file",
    default=None, 
    type=str,
    help="The output file for store nbest results temporarily",
)
parser.add_argument(
    "--language",
    default="en",
    type=str,
    help="The language of task",
)
parser.add_argument(
    "--eval_batch_size",
    default=32,
    type=int,
    help="batch size for evaluation",
)
parser.add_argument(
    "--topk",
    default=10,
    type=int,
    help="The number of contexts retrieved for a question",
)
parser.add_argument(
    "--support_no_answer",
    action="store_true",
    help="support no answer prediction",
)
parser.add_argument(
    "--strip_accents",
    action="store_true",
    help="script accents for questions",
)
args = parser.parse_args()