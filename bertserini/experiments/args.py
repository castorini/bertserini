import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset_path",
    default=None,
    type=str,
    required=True,
    help="Path to the [dev, test] dataset",
)

parser.add_argument(
    "--index_path",
    default=None,
    type=str,
    required=True,
    help="Path to the indexes of contexts",
)
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)
parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--output",
    default=None,
    type=str,
    required=True,
    help="The output file where the runs results will be written to",
)
parser.add_argument(
    "--language",
    default="en",
    type=str,
    help="The language of task",
)
parser.add_argument(
    "--topk",
    default=10,
    type=int,
    help="The number of contexts retrieved for a question",
)
args = parser.parse_args()