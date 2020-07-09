import unicodedata
import string
import logging
import json

def strip_accents(text):
    return "".join(char for char in unicodedata.normalize('NFKD', text)
                   if unicodedata.category(char) != 'Mn')

def convert_squad_to_list(squad_filename):
    data = json.load(open(squad_filename, 'r'))
    data = data["data"]
    converted_data = []
    for article in data:
        for paragraph in article["paragraphs"]:
            text = paragraph["context"]
            for qa in paragraph["qas"]:
                id_ = qa["id"]
                question = qa["question"]
                answers = qa["answers"]
                converted_data.append({"id": id_, "question": question, "answers": answers, "context": text})
    return converted_data

def init_logger(bot):
    # create logger with 'spam_application'
    # bot = 'server_cn' if args.chinese else 'server_en'
    logger = logging.getLogger('{} log'.format(bot))
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('{}.log'.format(bot))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def split_title(paragraph):
    sents = paragraph.split(".")
    text =  ".".join(sents[1:]).strip()
    title = sents[0].strip()
    return title, text

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_text(s):
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return remove_punc(lower(s))

def normalize_chinese_text(s):
    def remove_punc(text):
        exclude = set(zhon.hanzi.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return remove_punc(s)
