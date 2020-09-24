## A Simple Question-Answer Example (Chinese)

```python
from bertserini.reader.base import Question, Context
from bertserini.reader.bert_reader import BERT
from bertserini.utils.utils_new import get_best_answer

language = "zh"
model_name = "rsvp-ai/bertserini-bert-base-cmrc"
tokenizer_name = "rsvp-ai/bertserini-bert-base-cmrc"
bert_reader = BERT(model_name, tokenizer_name)

# Here is our question:
question = Question("《战国无双3》是由哪两个公司合作开发的？", language)

# Option 1: fetch some contexts from Wikipedia with Pyserini
from bertserini.retriever.pyserini_retriever import retriever, build_searcher
searcher = build_searcher("indexes/lucene-index.wiki_zh_paragraph_with_title_0.6.0.pos+docvectors")
contexts = retriever(question, searcher, 10)

# Option 2: hard-coded contexts
contexts = [Context('《战国无双3》（）是由光荣和ω-force开发的战国无双系列的正统第三续作。本作以三大故事为主轴，分别是以武田信玄等人为主的《关东三国志》，织田信长等人为主的《战国三杰》，石田三成等人为主的《关原的年轻武者》，丰富游戏内的剧情。此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型等，请至战国无双系列1.由于乡里大辅先生因故去世，不得不寻找其他声优接手。从猛将传 and Z开始。2.战国无双 编年史的原创男女主角亦有专属声优。此模式是任天堂游戏谜之村雨城改编的新增模式。本作中共有20张战场地图（不含村雨城），后来发行的猛将传再新增3张战场地图。但游戏内战役数量繁多，部分地图会有兼用的状况，战役虚实则是以光荣发行的2本「战国无双3 人物真书」内容为主，以下是相关介绍。（注：前方加☆者为猛将传新增关卡及地图。）合并本篇和猛将传的内容，村雨城模式剔除，战国史模式可直接游玩。主打两大模式「战史演武」&「争霸演武」。系列作品外传作品')]

# Either option, we can ten get the answer candidates by reader
# and then select out the best answer based on the linear 
# combination of context score and phase score
candidates = bert_reader.predict(question, contexts)
answer = get_best_answer(candidates, 0.45)
print(answer.text)
```


NOTE:

 The index we used above is Chinese Wikipedia, which can be download via:
```
wget ftp://72.143.107.253/BERTserini/chinese_wiki_2018_index.zip
```

After unzipping these file, we suggest you putting it in `indexes/`.

We have uploaded following finetuned checkpoints to the huggingace models:\
[bertserini-bert-base-cmrc](https://huggingface.co/rsvp-ai/bertserini-bert-base-cmrc)