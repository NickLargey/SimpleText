import math
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import torch
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator, CEBinaryClassificationEvaluator, CERerankingEvaluator
from torch.utils.data import DataLoader
from transformers import *

def test():
    from sentence_transformers.cross_encoder import CrossEncoder
    import numpy as np
    #Pre-trained cross encoder
    model = CrossEncoder('cross-encoder/stsb-distilroberta-base')
    # We want to compute the similarity between the query sentence
    query = 'A man is eating pasta.'
    # With all sentences in the corpus
    corpus = ['A man is eating food.',
        'A man is eating a piece of bread.',
        'The girl is carrying a baby.',
        'A man is riding a horse.',
        'A woman is playing violin.',
        'Two men pushed carts through the woods.',
        'A man is riding a white horse on an enclosed ground.',
        'A monkey is playing drums.',
        'A cheetah is running behind its prey.'
        ]
    # So we create the respective sentence combinations
    sentence_combinations = [[query, corpus_sentence] for corpus_sentence in corpus]
    # Compute the similarity scores for these combinations
    similarity_scores = model.predict(sentence_combinations)
    # Sort the scores in decreasing order
    sim_scores_argsort = reversed(np.argsort(similarity_scores))
    # Print the scores
    print("Query:", query)
    for idx in sim_scores_argsort:
        print("{:.2f}\t{}".format(similarity_scores[idx], corpus[idx]))
    test()

from Task_1.Preprocessing_tools import *
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', device='cuda')
model = SentenceTransformer('allenai/scibert_scivocab_uncased', device='cuda')

import filelock
from sentence_transformers import InputExample
import evaluation

model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
model_name = 'cross-encoder/stsb-roberta-base'
model_name = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
model_name = 'cross-encoder/ms-marco-electra-base'
model = CrossEncoder(model_name, device='cuda', max_length=512)
scores = model.predict([('Query', 'Paragraph1'), ('Query', 'Paragraph2') , ('Query', 'Paragraph3')])
tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096', device='cuda')
model = SentenceTransformer('allenai/longformer-base-4096', device='cuda')

tokens = ["[QSP]", "[TAT]"]
model.tokenizer.add_tokens(tokens, special_tokens=True)
model.model.resize_token_embeddings(len(model.tokenizer))

initial_retrieval = read_all_jsons(target_dir="/home/behrooz.mansouri/simpleText/Top-2000_InitialQuery/")
topic_dic = read_topic_file("SP12023topics.csv")
sample_dic = evaluation("simpletext_2024_task1_train.qrels", initial_retrieval)
counter = 1
train_samples = []
dev_samples = {}

for qid in sample_dic:
    list_current_sample = sample_dic[qid]
    original_query, topic_text = topic_dic[qid]
    temp_list = []

    if counter <= 24:
        for item in list_current_sample:
            label = item[2]
            if label >= 1:
                label = 1
            train_samples.append(InputExample(texts=[original_query+"[QSP]"+topic_text, item[0]+"[TAT]"+item[1]], label=label))
    else:
        for item in list_current_sample:
            label = item[2]
            if qid not in dev_samples:
                dev_samples[qid] = {'query': original_query+"[QSP]"+topic_text, 'positive': set(), 'negative': set()}
            if label == 0:
                label = 'negative'
            else:
                label = 'positive'
            dev_samples[qid][label].add(item[0]+"[TAT]"+item[1])
            dev_samples[qid].extend(temp_list)
    counter += 1

num_epochs = 100
model_save_path = "./ft_electra"
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=4)

#During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
evaluator = CERerankingEvaluator(dev_samples, name='train-eval')
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          save_best_model=True)

model.save(model_save_path+'-latest')
with open('cross_bert_tiny.tsv', 'w', newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
    for topic in topic_dic:
        if topic.startswith("G"):
            #     continue
            num1 = int(topic.split("T")[1].split("_")[0])
        if num1 < 13:
            #     continue
            query, topic_text = topic_dic[topic]
            query_embedding = model.encode(query+" "+topic_text, convert_to_tensor=True)
            initial_ret_topic = initial_retrieval[topic]
            # list of text to be indexed (encoded)
            corpus = []
            # this dictionary is used as key: corpus index [0, 1, 2, ...] and value: corresponding question id
            index_to_question_id = {}
            idx = 0
        for doc_id in initial_ret_topic:
            title, abstract = initial_ret_topic[doc_id]
            scores = model.predict([query + " " + topic_text, title + " " + abstract])
            # Sort the scores in decreasing order
            index_to_question_id[doc_id] = scores
            index_to_question_id = {k: v for k, v in sorted(index_to_question_id.items(), key=lambda item: item[1], reverse=True)}
            print(index_to_question_id)
            counter = 1
            for doc_id in index_to_question_id:
                writer.writerow([topic.replace("_", "."), "0", str(doc_id), str(counter), str(index_to_question_id[doc_id]), "msMarcoCross"])
                counter += 1
                if counter>100:
                    break