import json
from Preprocessing_tools import *
from sentence_transformers import CrossEncoder
import torch
import sys
from tqdm import tqdm
import signal
from transformers import AutoTokenizer

def handle_sigint(signum, frame):
    # do nothing
    pass

signal.signal(signal.SIGINT, handle_sigint)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# get abstracts from jsons
def json_abstracts(doc_ids, topic):
    json_path = ""
    # find the JSON file that contains the topic string in its filename
    for filename in os.listdir("Top-2000_InitialQuery/"):
        if topic in filename:
            json_path = os.path.join("Top-2000_InitialQuery/", filename)
            break
    else:
        raise ValueError(f"No JSON file found for topic '{topic}'")
    with open(json_path) as f:
        data = json.load(f)
        abstracts = {}
        for doc_id in doc_ids:
            for hit in data['hits']['hits']:
                if hit['_id'] == doc_id:
                    abstracts[doc_id] = hit['_source']['abstract']
                    break
            else:
                # if doc_id is not found in the JSON file, set abstract to None
                abstracts[doc_id] = None
        return abstracts

# rerank results with cross encoder
def cross_encoder_ranking(topic_dic, trec_file_path, model, tokenizer):
    initial_retrieval = read_trec_file(trec_file_path)
    
    cross_encoder_ranked_docs = []
    max_scores = {}

    for topic in tqdm(topic_dic, desc="Processing topics"):
        query, topic_text = topic_dic[topic]
        corrected_topic = topic.replace("_", ".")
        if corrected_topic not in initial_retrieval:
            continue
        initial_ret_topic = initial_retrieval[corrected_topic]
        doc_ids = [doc_id for doc_id in initial_ret_topic]
        abstract_dict = json_abstracts(doc_ids, topic)

        corpus = []
        index_to_question_id = {}
        idx = 0

        for doc_id in initial_ret_topic:
            abstract = abstract_dict[doc_id]
            corpus.append(abstract)
            index_to_question_id[idx] = doc_id
            idx += 1

        ce_scores = []
        for abstract in corpus:
            score = model.predict([[query + " " + topic_text, abstract]])
            ce_scores.append(score[0])

        k_value = min(100, len(ce_scores))
        top_results = torch.topk(torch.Tensor(ce_scores), k=k_value)

        ranked_docs = [(index_to_question_id[int(idx)], score.item()) for score, idx in zip(top_results[0], top_results[1])]
        cross_encoder_ranked_docs.append((topic.replace("_", "."), ranked_docs))
        max_scores[topic.replace("_", ".")] = max(score for doc_id, score in ranked_docs)

    # Normalize scores for each topic by the corresponding max score using the sigmoid function
    for i in range(len(cross_encoder_ranked_docs)):
        topic_id, ranked_docs = cross_encoder_ranked_docs[i]
        max_score = max_scores[topic_id]
        ranked_docs_norm = []
        for doc_id, score in ranked_docs:
            score_tensor = torch.tensor(score)
            max_score_tensor = torch.tensor(max_score)
            score_norm = sigmoid(score_tensor) / sigmoid(max_score_tensor)
            ranked_docs_norm.append((doc_id, score_norm.item()))
        cross_encoder_ranked_docs[i] = (topic_id, ranked_docs_norm)

    return cross_encoder_ranked_docs

# transform trec file to dictionary
def read_trec_file(trec_file_path):
    result_dict = {}
    with open(trec_file_path, 'r') as f:
        for line in f.readlines():
            topic_id, _, doc_id, rank, score, _ = line.strip().split()
            if topic_id not in result_dict:
                result_dict[topic_id] = []
            result_dict[topic_id].append(doc_id)
    return result_dict

# write the results of cross encoder in trec format
def write_rankings(rankings, file_path):
    with open(file_path, "w") as f:
        for query_id, docs in rankings:
            for rank, (doc_id, score) in enumerate(docs, start=1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score} donut_graph\n")

def main():
    args = sys.argv[1:]
    # file that holds elastic search results
    read_file = args[0]
    # new file to print results to
    results_file = args[1]
        
    print("Reading json")
    initial_retrieval = read_all_jsons(target_dir="Top-2000_InitialQuery/")
    topic_dic = read_topic_file("simpletext_2024_task1_queries.csv")

    model = CrossEncoder('cross-encoder/ms-marco-electra-base')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-electra-base')

    print("CrossEncoder ranking")
    cross_encoder_ranked_docs = cross_encoder_ranking(topic_dic, read_file, model, tokenizer)

    print("Writing CrossEncoder ranked file")
    write_rankings(cross_encoder_ranked_docs, results_file)

if __name__ == "__main__":
    main()
