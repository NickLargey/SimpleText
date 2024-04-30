import json
import textstat
import requests
import sys
import os
from tqdm import tqdm


############# SMOG
# The SMOG Readability Formula (Simple Measure of Gobbledygook) is a popular method to use on health literacy materials.

############# FLESCH
# The U.S. Department of Defense uses the Reading Ease test as the standard test of readability for
# its documents and forms. Florida requires that life insurance policies have
# a Flesch Reading Ease score of 45 or greater.

############# Coleman Liau
# The Coleman-Liau Formula usually gives a lower grade value than any of the Kincaid,
# ARI and Flesch values when applied to technical documents.

def read_trec_file(trec_file_path):
    result_dict = {}
    with open(trec_file_path, 'r') as f:
        for line in f.readlines():
            topic_id, _, doc_id, rank, score, _ = line.strip().split()
            if topic_id not in result_dict:
                result_dict[topic_id] = []
            result_dict[topic_id].append(doc_id)
    return result_dict


def json_abstracts(doc_ids, topic):
    json_path = ""
    topic = topic.replace('.', '_')
    # find the JSON file that contains the topic string in its filename
    for filename in os.listdir("Selective_JSONS"):
        if topic in filename:
            json_path = os.path.join("Selective_JSONS/", filename)
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
def fast_readability(doc_ids, topic):
    dic_readability_scores = {}
    abstracts_dic = json_abstracts(doc_ids, topic)
    for doc_id in doc_ids:
        abstract = abstracts_dic[doc_id]
        if abstract == None:
            continue
        dic_readability_scores[doc_id] = {
            "flesch": textstat.flesch_reading_ease(abstract),
            "smog": textstat.smog_index(abstract),
            "Coleman Liau": textstat.coleman_liau_index(abstract),
        }
    return dic_readability_scores

args = sys.argv[1:]

filename = args[0]
all_doc_ids = set()
file_dict = read_trec_file(filename)
all_scores = {}
for topic in file_dict:
    doc_ids = [doc_id for doc_id in file_dict[topic]]
    readability_scores = fast_readability(doc_ids, topic)
    for doc_id in readability_scores:
        all_scores[doc_id] = readability_scores[doc_id]

flesch_scores = []
smog_scores = []
coleman_scores = []

for doc_id, scores in all_scores.items():
    flesch_scores.append(scores["flesch"])
    smog_scores.append(scores["smog"])
    coleman_scores.append(scores["Coleman Liau"])

flesch_avg = sum(flesch_scores) / len (flesch_scores)
smog_avg = sum(smog_scores) / len(smog_scores)
coleman_avg = sum(coleman_scores) / len(coleman_scores)

print(f"flesch average: {flesch_avg}")
print(f"smog average: {smog_avg}")
print(f"Coleman Liau average: {coleman_avg}")
