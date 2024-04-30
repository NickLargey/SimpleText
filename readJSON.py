import os
import json
import sys
import subprocess
from collections import defaultdict

# argument 1 is name of txt
# argument 2 is directory path to folder of jsons

def make_json_file(directory_path, txtname):
    # Get a list of JSON file names in the directory and sort them in alphabetical order
    json_files = sorted([filename for filename in os.listdir(directory_path) if filename.endswith('.json')])

    # Loop through each file in the directory
    topic_doc_scores = []
    for filename in json_files:
        if filename.endswith('.json'): # Make sure the file is a JSON file
            filepath = os.path.join(directory_path, filename)
            topic_id = os.path.splitext(filename)[0]
            topic_id = topic_id.replace('_', '.')
            with open(filepath) as f:
                data = json.load(f)

            # Navigate to the "hits" -> "hits" level of the JSON data
            hits = data['hits']['hits']
            rank = 1
            for hit in hits:
                doc_id = hit['_id']
                score = hit['_score']
                abstract = hit['_source']['abstract']
                if abstract == "":
                    continue
                topic_doc_scores.append(f"{topic_id} 1 {doc_id} {rank} {score} donut-graph\n")
                rank += 1  
                if rank > 100:
                    break

    # normalize the scores in the list
    normalized_topic_doc_scores = normalize(topic_doc_scores)

    # write the normalized scores to the output file
    with open(txtname, 'w') as f:
        f.writelines(normalized_topic_doc_scores)


def normalize(topic_doc_scores):
    topic_scores = defaultdict(list)
    for line in topic_doc_scores:
        topic_id, _, _, _, score, _ = line.split()
        score = float(score)
        topic_scores[topic_id].append(score)

    max_scores = {}
    for topic_id, scores in topic_scores.items():
        max_score = max(scores)
        max_scores[topic_id] = max_score

    normalized_topic_doc_scores = []
    for line in topic_doc_scores:
        topic_id, _, doc_id, rank, score, _ = line.split()
        score = float(score)
        max_score = max_scores[topic_id]
        normalized_score = score / max_score
        normalized_topic_doc_scores.append(f"{topic_id} 1 {doc_id} {rank} {normalized_score} donut-graph\n")

    return normalized_topic_doc_scores



args = sys.argv[1:]

val = True

for arg in args:
    if val:
        txtname = arg
        val = False
    else:
        make_json_file(arg, txtname)
        val = True