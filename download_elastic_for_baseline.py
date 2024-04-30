import requests
import csv
import json
import os

target_dir = "Top-2000_InitialQuery/"
with open("simpletext_2024_task1_queries.csv", "r") as f:
    reader = csv.reader(f, delimiter=";")
    next(reader)
    user, password = 'inex', 'qatc2011'
    pre_qid = ""
    counter = 1
    for line in reader:
        query_to_es = line[-1]
        url = query_to_es+"&size=2000"
        q_id = line[0]
        if q_id == pre_qid:
            counter += 1
        else:
            counter = 1
        pre_qid = q_id
        result = requests.get(url, auth=(user, password)).content.decode("utf-8")
        obj = json.loads(result)

        # remove quote
        hits_count = len(obj['hits']['hits'])
        if hits_count<2000:
            if query_to_es.endswith("\""):
                original_query = query_to_es.split("q=")[1][1:-1]
                remaining = 2000-hits_count
                remade_query = query_to_es.split("q=")[0] + "q=" + original_query + "&size="+str(remaining)
                result = requests.get(remade_query, auth=(user, password)).content.decode("utf-8")
                obj2 = json.loads(result)
                for item in obj2['hits']['hits']:
                    if item not in obj['hits']['hits']:
                        obj['hits']['hits'].append(item)

        # replace with topic_text
        hits_count = len(obj['hits']['hits'])
        if hits_count < 2000:
                original_query = query_to_es.split("q=")[1][1:-1]
                remaining = 2000 - hits_count
                remade_query = query_to_es.split("q=")[0] + "q=" + line[1] + "&size=" + str(remaining)
                result = requests.get(remade_query, auth=(user, password)).content.decode("utf-8")
                obj2 = json.loads(result)
                for item in obj2['hits']['hits']:
                    if item not in obj['hits']['hits']:
                        obj['hits']['hits'].append(item)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # construct the output file path
        output_file = os.path.join(target_dir, f"{q_id}_{counter}.json")

        # dump the JSON to the output file
        with open(output_file, "w") as file:
            json.dump(obj, file)
