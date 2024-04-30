import csv
import json
import os
import markdown

def read_json(filename: str):
    # reads one json file
    with open(filename) as f_in:
        return json.load(f_in)

def read_all_jsons(target_dir):
    # takes in teh directory of topics and return dictionary of topics with id corresponding to qrel files
    dict_top_res = {}
    for file in os.listdir(target_dir):
        temp_dic = read_json(target_dir+file)
        hits = temp_dic['hits']['hits']
        temp_dic_result = {}
        for hit in hits:
            source = hit['_source']
            paper_id = source['id']
            title = source['title']
            abstract = source['abstract']
            temp_dic_result[paper_id] = (title, abstract)

        query_id = file.split(".")[0]
        # if len(temp_dic_result) < 2000:
        #     print(query_id + "\t" + str(len(temp_dic_result)))
        dict_top_res[query_id] = temp_dic_result
    return dict_top_res


def read_topic_file(topic_filepath):
    # a method used to read the topic file for this year of the lab; to be passed to BERT/PyTerrier methods
    result = {}
    with open(topic_filepath, "r") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        pre_qid = ""
        counter = 1
        for line in reader:
            query_to_es = line[-1]
            original_query = query_to_es.split("q=")[1][1:-1]
            topic_text = line[1]
            q_id = line[0]
            if q_id == pre_qid:
                counter += 1
            else:
                counter = 1
            pre_qid = q_id
            result[q_id + "_" + str(counter)] = (original_query, topic_text)
    return result
