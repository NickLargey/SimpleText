import requests
import csv
import json
import os

# directory to save the top-100(ish) results per query using elastic search API
the_directory = 'Selective_JSONS'

def download_elastic(target_dir, num_of_results):
    # Reading the topic file
    with open("simpletext_2024_task1_queries.csv", "r") as f:
        reader = csv.reader(f, delimiter=";")
        # skip header
        next(reader)

        user, password = 'inex', 'qatc2011'
        pre_qid = ""
        counter = 1
        for line in reader:
            query_to_es = line[-1]
            url = query_to_es + "&size=" + str(num_of_results)
            q_id = line[0]
            if q_id == pre_qid:
                counter += 1
            else:
                counter = 1
            pre_qid = q_id
            result = requests.get(url, auth=(user, password)).content.decode("utf-8")
            obj = json.loads(result)

            # create the target directory if it doesn't already exist
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # construct the output file path
            output_file = os.path.join(target_dir, f"{q_id}_{counter}.json")

            # dump the JSON to the output file
            with open(output_file, "w") as file:
                json.dump(obj, file)

download_elastic(the_directory, 100)
