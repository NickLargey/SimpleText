import json

print("sample command for sentence: python3 task3_evaluation.py 0 LlAMA3_8B_2")
print("sample command for abstract: python3 task3_evaluation.py 1 LlAMA3_8B_2")
from evaluate import load
sari = load("sari")
import textstat
import sys

abstract = int(sys.argv[1])
run_id = sys.argv[2]

print(abstract)
ABSTRACT = True
if abstract == 0:
    ABSTRACT = False

if ABSTRACT:
    context = "source_abs"
    result = "simplified_abs"
    Loc = "abs_id"
    file_in = "abs"
else:
    context = "source_snt"
    Loc = "snt_id"
    result = "simplified_snt"
    file_in = "snt"

def read_references(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def get_sorted_list(lst_dic, id_str, id_sent,is_ref=False):
    temp = {}
    for item in lst_dic:
        dic_key = item[id_str]
        dic_value = item[id_sent]
        if is_ref:
            temp[dic_key] = [dic_value]
        else:
            temp[dic_key] = dic_value
    temp = dict(sorted(temp.items(), key=lambda item: item[0]))
    print(len(temp.keys()))
    return list(temp.values())


ref_sen_file = r"train/simpletext_task3_2024_train_"+file_in+"_reference.json"
source_sen_file = r"train/simpletext_task3_2024_train_"+file_in + "_source.json"
prediction_sen_file = r"results/"+run_id+"_"+file_in + ".json"

references = read_references(ref_sen_file)
sources = read_references(source_sen_file)
predictions = read_references(prediction_sen_file)

references = get_sorted_list(references, Loc, result, True)
sources = get_sorted_list(sources, Loc, context)
predictions = get_sorted_list(predictions, Loc, result)


dic_source = {}

# sources=["About 95 species are currently accepted.", "Here we go, let's go automobile that has no driver."]
# predictions=["About 95 you now get in.", "Let's go!"]
# references=[["About 95 species are currently known.","About 95 species are now accepted.","95 species are now accepted."], ["Way to go!"]]
sari_score = sari.compute(sources=sources, predictions=predictions, references=references)
print(sari_score)

count = 0
sum = 0.0
for item in predictions:
    try:
        temp = textstat.flesch_kincaid_grade(item)
        # print(temp)
        sum += temp
        count += 1
    except:
        continue
print(sum/count)
