# !pip install transformers
# !pip install huggingface
# !pip install accelerate

# !huggingface-cli login

from evaluate import load
from transformers import AutoTokenizer
import nltk
import os
import huggingface
import evaluate
import transformers
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
nltk.download('punkt')

train_df = pd.read_csv("SimpleText_data/simpletext_task3_train.tsv", sep='\t')
ref_df = pd.read_csv("SimpleText_data/simpletext_task3_qrels.tsv", sep='\t')

merged_df = pd.merge(train_df, ref_df, on='snt_id')

# Merge to drop any missing values
df = merged_df[['snt_id', 'source_snt', 'simplified_snt']]

X = df.drop('simplified_snt', axis=1)
y = df['simplified_snt']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


if torch.cuda.is_available():
    torch.set_default_device("cuda")
    print("cuda")
else:
    torch.set_default_device("cpu")
model = "meta-llama/Llama-2-7b-chat-hf"
# model = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=7500,
)


bleu = evaluate.load("bleu")
# blue_results = bleu.compute(predictions=predictions, references=references)

rouge = evaluate.load('rouge')
# rouge_results = rouge.compute(predictions=predictions, references=references)

sari = evaluate.load("sari")
# sources = ["About 95 species are currently accepted ."]
# predictions = ["About 95 you now get in ."]
# references = [["About 95 species are currently known .",
#                "About 95 species are now accepted .", "95 species are now accepted ."]]
# sari_score = sari.compute(
#     sources=sources, predictions=predictions, references=references)

system_message = "You are Yan Lecun"
user_message = "You want to insert n items in a queue, and then remove them one by one. However, the only data structure that you can use is Stack. How would you do it? Just explain your approach."

prompt = "<s>[INST]<<SYS>>" + system_message + \
    "<</SYS>>\n" + user_message + "[/INST]</s>"
sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)
# print(len(sequences))
for seq in sequences:
    print(seq)
    result = seq['generated_text']
