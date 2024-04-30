from evaluate import load, evaluator
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import huggingface
import evaluate
import transformers
import torch
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import login
import os
from dotenv import load_dotenv
import pprint

pp = pprint.PrettyPrinter(indent=4)

load_dotenv()

token = os.getenv('HF_TOKEN')
login(token=token)

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    print("cuda")
else:
    torch.set_default_device("cpu")

###################### LOAD DATA ######################
train_df = pd.read_csv("SimpleText_data/simpletext_task3_train.tsv", sep='\t')
ref_df = pd.read_csv("SimpleText_data/simpletext_task3_qrels.tsv", sep='\t')

merged_df = pd.merge(train_df, ref_df, on='snt_id')

# Merge to drop any missing values
df = merged_df[['snt_id', 'source_snt', 'simplified_snt']]

X = df.drop('snt_id', axis=1)
# y = df['simplified_snt']

X_train, X_val = train_test_split(X, test_size=0.2)
X_list = X_train['source_snt'].tolist()
references = X_train['simplified_snt'].tolist()


###################### LOAD EVALUATORS ######################
# bleu = evaluate.load("bleu")
sari = load("sari")
rouge = evaluate.load('rouge')

###################### LOAD MODEL ######################
Orca = "microsoft/Orca-2-7b"
model = AutoModelForCausalLM.from_pretrained(Orca, device_map='auto',torch_dtype=torch.float16)
Orca_tokenizer = AutoTokenizer.from_pretrained(Orca, use_fast=False, padding=True, truncation=True, padding_side='left')
task_evaluator = evaluator("summarization")
Orca_pipeline = transformers.pipeline(
    "summarization",
    model=model,
    tokenizer=Orca_tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=5000,
)

###################### DEFINE PROMPT ENGINEERING VARIABLES ######################
system_message = "You are teaching a science class for middle and high school students. You have been given exerpts from scientific writings, but need to simplify them for your students. Please rewrite the text in order to maximize understanding while maintaining the original meaning.Your results should maximize desired scores on FKGL, SARI, ROUGE and BLEU metrics."
user_message = X_list
predictions = []

for i in tqdm(range(len(user_message))):
  prompt =  f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message[i]}<|im_end|>\n<|im_start|>assistant"
  sequences = Orca_pipeline(
      prompt,
      do_sample=True,
      top_k=1,
      num_return_sequences=1,
      eos_token_id=Orca_tokenizer.eos_token_id
  )

  inputs = Orca_tokenizer(prompt, return_tensors='pt')
  output_ids = model.generate(inputs["input_ids"],)
  answer = Orca_tokenizer.batch_decode(output_ids)[0]
  predictions.append(answer)

cleaned_predictions = [pred.split('assistant')[1].strip().replace('</s>','') for pred in predictions]

Orca_df = pd.DataFrame()
Orca_df["predictions"] = cleaned_predictions
Orca_df['source_snt'] = user_message
Orca_df['references'] = references

# Orca_df = pd.read_csv("Orca_predictions.tsv", sep='\t')

cleaned_predictions = Orca_df['predictions'].tolist()
user_message = Orca_df['source_snt'].tolist()
references = Orca_df['references'].tolist()

pp.pprint([cleaned_predictions[0]])
pp.pprint([references[0]])

res = []

for i in tqdm(range(len(cleaned_predictions))):
    rouge_results = rouge.compute(predictions=[cleaned_predictions[i]], references=[references[i]])
    res.append(rouge_results)
    
# pp.pprint(res)

Orca_df['ROUGE'] = res
Orca_df.to_csv("Orca_predictions.tsv",sep='\t', index=False)

# outputs = pd.DataFrame()
# outputs["predictions"] = predictions
# outputs['source_snt'] = user_message
# outputs['simplified_snt'] = references[:5]
# print(outputs)

####################### EVALUATE ######################


# blue_results = bleu.compute(predictions=predictions, references=references)

# sources = outputs['source_snt'].tolist()
# predictions = outputs['predictions'].tolist()
# references = outputs['simplified_snt'].tolist()
# sari_results = sari.compute(sources=sources, prediction=predictions, references=references)
# print(sari_results)


