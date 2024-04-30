import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_metric
import evaluate
from evaluate import evaluator
from huggingface_hub import login
from dotenv import load_dotenv
import os
import re
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, model_id):
        self.df = df
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        src = data['source_snt']
        tgt = data['simplified_snt']

        messages = [
            {"role": "system", "content": """Simplify this text for english speaking science students in college. 
                                            Maximize the use of simple words and short sentences but include key words from the original text. 
                                            Optimize the output ROUGE, SARI and BLEU scores.
                                            Don't give an explaination, just output the simplified text."""},
            {"role": "user", "content": f'{src}'},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        pad_token_id = self.tokenizer.eos_token_id
        outputs = self.pipeline(
            prompt,
            max_new_tokens=4096,
            pad_token_id=pad_token_id,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        result = outputs[0]["generated_text"][len(prompt):].split('\n\n\\')[-1].strip()
        cleaned_res = re.sub(r'Here is the simplified text:\n\n', '', result)
        return {
            'snt_id': data['snt_id'],
            'simplified_snt': cleaned_res,
            'source_snt': src,
            'target_snt': tgt
        }

load_dotenv()
token = os.getenv('HF_TOKEN')
login(token=token)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"


# Load the data
X = pd.read_json('task 3/train/simpletext_task3_2024_train_snt_source.json')
y = pd.read_json('task 3/train/simpletext_task3_2024_train_snt_reference.json')

merge_df = pd.merge(X, y, on='snt_id')

drop_df = merge_df.drop_duplicates(subset='snt_id')
df = drop_df.drop(["query_id", "query_text", "doc_id"], axis=1)

# Create a dataset instance
dataset = TextDataset(df, transformers.AutoTokenizer.from_pretrained(model_id), model_id)
sari = evaluate.load('sari')
rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')

# Create a DataLoader instance
batch_size = 1
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Initialize an empty list to store the results
results = []

# Loop through the data in batches
for batch in tqdm(data_loader):
    # Get the batch data
    snt_id = batch['snt_id']
    simplified_snt = batch['simplified_snt']
    source_snt = batch['source_snt']
    target_snt = batch['target_snt']

    bleu_results = bleu.compute(predictions=simplified_snt, references=target_snt) 
    sari_results = sari.compute(sources=source_snt, predictions=simplified_snt, references=[target_snt]) 
    rouge_results = rouge.compute(predictions=simplified_snt, references=target_snt) 

    results.append({
        'snt_id': snt_id,
        'simplified_snt': simplified_snt,
        'source_snt': source_snt,
        'target_snt': target_snt,
        'BLEU': bleu_results,
        'ROUGE': rouge_results,
        'SARI': sari_results['sari']
    })

# Create a DataFrame from the results list
output_df = pd.DataFrame(results)
print("Average SARI score: ", output_df['SARI'].mean())
comp_sum = 0

for i,r in output_df.iterrows():
    comp_sum += r['BLEU']["length_ratio"]

avg_comp = comp_sum/len(output_df['BLEU'])
print("Average Length Ratio score: ", avg_comp)
# Write the DataFrame to a JSON file
output_df.to_json('simpletext_output.json', orient='records')