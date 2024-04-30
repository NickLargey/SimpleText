import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, Features, Value
from torch.utils.data import DataLoader
import pandas as pd
import torch
import evaluate
from evaluate import load, evaluator
from huggingface_hub import login
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()
token = os.getenv('HF_TOKEN')
login(token=token)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

X = pd.read_json('task 3/train/simpletext_task3_2024_train_snt_source.json')
y = pd.read_json('task 3/train/simpletext_task3_2024_train_snt_reference.json')

merge_df = pd.merge(X, y, on='snt_id')

print('Before: ', merge_df.shape)

drop_df = merge_df.drop_duplicates(subset='snt_id')
df = drop_df.drop(["query_id", "query_text", "doc_id"], axis=1)

print('After: ', df.info())

# features = Features({'snt_id': Value('string'),'source_snt': Value('string'),'simplified_snt': Value('string') })

# train_ds = Dataset.from_pandas(df.head(10), features=features, preserve_index=False).with_format("torch")
# train_dl = DataLoader(train_ds, batch_size=8, shuffle=False)

id = []
simplified_snts = []
src_snt = []
tgt_snt = []

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

for idx, data in df.iterrows():
  # for idx in batch:
    src = data['source_snt']
    tgt = data['simplified_snt']
    
    messages = [
        {"role": "system", "content": """Simplify this text for english speaking students in grades 9 and 10. 
                                        Maximize the use of simple words and short sentences. 
                                        Optimize the output to achieve top ROUGE, SARI and BLEU scores.
                                        """},
        {"role": "user", "content": f'{src}'},
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
              messages, 
              tokenize=False, 
              add_generation_prompt=False
      )

    terminators = [
          pipeline.tokenizer.eos_token_id,
          pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
      ]
    pad_token_id = tokenizer.eos_token_id
    outputs = pipeline(
          prompt,
          max_new_tokens=4096,
          eos_token_id=terminators,
          pad_token_id=pad_token_id,
          do_sample=True,
          temperature=0.6,
          top_p=0.9,
      )
    
    result = outputs[0]["generated_text"][len(prompt):]

    id.append(data['snt_id'])
    simplified_snts.append(result)
    src_snt.append(src)
    tgt_snt.append(tgt)


sari = load("sari")
rouge = load('rouge')
bleu = load("bleu")

output_df = pd.DataFrame()

output_df['snt_id'] = id
output_df['simplified_snt'] = simplified_snts
output_df['source_snt'] = src_snt
output_df['target_snt'] = tgt_snt

sources = output_df['source_snt'].tolist()
predictions = output_df['simplified_snt'].tolist()
references = output_df['target_snt'].tolist()

bleu_results = bleu.compute(predictions=predictions, references=references)
rouge_results = rouge.compute(predictions=predictions, references=references)
sari_results = sari.compute(sources=src_snt, predictions=simplified_snts, references=[tgt_snt])

output_df['BLEU'] = bleu_results
output_df['ROUGE'] = rouge_results
output_df['SARI'] = sari_results


output_df.to_json('simpletext_output.json')





