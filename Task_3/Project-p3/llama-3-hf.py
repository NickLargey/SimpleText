import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
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

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

X = pd.read_json('task 3/train/simpletext_task3_2024_train_snt_source.json')
y = pd.read_json('task 3/train/simpletext_task3_2024_train_snt_reference.json')

df = pd.merge(X, y, on='snt_id')

print('Before: ', df.shape)

df.drop_duplicates(subset='snt_id', inplace=True)

print('After: ', df.shape)
print(df.head())  

id = []
simplified_snts = []
src_snt = []
tgt_snt = []

pipeline = transformers.pipeline(
    "summarization",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
outputs = pd.DataFrame(simplified_snts, columns=['snt_id', 'simplified_snt', 'source_snt', 'target_snt'])

for idx, row in tqdm(df.iterrows()):
  src = row['source_snt']
  tgt = row['simplified_snt']
  
  messages = [
      {"role": "system", "content": """You are a high school teacher who needs to simplify this text english speaking student in grades 9 and 10. 
                                       Maximize the use of simple words and short sentences. 
                                       Optimize the output FKGL, ROUGE, SARI and BLEU scores.
                                       You are provided the full text and a target simplified text seperated by the string '<SPLIT>'."""},
      {"role": "user", "content": f'{src}<SPLIT>{tgt}'},
  ]
  prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

  terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

  outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
  
  result = outputs[0]["generated_text"][len(prompt):]
#   response = outputs[0][input_ids.shape[-1]:]

  id.append(row['snt_id'])
  simplified_snts.append(result)
#   simplified_snts.append(tokenizer.decode(response, skip_special_tokens=True))
  src_snt.append(src)
  tgt_snt.append(tgt)


sari = load("sari")
rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")
fkgl = evaluate.load("fkgl")

sources = outputs['source_snt'].tolist()
predictions = outputs['simplified_snt'].tolist()
references = outputs['target_snt'].tolist()

bleu_results = bleu.compute(predictions=predictions, references=references)
fkgl_results = fkgl.compute(predictions=predictions)
rouge_results = rouge.compute(predictions=predictions, references=references)
sari_results = sari.compute(sources=sources, prediction=predictions, references=references)

outputs['BLEU'] = bleu_results
outputs['FKGL'] = fkgl_results
outputs['ROUGE'] = rouge_results
outputs['SARI'] = sari_results


# outputs.to_json('simpletext_output.json')
# 
# 
#   input_ids = tokenizer.apply_chat_template(
    # messages,
    # add_generation_prompt=True,
    # return_tensors="pt"
    # ).to(model.device)
# 
# 
#   terminators = [
    #   pipeline.tokenizer.eos_token_id,
    #   pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#   ]
# 
#   outputs = pipeline(
    #   input_ids,
    #   max_new_tokens=256,
    #   eos_token_id=terminators,
    #   do_sample=True,
    #   temperature=0.6,
    #   top_p=0.9,
#   )