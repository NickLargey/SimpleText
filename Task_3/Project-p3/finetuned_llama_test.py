import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv
import peft
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import os
import re
import textstat
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, model_id):
        self.df = df
        self.tokenizer = tokenizer
        self.model_id = model_id

        # compute_dtype = getattr(torch, "float16")
        # bnb_config = BitsAndBytesConfig(
        #         load_in_4bit=True,
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_compute_dtype=compute_dtype,
        #         bnb_4bit_use_double_quant=True,
        # )
        # model = AutoModelForCausalLM.from_pretrained(
        #         self.model_id, quantization_config=bnb_config, device_map="auto"
        # )
        
        # #Resize the embeddings
        # model.resize_token_embeddings(len(self.tokenizer))
        # #Configure the pad token in the model
        # model.config.pad_token_id = self.tokenizer.pad_token_id
        # model.config.eos_token_id = self.tokenizer.eos_token_id
        # model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching
        
        # peft_config = LoraConfig(
        #         lora_alpha=32,
        #         lora_dropout=0.1,
        #         r=8,
        #         bias="none",
        #         task_type="CAUSAL_LM",
        #         target_modules= ["q_proj","v_proj"]
        # )
        
        # model = prepare_model_for_kbit_training(model)
        # model = get_peft_model(model, peft_config)

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
        }

load_dotenv()
token = os.getenv('HF_TOKEN')
login(token=token)

model_id = "llama-3-8b-SimpleText-3.2-23"

# Load the data
X = pd.read_json('task 3/test/simpletext_task3_2024_test_snt_source.json')
drop_df = X.drop_duplicates(subset='snt_id')
df = drop_df.drop(["query_id", "query_text", "doc_id"], axis=1)

# Create a dataset instance
dataset = TextDataset(df, transformers.AutoTokenizer.from_pretrained(model_id), model_id)

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


    results.append({
        'run_id': 'AIIRLab_Task3.1_llama-3-8b',
        'manual': 0,       
        'snt_id': snt_id,
        'simplified_snt': simplified_snt,
    })

# Create a DataFrame from the results list
output_df = pd.DataFrame(results)

fkgl_sum = 0
for i,r in output_df.iterrows():
    fkgl_sum += textstat.flesch_kincaid_grade(str(r['simplified_snt']))

avg_fkgl = fkgl_sum/len(output_df['simplified_snt'])
print(avg_fkgl)

# Write the DataFrame to a JSON file
output_df.to_json('simpletext_output.json', orient='records')