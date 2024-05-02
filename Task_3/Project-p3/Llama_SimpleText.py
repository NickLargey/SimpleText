import evaluate
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import login
import os
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
import torch
import peft
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer

torch.cuda.empty_cache()
load_dotenv()

token = os.getenv('HF_TOKEN')
login(token=token)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(Llama, device_map='auto',torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, truncation=True)
tokenizer.add_special_tokens({"pad_token":"<pad>"})
tokenizer.padding_side = 'left'

train_df = pd.read_csv("data_23/simpletext_task3_train.tsv", sep='\t',encoding = "ISO-8859-1")
ref_df = pd.read_csv("data_23/simpletext_task3_qrels.tsv", sep='\t',encoding = "ISO-8859-1")

merged_df = pd.merge(train_df, ref_df, on='snt_id')
print("TSV Merged: ", merged_df.shape)
# Merge to drop any missing values
x_df = merged_df[['snt_id', 'source_snt', 'simplified_snt']]

x_df['snt_id'] = x_df.apply(lambda x:'TSV'+x['snt_id'], axis=1)
x_df = x_df.rename(columns={'snt_id':'abs_id', 'source_snt':'source_abs', 'simplified_snt':'simplified_abs'})

X = pd.read_json('task 3/train/simpletext_task3_2024_train_abs_source.json')
y = pd.read_json('task 3/train/simpletext_task3_2024_train_abs_reference.json')

merge_df = pd.merge(X, y, on='abs_id')

df_list = [x_df, merge_df]

new_merged_df = pd.concat(df_list)
print("New Merged: ", new_merged_df.shape)
print('Combined: ', new_merged_df.shape)

drop_df = new_merged_df.drop_duplicates(subset='abs_id')
df = drop_df.drop(["query_id", "query_text", "doc_id"], axis=1)

print('After: ', df.info())

def format_instruction(sample):
    return f"""### Instruction:
        You are a high school science teacher, summarize and simplify the text in order to maximize understanding while maintaining the original meaning.

        ### Input:
        {sample['text']}

        ### Response:
        {sample['labels']}
        """


def label(data):
    return {'text': data['source_abs'], 'labels': data['simplified_abs']}

def tokenize_format(data):
    tokenized = tokenizer(data['text'], truncation=True, max_length=4096)
    return tokenized


################ PREPROCESS DATA ################ 
train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True)

train_dataset = Dataset.from_pandas(df).map(label, batched=True)
# train_dataset = train_dataset.map(tokenize_format, batched=True)

val_dataset = Dataset.from_pandas(val_df).map(label, batched=True)
# val_dataset = val_dataset.map(tokenize_format, batched=True)

train_dataset = train_dataset.remove_columns(["simplified_abs", "source_abs"])
train_dataset.set_format("torch")

val_dataset = val_dataset.remove_columns(["simplified_abs", "source_abs"])
val_dataset.set_format("torch")

dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
})

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
)

#Resize the embeddings
model.resize_token_embeddings(len(tokenizer))
#Configure the pad token in the model
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching

peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["q_proj","v_proj"]
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

training_arguments = TrainingArguments(
        output_dir="./results/8B/",
        evaluation_strategy="steps",
        do_eval=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=4,
        log_level="debug",
        optim="paged_adamw_32bit",
        save_steps=500, #change to 500
        logging_steps=10, #change to 100
        learning_rate=2e-4,
        save_strategy="epoch",
        eval_steps=200, #change to 200
        save_embedding_layers=True,
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=20,
        # max_steps=10, #remove this
        warmup_ratio=0.03,
        lr_scheduler_type="constant"
)

trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        packing=True,
        tokenizer=tokenizer,
        formatting_func=format_instruction,
        args=training_arguments,
)

trainer.train()

trainer.save_model("llama-3-8b-SimpleText-3.2-23")