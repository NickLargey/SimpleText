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
    TrainingArguments,
    GenerationConfig,
    pipeline
)
from trl import SFTTrainer

torch.cuda.empty_cache()
load_dotenv()

token = os.getenv('HF_TOKEN')
login(token=token)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     torch.set_default_device("cuda")
#     print("cuda")
# else:
#     torch.set_default_device("cpu")

model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(Llama, device_map='auto',torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, truncation=True)
tokenizer.add_special_tokens({"pad_token":"<pad>"})
tokenizer.padding_side = 'left'

X = pd.read_json('task 3/train/simpletext_task3_2024_train_snt_source.json')
y = pd.read_json('task 3/train/simpletext_task3_2024_train_snt_reference.json')

merge_df = pd.merge(X, y, on='snt_id')

print('Before: ', merge_df.shape)

drop_df = merge_df.drop_duplicates(subset='snt_id')
df = drop_df.drop(["query_id", "query_text", "doc_id"], axis=1)



def format_instruction(sample):
    return f"""### Instruction:
        Summarize and simplify the text in order to maximize understanding while maintaining the original meaning. The output should maximize desired scores on FKGL, SARI, ROUGE and BLEU metrics for a student in 10th grade.

        ### Input:
        {sample['text']}

        ### Response:
        {sample['labels']}
        """


def label(data):
    return {'text': data['source_snt'], 'labels': data['simplified_snt']}

def tokenize_format(data):
    tokenized = tokenizer(data['text'], truncation=True, max_length=4096)
    return tokenized


################ PREPROCESS DATA ################ 
train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True)

train_dataset = Dataset.from_pandas(df).map(label, batched=True)
# train_dataset = train_dataset.map(tokenize_format, batched=True)

val_dataset = Dataset.from_pandas(val_df).map(label, batched=True)
# val_dataset = val_dataset.map(tokenize_format, batched=True)

train_dataset = train_dataset.remove_columns(["simplified_snt", "source_snt","__index_level_0__"])
train_dataset.set_format("torch")

val_dataset = val_dataset.remove_columns(["simplified_snt", "source_snt","__index_level_0__"])
val_dataset.set_format("torch")

dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
})

# features = Features({'snt_id': Value('string'),'source_snt': Value('string'),'simplified_snt': Value('string') })

# train_ds = Dataset.from_pandas(df.head(10), features=features, preserve_index=False).with_format("torch")
# train_dl = DataLoader(train_ds, batch_size=8, shuffle=False)

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

# user_message = X_src[:5]
# references = X_ctx[:5]

training_arguments = TrainingArguments(
        output_dir="./results",
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
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=3,
        # max_steps=10, #remove this
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=True
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

trainer.save_model("llama-3-70b-SimpleText")
