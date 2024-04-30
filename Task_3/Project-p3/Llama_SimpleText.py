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

Llama = "/home/nicholas.largey/Desktop/code/Llama/llama-2-7b/weights"
# model = AutoModelForCausalLM.from_pretrained(Llama, device_map='auto',torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(Llama, use_fast=True, truncation=True)
tokenizer.add_special_tokens({"pad_token":"<pad>"})
tokenizer.padding_side = 'left'

train_df = pd.read_csv("SimpleText_data/simpletext_task3_train.tsv", sep='\t')
ref_df = pd.read_csv("SimpleText_data/simpletext_task3_qrels.tsv", sep='\t')

merged_df = pd.merge(train_df, ref_df, on='snt_id')

# Merge to drop any missing values
df = merged_df[['snt_id', 'source_snt', 'simplified_snt']]

X = df.drop('snt_id', axis=1)

X_train, X_val = train_test_split(X, test_size=0.2)

X_src = X_train['source_snt'].tolist()
X_ctx = X_train['simplified_snt'].tolist()



def format_instruction(sample):
    return f"""### Instruction:
        Summarize and simplify the text in order to maximize understanding while maintaining the original meaning. The output should maximize desired scores on FKGL, SARI, ROUGE and BLEU metrics for a student in 9th grade.

        ### Input:
        {sample['text']}

        ### Response:
        {sample['labels']}
        """


def label(data):
    return {'text': data['source_snt'], 'labels': data['simplified_snt']}

def tokenize_format(data):
    tokenized = tokenizer(data['text'], truncation=True, max_length=256)
    return tokenized


################ PREPROCESS DATA ################ 
train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True)

train_dataset = Dataset.from_pandas(X_train).map(label, batched=True)
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


compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
        Llama, quantization_config=bnb_config, device_map="auto"
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

user_message = X_src[:5]
references = X_ctx[:5]

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

trainer.save_model("llama-2-7b-SimpleText")

# system_message = """Summarize and simplify the text in order to maximize understanding while maintaining the original meaning. The output should maximize desired scores on FKGL, SARI, ROUGE and BLEU metrics for a student in 9th grade."""


# print(predictions)
# bleu = evaluate.load("bleu")
# blue_results = bleu.compute(predictions=predictions, references=references)

# rouge = evaluate.load('rouge')
# for i in range(5):
#     rouge_results = rouge.compute(predictions=predictions[i], references=references[i])
#     print(rouge_results)

# rouge_results = rouge.compute(predictions=predictions, references=references)

# sari = evaluate.load("sari")
# sources = ["About 95 species are currently accepted ."]
# predictions = ["About 95 you now get in ."]
# references = [["About 95 species are currently known .",
#                "About 95 species are now accepted .", "95 species are now accepted ."]]
# sari_score = sari.compute(
#     sources=sources, predictions=predictions, references=references)


# system_message = "You are teaching a science class for middle and high school students. You have been given exerpts from scientific writings, but need to simplify them for your students. Please rewrite the text in order to maximize understanding while maintaining the original meaning.Your results should maximize desired scores on FKGL, SARI, ROUGE and BLEU metrics."
# user_message = X_list[:5]

# prompt = "<s>[INST]<<SYS>>" + system_message + \
#     "<</SYS>>\n" + user_message + "[/INST]</s>"

    # Llama_pipeline = transformers.pipeline(
    #     "summarization",
    #     model=Llama,
    #     tokenizer=Llama_tokenizer,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    #     max_new_tokens=5000,
    # )


# sequences = Llama_pipeline(
#     prompt,
#     do_sample=True,
#     top_k=1,
#     num_return_sequences=1,
#     eos_token_id=Llama_tokenizer.eos_token_id
# )
# # print(len(sequences))
# for seq in sequences:
#     print(seq)
#     result = seq['generated_text']
