## Goal

The goal of this task is to provide a simplified version of sentences extracted from scientific abstracts.
Participants will be provided with the popular science articles and queries and matching abstracts
of scientific papers, split into individual sentences.

## Data

3 uses the same corpus based on the sentences in high-ranked abstracts to the requests of Task 1.
Our training data is a truly parallel corpus of directly simplified sentences coming from scientific
abstracts from the DBLP Citation Network Dataset for Computer Science and Google Scholar and PubMed
articles on Health and Medicine. In 2024, we will expand the training and evaluation data. In addition
to sentence-level text simplification, we will provide passage-level input and reference simplifications.

## Evaluation

We will emphasize large-scale automatic evaluation measures (SARI, ROUGE, compression, readability) that
provide a reusable test collection. This automatic evaluation will be supplemented with a detailed human
evaluation of other aspects, essential for deeper analysis. We evaluate the complexity of the provided
simplifications in terms of vocabulary and syntax as well as the errors (incorrect syntax; unresolved anaphora
due to simplification; unnecessary repetition/iteration; spelling, typographic or punctuation errors). In
previous runs almost all participants used generative models for text simplification, yet existing evaluation
measures are blind to potential hallucinations with extra or distorted content. In 2024, we will provide new
evaluation measures that detect and quantify hallucinations in the output.

## Task 2:

In this part, students will explore one of the existing systems for their project. The goal is to
provide a baseline system for Part III. In the rare case that there are no existing systems for the
problem, students can use a baseline model of their choice. For example, if you decide to work
on legal information retrieval, and there are no systems available, you can use the TF-IDF model
as your baseline which is a general information retrieval system, not specific to a domain.
The objective is to run the baseline system on the project data, get the evaluation results, and
do a proper analysis of the results. Students should be able to discuss what worked and what
did not, by providing examples for each. Then they should provide an analysis of what they
think can be improved and form a research question/hypothesis for Part III of the project.
The deliverables for this phase include a minimum of three pages of report, codes of Git, and a
short presentation to the class. This phase also includes in-person delivery to the instructor
during the office hours.

## Model and Approach:

BERT for simplifying $\to$ Llama-2 for fine tuning

## Helpful Links

_Llama-2_
(LLaMA 2 - Every Resource you need)[https://www.philschmid.de/llama-2]

### Training

(Instruction-Tune Llama)[https://www.philschmid.de/instruction-tune-llama-2]
(Fine-tuning with PEFT)[https://huggingface.co/blog/llama2#fine-tuning-with-peft]
(Meta Examples and recipes for Llama model)[https://github.com/facebookresearch/llama-recipes/tree/main]
(The EASIEST way to finetune LLAMA-v2 on local machine!)[https://www.youtube.com/watch?v=3fsn19OI_C8&ab_channel=AbhishekThakur]
(Fine-tune Llama-2 in colab with QLoRA)[https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing]

### Deploying

(llama.cpp)[https://github.com/ggerganov/llama.cpp]
(Deploy LLaMa 2 Using text-generation-inference and Inference Endpoints)[https://huggingface.co/blog/llama2#using-text-generation-inference-and-inference-endpoints]

### Misc

(Llama-2 Resources)[https://gpus.llm-utils.org/llama-2-resources/]
