## Task 3: Simplify Scientific Text

The goal of this task is to provide a simplified version of sentences extracted from scientific abstracts. Participants will be provided with the popular science articles and queries and matching abstracts of scientific papers, split into individual sentences.

### Data
3 uses the same corpus based on the sentences in high-ranked abstracts to the requests of Task 1. Our training data is a truly parallel corpus of directly simplified sentences coming from scientific abstracts from the DBLP Citation Network Dataset for _Computer Science_ and Google Scholar and PubMed articles on _Health and Medicine_. In 2024, we will expand the training and evaluation data. In addition to sentence-level text simplification, we will provide passage-level input and reference simplifications.

### Evaluation

We will emphasize large-scale automatic evaluation measures (SARI, ROUGE, compression, readability) that provide a reusable test collection. This automatic evaluation will be supplemented with a detailed human evaluation of other aspects, essential for deeper analysis. We evaluate the complexity of the provided simplifications in terms of vocabulary and syntax as well as the errors (incorrect syntax; unresolved anaphora due to simplification; unnecessary repetition/iteration; spelling, typographic or punctuation errors). In previous runs almost all participants used generative models for text simplification, yet existing evaluation measures are blind to potential hallucinations with extra or distorted content. In 2024, we will provide new evaluation measures that detect and quantify hallucinations in the output.

### A Prompt Engineering Approach to Scientific Text Simplification (2023):

-  **Transformer-based models:**
	- T5 is based on the Transformer encoder-decoder architecture,
	- T5, or Text-to-Text Transfer Transformer [1], is a Transformer based
	  architecture that uses a text-to-text approach.
	- T5 is trained on a large corpus of text using a text-to-text framework, where it learns to map any input text to any output text.

	- GPT4 is based on the Transformer decoder-only architecture.
	- GPT4, on the other hand, is trained on a large corpus of text using an
	  auto-regressive language modeling objective, where it learns to predict the next word given the previous words.
	- GPT4 model is it may not always produce factual or ethical texts.
	- Not be able to capture the nuances and contexts of human communication,    such as sarcasm, humor, irony, etc. 
	- require a lot of computational resources and energy to run and maintain.

- **Evaluation:**
	- evaluate them with three metrics.
	- **Flesch reading ease** is an index that measures the level of sentences; the higher the score, the easier it is for the reader.
	- **Flesch-Kincaid grade level** (FKGL) is an index that measures the corresponding reader level; the lower the score, the younger the reader.
	- The **percentage of academic words** according to a list provided by Coxhead.

- **Example Prompts Used:**
	- GPT-3.5 - Your task is to simplify the following sentences to make them
		easier to understand. Please note that your response should be flexible enough to allow for various relevant and creative simplifications, as long as they accurately convey the intended meaning.
	- GPT-4 - Please simplify this sentence:{text}

- **Results:**

| FRE   | FKGL | Length | Run    |
| ----- | ---- | ------ | ------ |
| 8.54  | 21.3 | 264    | Source |
| 52.53 | 12.6 | 165    | Run 1  |
| 54.22 | 9.9  | 118    | Run 2  |
| 28.84 | 13.5 | 130    | Run 3  |
| 35.27 | 13.1 | 140    | Run 4  |



### Automatic Simplification of Scientific Texts using Pre-trained Language Models (2023):

- **Models:**
	- Simple T5 is a model built on top of PyTorch Lightning and Transformers. It allows users to quickly train their T5 models, including T5, mT5, and by T5 models, with only a few lines of code [10]. 
	- AI21 Labs - Jurassic-2 Grande Instruct The J2-Grande-Instruct model is a variation of the Jurassic-2 series developed by AI21. It is an auto-regressive language model based on the Transformer architecture and designed with modifications for improved efficiency.
	- BLOOM (BigScience Large Open-science Open-access Multilingual Language Model) The BLOOM model is an auto-regressive Large Language Model (LLM) that leverages a decoder only transformer architecture, derived from Megatron-LM GPT-2.
- **Evaluation:**
- **BLEU (Bilingual Evaluation Understudy)**: A metric for evaluating a generated text based on how many words it has in common with reference texts, considering both the order of the words and their presence. It's widely used in machine translation but also applicable to text simplification.
    
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: This set of metrics includes ROUGE-N (evaluating n-gram overlap between generated and reference texts), ROUGE-L (considering the longest common subsequence), and variations focusing on precision (the proportion of generated n-grams present in the reference), recall (the proportion of reference n-grams captured in the generated text), and f-measure (a harmonic mean of precision and recall).
    
- **Semantic Match**: While not a standardized metric like BLEU or ROUGE, this term refers to the extent to which the simplified text retains the original meaning of the source text. It can involve various techniques for evaluation, including manual assessment or computational methods that estimate semantic similarity.

- **SARI (System output Against References and against the Input sentence):** Unlike other metrics that mainly focus on the similarity between the system output and reference simplifications (e.g., BLEU, ROUGE), SARI also considers the input text to evaluate the simplification quality. It measures the goodness of words that are added, deleted, or kept by the simplification system, providing a more comprehensive assessment of how well the system performs across these three aspects. This approach enables SARI to more accurately reflect the quality of text simplifications, especially in terms of preserving meaning and improving readability, which are crucial for simplification tasks.

### Controllable Sentence Simplification Using Transfer Learning (2022):

- **Models:**
	- To predict masked tokens, we used the COVID-SciBERT model, an expanded version of SciBert with the articles present in the competition COVID-19 Open Research Dataset Challenge (CORD-19). If a sentence contains simple concepts, the Fill-Mask model will be able to predict them before the complex version.
	- T5 has shown promising results in different tasks such as text summation, question-answering and classification problems.

- Control Tokens: The amount of text compression (Chars), word length (Words), paraphrasing (LevSim) and syntactic complexity (DepTreeDepth)
- Chars (CLR): Character length ratio between the original and the target sentence (the simplified version). The number of characters in the simplified version is divided by the number of characters in the original. Previous work has shown a correlation between simplicity and the number of characters in the sentence.
- LevSim (LR): Levenshtein normalized similarity at character-level between the original and the simplified version. This feature is a measure of the modifications made including its paraphrase level.
- DepTreeDepth (DTDR): Maximum depth of the dependency tree of the simplified version divided by that of the source, under the assumption that a simple sentence makes use of syntactic structures with fewer dependencies than its complex version.
- Words (WLR): Ratio of the number of words between the original and the simplified sentence. The number of words in a simple sentence is divided by the number of words in the complex sentence.
- Language Model Fill-Mask (LMFMR): Position within the prediction ranking of all masked words in the simplified version divided by that of the original. A language model trained on a masking task can predict earlier the set of masked words in a simple sentence than in a complex sentence. LMFMR feature values distribution of the training dataset can be found in Figure 1.
![[Pasted image 20240302163356.png]]

- **Results:**
	- From the hyperparameter search performed with Optuna, the best result was obtained with CLR=0.6, DTDR=0.95, LMFMR=0.75, LR=0.6 and WLR=0.75. With these hyperparameters, the final result was a SARI value of 37.40. Adding our feature, the model obtains a +0.35 SARI improvement.

### CLEF 2023: Scientific Text Simplification and General Audience

- **Models:**
	- SimpleT5 and GPT-3

- **Process:**
	- Workflow followed these steps:
		 a. Prepared a train dataset by merging input and output sentences.
		 b. Added the prefix "simplify: " to each source_text.
	- For SimpleT5:
		 c. Trained a pretrained SimpleT5 model on the train dataset for five epochs. The best model was selected based on the performance after the first epoch, with a train loss of 0.9715 and a validation loss of 1.1263.
		 d. Predicted simplified sentences using the trained model on a small test dataset.
	- For GPT-3:
		 e. Split the test dataset into 20 chunks for efficiency.
		 f. Performed partial predictions using GPT-3 and saved the results in JSON files.
		 g. Joined all JSON files to compile the predictions.

**Results:**

| Model    | FKGL   | SARI   | BLEU   | Comp. Ratio | Sen. Splits | Levenshtein Sim. | Lexical Complexity |
| -------- | ------ | ------ | ------ | ----------- | ----------- | ---------------- | ------------------ |
| GPT-3    | 8.083  | 34.597 | 6.909  | 0.433       | 0.989       | 0.482            | 8.620              |
| SimpleT5 | 13.397 | 41.401 | 45.189 | 0.894       | 0.996       | 0.915            | 8.686              |
### HULAT-UC3M at SimpleText@CLEF-2022: Scientific text simplification using BART:

- **Model:**
	- BART - The main component of BART are transformers, as BART stands for Bidirectional Auto Regressive Transformers. A transformer is a sequence-to-sequence component based in a encoder-decoder architecture
- **Code:** https://github.com/Adrubio12/text-simplification

### Text Simplification of Scientific Texts for Non-Expert Readers (2023):

- **Model:**
	- Prompt Engineering and complex phrase identification preprocessing with ChatGPT 
	- T5
	- PEGASUS

- **ChatGPT Approach:**
![[Pasted image 20240303121611.png]]
	
- Since identifying keyphrases is an already established task, we use KBIR-inspec, a pretrained model which is available on Hugginface. 
- Since the domain of the task is scientific texts, we evaluate the complexity
  of terms using the term statistics of two text corpora: a dataset that is comprised of texts from lifestyle forums and one that is comprised of texts from science-focused forums.
- We obtain our complexity estimate by the difference of the inverse document   frequency (idf) of term t from the lifestyle dataset with the idf value of the
  scientific dataset. The complexity of a phrase < t1 , ..., tn > is defined by the function φ:

![[Pasted image 20240303123108.png]]
- Here $df_{lf}$ (t) is the number of documents from the lifestyle dataset in which the term t appears,$df_{sc}(t)$ for the science dataset respectively. The total number of documents is N. We set the complexity threshold to 0.01, so every phrase above this threshold gets tagged as complex.
- To index and analyze the datasets and term statistics, we use the PyTerrier framework.

- **"Out-of-the-Box Sequence-to-Sequence Models" Approach:**
	-  Two runs based on the T5, One run that uses PEGASUS
	- We instruct the model to summarize all source texts. We do not consider the queries but only the source sentences, so the source_snt.
	-  We use batch processing of texts which we want to simplify, set the maximum input length to 512 tokens per example, pad the inputs, and determine outputs to have a maximum length of 100 tokens.
	- This leads to a model occasionally creating multiple sentences in one summarization, each fully summarizing the original text. Therefore, out of these merged simplifications (summarizations), we pick the shortest sentence as the simplified version of the original text.
	- Readability measures have all been implemented using the Python library [textstat](https://pypi.org/project/textstat/)

- **Results:**
![[Pasted image 20240303125614.png]]

![[Pasted image 20240303125816.png]]
- Scores between 30 and 49 indicate difficult text.
- The **new Dale-Chall** score indicates the reading level of a text as a grade corresponding to the familiarity of persons from that grade with a list of the 3000 most common English words. Scores are defined up to a value between 9 and 9.9 which corresponds to the reading level of an average college student. The higher the score, the more difficult a text.
- The number of **difficult words** indicates the number of words with ≥ 3 syllables and which are not in a predefined list of easy words.
- The **reading time** indicates the seconds required to read a text; each character taking 14.69ms.
- The **syllable count** gives the number of syllables in a text.
- The **lexicon count** gives the number of different words in a text.
- The **sentence count** indicates how many sentences a text consists of.

For these last three measures, one could argue that a simplified text should be shorter in terms of containing fewer syllables, words and sentences compared to the original text which also is in line with previous work. The artificially constructed example in Table 2 shows that this oversimplified assumption does not always hold.

- **Manual Evaluation:**
	- **PEGASUS** model produces texts, that are almost identical to the original   text.
	- The **T5** model is significantly shorter and grammatically simpler but omits important information from the source sentence. the text still contains scientific formulations ("conceptual framework") and a case of AI hallucination in the form of reporting the text as a quote.
	- Our automatic evaluation did not rank **ChatGPT** as the best run, a manual analysis evaluated the texts produced through ChatGPT as the best. Although we did not explicitly evaluate the inclusion of complex phrase identification in the ChatGPT run, we found it to improve the system’s effectiveness. 

- **Challenges:**
	- Labeling complex phrases using square brackets in our approach may pose challenges when the input text already contains square brackets. This problem could be circumvented by implementing an additional preprocessing step.
	- Hallucinated content, possibly attributed to the absence of context in data. A possible improvement for the summarization model approaches would be flagging difficult words as ones we want to exclude in the simplified (summarized) variant.