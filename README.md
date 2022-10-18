# Debiasing Pretrained Text Encoders by Paying Attention to Paying Attention
This implementation contains Pytorch code for AttenD (Attention Debiasing), a finetuning method for reducing social biases from transformer-based text encoders (e.g. BERT, RoBERTa, ALBERT, DistilBert, SqueezeBERT) and evaluating both fairness and represnetativeness of the final debiased models.

You can find the difinition tuples in **'data/train/definition_words.json'**


## Requirements
- python 3.6 or higher
- pytorch_lightning==1.3.8
- numpy==1.19.5
- pandas==1.1.5
- tqdm==4.61.2
- transformers==4.8.2
- datasets==1.9.0
- torch==1.9.0
- scikit_learn==1.0
- matplotlib==3.4.3
- seaborn==0.11.2


## Training AttenD
To reduce social biases of any model, there is a need to specify its name (as in Huggingface's Transformers library) in both the tokenizer and model arguments. For example, if you want to finetune BERT and make it less biased, run the following:

`python train.py --tokenizer bert-base-uncased --model bert-base-uncased --output debiased_bert_base`

In our experiments, we applied AttenD on:
- bert-base-uncased
- bert-large-uncased
- roberta-base
- roberta-large
- albert-base-v2
- albert-large-v2
- distilbert-base-uncased
- squeezebert/squeezebert-uncased.

By defaut, equalization lambda is set to 2.0 and negative ratio to 0.8. You can change these and many other training arguments with their corresponding command line arguments.

You can name your debiased model however you like in the --output argument. The resulting model will be stored in **'saved_models/encoder/'**

To train only the top k heads, you have to use train_topk_heads.py. For example, to debias the top 10 most biased heads, run the following:

`python train_topk_heads.py --tokenizer bert-base-uncased --model bert-base-uncased --output debiased_bert_base_top10 --num_heads 10`


## Evaluating AttenD
In the paper, we propose several evalutions: to assess fairness and representativeness.

### 1) Fairness Evaluations


#### A. Likelihood-based evaluations
In order to run likelihood-based experiments, the text encoder needs first to be finetuned on a masked language modeling (mlm) objective. To finetune your model, run the following:

`python evaluation/train_mlm.py --tokenizer bert-base-uncased --model saved_models/encoder/debiased_bert_base --output debiased_bert_base_mlm`

The correposnding masked language model will be stored in **'saved_models/tasks/mlm/'**

To use StereoSet benchmark, run:

`python evaluation/evaluate_stereoset.py --tokenizer bert-base-uncased --model saved_models/tasks/mlm/debiased_bert_base_mlm`

As for Crows-Pairs challange benchmark, run the following:

`python evaluation/evaluate_crows-pairs.py --input_file data/evaluation/crows-pairs/crows_pairs_anonymized.csv --output_file tmp.txt --tokenizer bert-base-uncased --lm_model saved_models/tasks/mlm/debiased_bert_base_mlm`



#### B. Computing attention bias
To compute the amount of attention bias present in a given text encoder with respect to a bias type (e.g. gender, race or religion), run:

`python evaluation/compute_attention_bias.py --tokenizer bert-base-uncased --model saved_models/encoder/debiased_bert_base --bias_type race`



#### C. Inference-based evaluations
The text encoder must be finetuned on an inference task in order to be usable in inference evaluations. To train your debiased text encoder on mnli dataset:

`python evaluation/train_glue.py --tokenizer bert-base-uncased --model saved_models/encoder/debiased_bert_base --task mnli --output debiased_bert_base_mnli`


Following previous work [(Dev and al., 2020)](https://arxiv.org/abs/1908.09369), we use their scripts to generate evaluation data from a set of templates. These templates have been specifically designed for fairness evaluations. Their Github repository can be found [here](https://github.com/sunipa/On-Measuring-and-Mitigating-Biased-Inferences-of-Word-Embeddings). 

Suppose we generate an evaluation dataset on religion that we keep in **'data/evaluation/nli/'**, we run the inference-based experiment like this:

`python evaluation/evaluate_bias_nli.py --tokenizer bert-base-uncased --model saved_models/tasks/mnli/debiased_bert_base_mnli --eval_filepath data/evaluation/nli/religion.csv`



#### D. Application-oriented evaluations
In this work, we use the task of hate speech detection where the model classifies a snippet of text as either offensive or not. To train the text encoder on HateXplain dataset:

`python evaluation/train_hate_speech.py --tokenizer bert-base-uncased --model saved_models/encoder/debiased_bert_base --output debiased_bert_base_hatexplain`

In this work, we use the bias metrics proposed by [(Borkan and al., 2019)](https://arxiv.org/abs/1903.04561) and later expanded by [(Mathew and al., 2020)](https://arxiv.org/abs/2012.10289).

Then to run the application-oriented experiment as measured by AUC-based metrics:

`python evaluation/evaluate_hate_speech.py --tokenizer bert-base-uncased --model saved_models/tasks/hate_speech/debiased_bert_base_hatexplain`


### 2) Representativeness Evaluations

We use GLUE benchmark to asses the representativeness of the debiased text encoder. We first train it for a task...

`python evaluation/train_glue.py --tokenizer bert-base-uncased --model saved_models/encoder/debiased_bert_base --task cola --output debiased_bert_base_cola`

...and then evaluate.

`python evaluation/evaluate_glue.py --tokenizer bert-base-uncased --model saved_models/tasks/cola/debiased_bert_base_cola --task cola`

In our scripts, we allow the following GLUE tasks: cola, sst2, mrpc, stsb, mnli, wnli and rte.



### 3) Qualitative analysis
In order to produce the figures available in the appendix:

`python evaluation/qualitative/draw_att_figures.py --tokenizer bert-base-uncased --model saved_models/encoder/debiased_bert_base --output attd_bert.png`
