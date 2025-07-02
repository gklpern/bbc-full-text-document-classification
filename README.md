# bbc-full-text-document-classification


# BBC Full-Text Document Classification

This project focuses on classifying full-text BBC news articles into five categories using transformer-based models fine-tuned with LoRA (Low-Rank Adaptation).

**Categories:**  
- business  
- entertainment  
- politics  
- sport  
- tech  

## Overview

The goal of this work is to implement a parameter-efficient document classification pipeline using `bert-base-uncased` and `bert-base-cased` models with LoRA fine-tuning. The steps followed in the notebook are:

- Basic exploratory data analysis was performed, including class distributions and length-based text statistics.
- A custom normalization function was applied to clean and prepare the text data.
- A LoRA configuration was defined with:
  - `r = 16`
  - `lora_alpha = 32`
  - `lora_dropout = 0.2`
  - `task_type = sequence classification`
- The model was trained using the `transformers` and `peft` libraries on the BBC dataset for 8 epochs.
- Fine-tuning was done using PyTorch on GPU with the `bert-base-cased` model achieving the best validation accuracy of **0.964**.
- Results indicate that higher accuracy is achievable with more computational resources, longer training, or optimized hyperparameters.

## Data

The dataset consists of full-text news articles collected from the [BBC news dataset](https://www.kaggle.com/datasets/cpichot/bbc-news). Each article is labeled with one of five topic categories.

## Text Normalization

A custom normalization function was used, which includes:

```python
def normalization_bert_cased(text):
    text = remove_whitespace(text)
    text = re.sub('\n' , '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub(r'@', '', text)
    text = text.replace('&', 'and')
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#', '', text)
    text = remove_http(text)
    text = remove_html(text)
    text = remove_emoji(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

```




# Requirements

    Python 3.10+

    PyTorch

    Transformers (Hugging Face)

    PEFT

    scikit-learn

    tqdm

# Install dependencies using:

```pip install -r requirements.txt```

# Results

    bert-base-cased fine-tuned with LoRA achieved a validation accuracy of 0.878.

    bert-base-uncased performed slightly lower under identical conditions.

    With additional resources, longer training or larger batch sizes, further improvements are possible.




