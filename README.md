---
library_name: transformers
base_model: facebook/mbart-large-50-many-to-many-mmt
tags:
- generated_from_trainer
model-index:
- name: results
  results: []
datasets:
- VudhanthiNeeraja/Telugu-day-2-day
language:
- en
- te
pipeline_tag: text2text-generation
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

This model was created as part of a hackathon for SAWiT.

# Results

This model is a fine-tuned version of [facebook/mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) on an unknown dataset.

## Model description

This model is fine-tuned to translate text from English to Telugu. It is based on the mBART architecture, which is a multilingual sequence-to-sequence model pre-trained on a large corpus of text in multiple languages. The model has been fine-tuned on a dataset containing English sentences and their corresponding Telugu translations.

## Intended uses & limitations

Intended usees:

- Translating English text to Telugu.
- Assisting in language learning and translation tasks.
- Enhancing multilingual applications and services.

Limitations:

- The model may not perform well on domain-specific or highly technical text.
- The quality of translation may vary depending on the context and complexity of the input text.

## Training and evaluation data

Training data
The training data consists of a custom dataset containing English sentences and their corresponding Telugu translations.
The dataset includes a variety of common phrases and sentences used in everyday conversations.

Evaluation data
The evaluation data is a subset of the training data, split into training and testing sets. 
The evaluation data is used to assess the model's performance on unseen examples.

## Training procedure

1. Data Preparation
Dataset: The dataset used for training consists of English sentences and their corresponding Telugu translations. This dataset includes a variety of common phrases and sentences used in everyday conversations.
Splitting the Data: The dataset is split into training and testing sets. Typically, 80% of the data is used for training, and 20% is used for testing.

2. Model and Tokenizer Initialization
Model: The mBART model (facebook/mbart-large-50-many-to-many-mmt) is used as the base model. This model is pre-trained on a large corpus of text in multiple languages and is well-suited for multilingual translation tasks.
Tokenizer: The mBART tokenizer (MBart50TokenizerFast) is used to tokenize the input and output text. The tokenizer is configured to use English (en_XX) as the source language and Telugu (te_IN) as the target language.

3. Data Tokenization
Tokenization: The input English sentences and the target Telugu sentences are tokenized using the mBART tokenizer. The tokenized inputs are padded and truncated to a maximum length of 128 tokens to ensure uniformity.
Labels: The tokenized target sentences are used as labels for the model. These labels are also padded and truncated to a maximum length of 128 tokens.

4. Training Arguments
Learning Rate: A learning rate of 2e-5 is used for training.
Batch Size: The batch size is set to 4 for both training and evaluation to manage GPU memory usage.
Gradient Accumulation: Gradient accumulation steps are set to 4, allowing the model to simulate a larger batch size by accumulating gradients over several smaller batches.
Weight Decay: A weight decay of 0.01 is applied to prevent overfitting.
Number of Epochs: The model is trained for 3 epochs.

5. Training the Model
Trainer Initialization: The Seq2SeqTrainer class from the transformers library is used to handle the training process. The trainer is initialized with the model, training arguments, training dataset, evaluation dataset, and tokenizer.
Training: The model is trained using the training dataset. During training, the model learns to map English sentences to their corresponding Telugu translations.

6. Evaluation
Evaluation Metrics: The model is evaluated on the test set using metrics such as BLEU score and accuracy. These metrics help assess the model's performance on unseen examples.
Evaluation Process: The evaluation process involves generating translations for the test set and comparing them with the actual translations to calculate the evaluation metrics.

7. Saving and Pushing the Model
Saving the Model: After training, the model is saved using the trainer.save_model() method.
Pushing to Hub: The trained model is pushed to the Hugging Face Hub using the trainer.push_to_hub() method, making it accessible for others to use.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3
- mixed_precision_training: Native AMP

### Framework versions

- Transformers 4.48.3
- Pytorch 2.5.1+cu124
- Tokenizers 0.21.0

The model is available on HuggingFace: https://huggingface.co/VudhanthiNeeraja/results
