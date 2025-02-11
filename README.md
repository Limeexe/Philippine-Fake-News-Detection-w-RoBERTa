# Philippine-Fake-News-Detection-w-RoBERTa

This is a machine learning model created for the purpose of the research thesis entitled "Development and Prototype Implementation of a Browser Extension for Fake News Detection in Philippine News Using Natural Language Processing Algorithms" in Computer Science Thesis 1 & 2 subject in Bachelor of Computer Science in Camarines Sur Polytechnic Colleges. The model uses a **Fine-Tuned pre-trained RoBERTa-Base Model** to predict if news article is either credible or suspicious.


This repository contains code for training a machine learning model to classify news content as either "Credible" or "Suspicious". The system includes:

1. Text preprocessing steps:
   - Cleaning and tokenization
   - Word cloud generation
   - Numerical feature calculation (sentiment, word count, readability)
   - Named entity recognition

2. Sentiment analysis using TextBlob
3. Custom model architecture using RoBERTa with numerical features
4. Training loop with early stopping validation
5. Performance evaluation metrics and visualizations

## Requirements

To run this code, you will need the following Python libraries:

```bash
pip install pandas numpy torch transformers spacy textstat fxsolve preprocessed-datasets datasets accelerate tabulate sentenceblob ROBERTATokenizeraires PretrainedConfig PreTrainedModel AdamW plotly seaborn
```
## Data

The `nixbel/dataset_train_thesis` dataset is used. You can load this dataset directly from Hugging Face.

```bash
pip install huggingface datasets
```

## Example Workflow

```bash
# Load and preprocess data
df = load_dataset("nixbel/dataset_train_thesis", split="train")
df = preprocess_data(df)

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Label'])

# Initialize model and training parameters
config = NewsClassifierConfig()
model = NewsClassifier(config)

# Train model
train_loader = DataLoader(TrainDataset(...), batch_size=32, shuffle=True)
val_loader = DataLoader(ValDataset(...), batch_size=32)

# Train for 10 epochs with early stopping
model, metrics = train_model(model, train_loader, val_loader, ...)

# Generate visualizations and report
print("\nClassification Report:")
print(classification_report(test_labels, test_predictions))

# Save model for inference
save_model_for_inference(model, tokenizer, df)
```

## Model Architecture

The model extends the `NewsClassifier` class from scratch. It combines a RoBERTa transformer with numerical features using PyTorch.

## Known Issues
- Initial model training and deployment may require additional optimization for performance.
- Limited testing has been conducted on edge cases; further validation is recommended.

## Contributors
The project was developed by Team LiveANet
