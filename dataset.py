from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_preprocess_dataset():
    dataset = load_dataset("conll2003")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_and_align_labels(example):
        tokenized_inputs = tokenizer(
            example["tokens"],
            truncation=True,
            padding="max_length",
            max_length=128,
            is_split_into_words=True
        )
        
        # Align labels with tokenized words
        labels = example["ner_tags"]
        tokenized_inputs["labels"] = labels  # Assign NER labels

        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    return tokenized_dataset, tokenizer

if __name__ == "__main__":
    dataset, tokenizer = load_and_preprocess_dataset()
    print("Dataset loaded and tokenized successfully!")
