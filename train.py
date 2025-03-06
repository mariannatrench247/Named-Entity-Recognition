import torch
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from dataset import load_and_preprocess_dataset
from model import load_model

def train():
    tokenized_dataset, tokenizer = load_and_preprocess_dataset()
    model, device = load_model()

    # Use a collator that correctly formats labels
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./ner_results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir="./logs",
        report_to="none",
        push_to_hub=False,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save model
    model.save_pretrained("./models/ner_model")
    print("Model training complete and saved!")

if __name__ == "__main__":
    train()
