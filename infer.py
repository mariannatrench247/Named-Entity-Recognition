import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load trained model
model_path = "./models/ner_model"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Use MPS acceleration if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Label mapping (conll2003 dataset labels)
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

def predict_entities(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = outputs.logits.argmax(dim=-1).cpu().numpy()[0]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].cpu().numpy()[0])
    entities = [(token, LABELS[pred]) for token, pred in zip(tokens, predictions)]

    return entities

if __name__ == "__main__":
    text = "Apple Inc. is based in Cupertino, California and was founded by Steve Jobs."
    predictions = predict_entities(text)

    print("Predicted Named Entities:")
    for token, entity in predictions:
        print(f"{token}: {entity}")
