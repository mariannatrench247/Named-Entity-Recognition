from fastapi import FastAPI
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

app = FastAPI()

# Load trained model
model_path = "./models/ner_model"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Use MPS acceleration if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

@app.post("/predict/")
def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = outputs.logits.argmax(dim=-1).cpu().numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].cpu().numpy()[0])
    entities = [{"token": token, "entity": LABELS[pred]} for token, pred in zip(tokens, predictions)]

    return {"entities": entities}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
