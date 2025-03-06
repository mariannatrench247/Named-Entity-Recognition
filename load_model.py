from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load the trained model
model_path = "./models/ner_model"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained(model_path)

print("Model loaded successfully!")
