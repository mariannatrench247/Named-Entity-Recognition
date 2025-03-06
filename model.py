import torch
from transformers import AutoModelForTokenClassification

def load_model():
    model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=9)

    # Use MPS for Apple Silicon acceleration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    return model, device

if __name__ == "__main__":
    model, device = load_model()
    print("Model loaded and moved to:", device)
