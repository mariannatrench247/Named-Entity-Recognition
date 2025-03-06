***Named Entity Recognition***


This trained Named Entity Recognition (NER) model is designed to identify and classify named entities in text. 

This means it can recognize specific types of words or phrases, such as:

Persons (PER) → Identifies names of people
Organizations (ORG) → Identifies companies, institutions, brands, etc.
Locations (LOC) → Recognizes cities, states, countries, and other locations
Miscellaneous (MISC) → Captures other significant named entities that don’t fit into the main categories


1️⃣ Dataset: conll2003 (NER Dataset)
This dataset contains news articles with annotated entities:
ORG (Organizations)
LOC (Locations)
PER (Persons)
MISC (Miscellaneous)
Source: Hugging Face conll2003
2️⃣ Model: DistilBERT for Token Classification
Lightweight Transformer model
Faster than BERT, requires less memory
Supports fine-tuning on MPS (Apple Metal Performance Shaders)
Works well for Named Entity Recognition (NER)
3️⃣ Steps to Train on MacBook Air M3
Step 1: Install Dependencies
bash
Copy
Edit
pip install torch torchvision torchaudio transformers datasets accelerate -U
Enable Apple MPS acceleration by using PyTorch with Metal backend.
