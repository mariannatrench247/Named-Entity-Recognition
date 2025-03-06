import coremltools as ct
import torch

# Load the trained model
model = AutoModelForTokenClassification.from_pretrained("./models/ner_model")

# Convert to Core ML
dummy_input = torch.ones(1, 128, dtype=torch.int32)
traced_model = torch.jit.trace(model, dummy_input)
mlmodel = ct.convert(traced_model)

# Save Core ML Model
mlmodel.save("ner_model.mlmodel")
