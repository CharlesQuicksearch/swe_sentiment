import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

with open("model_config.json", "r") as f:
    config = json.load(f)

model = AutoModelForSequenceClassification.from_pretrained(config.get("model_path"))
tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_path"))

async def rate(input):
    tokens = tokenizer.encode(input, return_tensors="pt")
    result = model(tokens)
    output_np = result.logits[0].detach().cpu().numpy()
    output = softmax(output_np)

    return output
