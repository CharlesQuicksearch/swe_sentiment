import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model = AutoModelForSequenceClassification.from_pretrained("model/KBLab/robust-swedish-sentiment-multiclass")
tokenizer = AutoTokenizer.from_pretrained("model/KBLab/robust-swedish-sentiment-multiclass")


async def swe_review_rating(input):
    tokens = tokenizer.encode(input, return_tensors="pt")
    result = model(tokens)
    output_np = result.logits[0].detach().cpu().numpy()

    return output_np
