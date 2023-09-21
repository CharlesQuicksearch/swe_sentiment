from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

model = AutoModelForSequenceClassification.from_pretrained("KBLab/KBLab_Swe_Model")
tokenizer = AutoTokenizer.from_pretrained("KBLab/KBLab_Swe_Tokenizer")


async def swe_review_rating(input):
    tokens = tokenizer.encode(input, return_tensors="pt")
    result = model(tokens)
    output_np = result.logits[0].detach().cpu().numpy()
    output = softmax(output_np)

    return output
