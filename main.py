import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentiment_swe import swe_review_rating

app = FastAPI()

class RateRequest(BaseModel):
    input: str
class RateResponse(BaseModel):
    Positive: float
    Neutral: float
    Negative: float

# NY KOMMENTAR

@app.get("/")
def home():
    return str("Test")

#asdasd
@app.post("/swe_string_to_rate/", response_model=RateResponse)
async def rate_a_string(request_data: RateRequest):
    try:
        rating = await swe_review_rating(request_data.input)
        return RateResponse(Positive=rating[0], Negative=rating[1], Neutral=rating[2])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))