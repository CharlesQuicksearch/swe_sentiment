from fastapi import FastAPI, HTTPException
from sentiment_swe import swe_review_rating
from request_and_response import Response, Request

app = FastAPI()

@app.get("/")
def home():
    return "Rate sentiment. Send '{'input':'example string'}'"

@app.post("/swe_sentiment/", response_model=Response)
async def rate_a_string(request_data: Request):
    try:
        rating = await swe_review_rating(request_data.input)
        print(rating)
        return Response(output = rating) #Negative, Neutral, Positive: Positive=rating[2], Negative=rating[0], Neutral=rating[1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
