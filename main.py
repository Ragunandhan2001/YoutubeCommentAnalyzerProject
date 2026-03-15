from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

sentiment_model = pipeline("sentiment-analysis")

class Comment(BaseModel):
    text : str

@app.post("/analyze")
def analyze_comment(data : Comment):

    result = sentiment_model(data.text)[0]

    label = result["label"]
    score = round(result["score"] * 100,2)

    return {
        "sentiment": label,
        "confidence_percent": score
    }


# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import pipeline
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# # Enable CORS for frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load sentiment analysis model
# sentiment_model = pipeline("sentiment-analysis")

# class Comment(BaseModel):
#     text: str


# @app.post("/analyze")
# def analyze_comment(data: Comment):

#     result = sentiment_model(data.text)[0]

#     label = result["label"]
#     score = round(result["score"] * 100, 2)

#     return {
#         "sentiment": label,
#         "confidence_percent": score
#     }