from fastapi import FastAPI
from pydantic import BaseModel

class Price(BaseModel):
    sqft_living: float
    waterfront: float
    view: float
    grade: float
    lat: float
    long: float
    sqft_living15: float
    