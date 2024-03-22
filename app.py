import uvicorn
import numpy as np
from fastapi import FastAPI
import schemas
import pickle
import warnings
warnings.filterwarnings('ignore')

# create pickle file from the code available on ipynb
app = FastAPI()
pickle_in = open("models/random_forest.pkl","rb")
rf_reg = pickle.load(pickle_in)

@app.post('/predict')
def predict_price(data: schemas.Price):
    data = data.model_dump()
    sqft_living = data['sqft_living']
    waterfront = data['waterfront']
    view = data['view']
    grade = data['grade']
    lat = data['lat']
    long = data['long']
    sqft_living15 = data['sqft_living15']

    pred = np.array([[grade, sqft_living, lat, long, sqft_living15, view, waterfront]]).reshape(1, -1)
    prediction = rf_reg.predict(pred)
    return {f'House price for the given feature is {int(prediction)}'}

@app.get('/')
def hello(data: str):
    return {'data' : 'hello, world'}

if __name__ == '__main__':
   uvicorn.run(app,host="127.0.0.1",port=8000)