# Put the code for your API here.
from fastapi import FastAPI()
from starter.ml.model import inference
from pydantic import BaseModel, Field
from typing import Optional

import pickle
import uvicorn
import pandas as pd

class Data(BaseModel):
    age: int = Field(..., example=28)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=185061)
    education: str = Field(..., example="11th")
    education_num: int = Field(..., alias="education-num", example=7)
    marital_status: str = Field(...,
                                alias="marital-status",
                                example="Never-married")
    occupation: str = Field(..., example="Handlers-cleaners")
    relationship: str = Field(..., example="Other-relative")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=0)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(...,
                                alias="native-country",
                                example="United-States")
    salary: Optional[str]

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from inference API"}

@app.post("/inference")
def model_inference(data: Data):


    try:
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("encoders/encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
    except FileNotFoundError as e:
        print("Unable to find model/encoder in path")
        raise e

    request_body = data.dict(by_alias=True)
    request_body = pd.DataFrame(request_dict)











if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)