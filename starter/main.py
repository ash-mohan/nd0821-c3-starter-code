# Put the code for your API here.
from fastapi import FastAPI
from starter.ml.model import export_model_files, import_model_files, inference
from starter.ml.data import process_data
from pydantic import BaseModel, Field
from typing import Optional

import pickle
import uvicorn
import pandas as pd

class Data(BaseModel):
    age: int = Field(..., gt=0, example=28)
    workclass: str = Field(..., min_length=0, example="Private")
    fnlgt: int = Field(..., gt=0, example=185061)
    education: str = Field(..., min_length=0, example="11th")
    education_num: int = Field(..., gt=0, alias="education-num", example=7)
    marital_status: str = Field(...,
                                min_length=0,
                                alias="marital-status",
                                example="Never-married")
    occupation: str = Field(..., min_length=0, example="Handlers-cleaners")
    relationship: str = Field(..., min_length=0, example="Other-relative")
    race: str = Field(..., min_length=0, example="White")
    sex: str = Field(..., min_length=0, example="Male")
    capital_gain: int = Field(..., gt=0, alias="capital-gain", example=0)
    capital_loss: int = Field(..., gt=0, alias="capital-loss", example=0)
    hours_per_week: int = Field(..., gt=0, alias="hours-per-week", example=40)
    native_country: str = Field(...,
                                min_length=0,
                                alias="native-country",
                                example="United-States")
    salary: Optional[str]

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from inference API"}


@app.post("/inference/")
def model_inference(data: Data):


    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    model, encoder, lb = import_model_files(
        "models/model.pkl",
        "encoders/encoder.pkl",
        "lb/lb.pkl"
    )


    request_body = data.dict(by_alias=True)
    request_body = pd.DataFrame(request_body)


    X, _, _, _ = process_data(
        request_body,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    y_pred = model.predict(X)
    label = lb.inverse_transform(y_pred)

    return {"results": {"binary_class": y_pred, "class": label}}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)