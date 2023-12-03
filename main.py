from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import math
import pandas as pd

app = FastAPI()
model_file_path = 'model.pickle'
cols_file_path = 'cols.pickle'
scaler_file_path = 'scaler.pickle'

with open(model_file_path, 'rb') as file:
    loaded_model = pickle.load(file)

with open(cols_file_path, 'rb') as file:
    cols = pickle.load(file)

with open(scaler_file_path, 'rb') as f:
    scaler = pickle.load(f)

def parseFloat(value):
    if type(value) == str:
        splited_value = value.split(' ')
        new_value = splited_value[0]
        return math.nan if new_value == '' else float(splited_value[0])
    else:
        return value

categorical_features = ['seats', 'fuel', 'seller_type', 'transmission', 'owner']
numeric_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power']

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

def prepare_data(item: Item):
    data = {}
    categorical = []
    item_dict = item.dict()

    for key in categorical_features:
        val = item_dict.get(key)
        col_name = f'{key}_{val}'
        categorical.append(col_name)

    for col in cols:
        if col in numeric_features:
            val = item_dict.get(col)
            data[col] = parseFloat(val)
        elif col in categorical:
            data[col] = 1
        else:
            data[col] = 0

    return data

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = prepare_data(item)
    df = pd.DataFrame([data])
    df[numeric_features] = scaler.transform(df[numeric_features])
    pred = loaded_model.predict(df.values)

    return pred


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    preds = []
    for item in items:
        data = prepare_data(item)
        df = pd.DataFrame([data])
        df[numeric_features] = scaler.transform(df[numeric_features])
        pred = loaded_model.predict(df.values)
        preds.append(pred)

    return preds
