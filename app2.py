from ast import List
from typing import Type
from fastapi import Form, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Form, Query, Path
from pydantic import BaseModel, Field, create_model
import pickle
from fastapi.encoders import jsonable_encoder
from typing import List, Dict
import mysql.connector
import ast

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    port=3307,
    password="password",
    database="mlops"
)
cursor = db.cursor()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

Model = ""

def get_model_info(given_tuple):
    # Extract input attributes and target
    input_attributes_str = given_tuple[2]
    target_str = given_tuple[3]

    # Parse input attributes string
    input_attributes_dict = ast.literal_eval(input_attributes_str)
    input_attributes = {attr: [details['type'], details['range']] for attr, details in input_attributes_dict.items()}

    # Parse target string
    target_dict = ast.literal_eval(target_str)
    target = {attr: [details['type']] for attr, details in target_dict.items()}

    # Create model_info dictionary
    model_info = {
        'input_attributes': input_attributes,
        'target_attributes': target,
        'path': given_tuple[5]  # Path remains the same
    }
    return model_info

model_global = None

@app.get("/predict_page/{model_name}", response_class=HTMLResponse)
async def predict_page(request: Request, model_name: str = Path(...)):
    cursor.execute("SELECT model_type, hyperparameters, input_attributes, target_attributes, results, model_path FROM model_registry WHERE model_name = %s", (model_name,))
    res = cursor.fetchone()
    model_info = get_model_info(res)
    Model = model_name
    global model_global
    model_global= model_info
    print(model_info, "\n", type(model_info))
    if model_info:
        attributes = model_info["input_attributes"]
        model_fields = {key: (attributes[key][0], ...) for key in attributes}
        model_ranges = {key: (attributes[key][1], ...) for key in attributes}
        print(model_ranges)
        input_model = create_model(f"{model_name}Input", **model_fields)
        return templates.TemplateResponse(
            "predict_model.html", {"request": request, "model_name": model_name, "input_model": input_model}
        )
    else:
        raise ValueError("Model not found")

@app.post("/predict", response_class=HTMLResponse)
async def predict_from_page(request: Request, model_name: str = Form(...)):
    model_info = model_global
    attributes = model_info["input_attributes"]
    target = model_info["target_attributes"].keys()
    da = await request.form()
    da = jsonable_encoder(da)
    input_parameters = {}
    for attr, details in attributes.items():
        attr_type, attr_range = details
        if attr_type == 'int':
            min_value, max_value = map(int, attr_range.split('-'))
            input_parameters[attr] = int(da[attr])
        elif attr_type == 'float':
            input_parameters[attr] = float(da[attr])

    model = pickle.load(open(model_info["path"], "rb"))
    model_fields = {key: (attributes[key][0], ...) for key in attributes}
    input_model = create_model(f"{model_name}Input", **model_fields)
    # Make predictions using the model
    prediction = model.predict([list(input_parameters.values())]).tolist()
    results = dict(zip(target, prediction))

    return templates.TemplateResponse(
        "predict_model.html", {"request": request, "prediction": results, "model_name": model_name, "attributes": model_info["input_attributes"], "input_model": input_model, "input_parameters": input_parameters}
    )

