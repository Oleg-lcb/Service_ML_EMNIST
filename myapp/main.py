import numpy as np
import pandas as pd

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from myapp.model import Model

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# load model
model = Model()

# app
app = FastAPI(title='Symbol detection', docs_url='/docs')

# api
@app.post('/api/predict')
def predict(image: str = Body(..., description='image pixels list')):
    image = np.array(list(map(int, image[1:-1].split(','))))
    image = image.reshape(1,784)
    image_df = pd.DataFrame(data=image, index=None)
    pred = model.predict(image_df)

    with open('emnist-balanced-mapping.txt', 'r') as file_code:
        code_char = dict()
        for line in file_code:
            if int(line.strip('\n').split(' ')[0]) not in code_char.keys():
                code_char[int(line.strip('\n').split(' ')[0])] = chr(int(line.strip('\n').split(' ')[1]))

    return {'prediction': code_char[int(pred[1:-1])]}

# static files
app.mount('/', StaticFiles(directory='static', html=True), name='static')
