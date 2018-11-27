from flask import Flask,jsonify,request
from flasgger import Swagger
from sklearn.externals import joblib
import numpy as np
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
Swagger(app)
CORS(app)

@app.route('/input/task', methods=['POST'])
def predict():
    """
    Ini Adalah Endpoint Untuk Mengevaluasi Sentimen
    ---
    tags:
        - Rest Controller
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: Sentence
          required:
            - textSentence
          properties:
            textSentence:
              type: string
              description: Please input with valid text.
              default: 0
    responses:
        200:
            description: Success Input
    """
    new_task = request.get_json()

    textSentence = new_task['textSentence']

    X_New = np.array([textSentence])

    pipe = joblib.load('SGDClassifierSentence.pkl')

    resultPredict = pipe[0].predict(X_New)

    return jsonify({'message': format(resultPredict)})

if __name__ == '__main__' :
    app.run(debug=True)