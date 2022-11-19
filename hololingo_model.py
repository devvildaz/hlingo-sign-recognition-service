import tensorflow as tf
import json
import os
from dotenv import load_dotenv

load_dotenv()

class HoloLingoModel():

    def __init__(self):
        self._MODEL_PATH = os.environ['MODEL_PATH']
        self._model = (tf.keras
                    .models
                    .load_model(self._MODEL_PATH))
        self._LABELS = self._load_labels_json(os.environ['LABELS_PATH'])

    def apply_model_on_coordinates(self, coordinates):
        predictions = (self._model
                            .predict(coordinates)
                            .tolist())
        
        return self._create_response(predictions)

    def _create_response(self, predictions):
        response = {}
        response['predictions'] = self._format_predictions(predictions)
        return response

    def _format_predictions(self, model_predictions):
        formatted_predictions = []

        for model_prediction in model_predictions:
            formatted_predictions.append(
                {label: prediction for (label, prediction) in zip(self._LABELS, model_prediction)}
            )

        return formatted_predictions
    
    def _load_labels_json(self, labels_path):
        f = open(labels_path)
        labels_json = json.load(f)

        return [labels_json[key] for key in labels_json.keys()]