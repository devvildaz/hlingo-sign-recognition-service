import tensorflow as tf

MODEL_PATH = 'model/model.h5'

labels = (
    'good_morning',
    'good_afternoon',
    'good_evening',
    'hi',
    'bye',
    'see_you_tomorrow',
    'see_you_later',
    'family',
    'mom',
    'dad',
    'son',
    'sibling',
    'cousin',
    'person',
    'neighbour',
    'kid',
    'youngster',
    'adult',
    'elder',
    'baby'
)

def apply_model_on_video(coordinates):
    predictions = (tf.keras
                    .models
                    .load_model(MODEL_PATH)
                    .predict(coordinates)
                    .tolist())
    
    return create_response(predictions)

def format_predictions(model_predictions):
    formatted_predictions = []

    for model_prediction in model_predictions:
        formatted_predictions.append(
            {label: prediction for (label, prediction) in zip(labels, model_prediction)}
        )

    return formatted_predictions

def create_response(predictions):
    response = {}
    response['predictions'] = format_predictions(predictions)
    return response
    