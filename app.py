from flask import jsonify, Flask, request
from classifier import main as predict_classification
import ast

app = Flask(__name__)


@app.route("/classify-images", methods=["POST"])
def classify_image_list():
    '''Endpoint per la classificazione di una lista di immagini'''

    request_image_list = request.json

    for item in request_image_list:
        item_path   = item['filename']

        for bounding_box in item['bb_box']:
            bbox_coordinates = ast.literal_eval(bounding_box['box'])

            prediction  = predict_classification(item_path, bbox_coordinates)
            pred_id     = prediction['id']
            pred_label  = prediction['label']
            pred_conf   = prediction['confidence']

            bounding_box['classification']  = pred_id
            bounding_box['class_label']     = pred_label
            bounding_box['confidence']      = pred_conf
            bounding_box['model_name']      = prediction['model_name']

    return jsonify(request_image_list)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)