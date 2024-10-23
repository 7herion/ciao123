from flask import jsonify, Flask, request
from classifier import main as predict_classification
import ast

app = Flask(__name__)


@app.route("/classify-images", methods=["POST"])
def classify_image_list():
    '''Endpoint per la classificazione di una lista di immagini'''

    request_image_list = request.json
    result_list: list = []

    for item in request_image_list:
        item_id     = item['id']
        item_path   = item['filename'] # dovrei avere il path completo già nel payload

        item_bounding_box_list: list = []
        for bounding_box in item['bb_box']:
            bbox_coordinates = ast.literal_eval(bounding_box['box']) # mi arrivano in formato stringa come salvati su db

            prediction  = predict_classification(item_path, bbox_coordinates)
            pred_class  = prediction['classification']
            pred_conf   = prediction['confidence']

            item_bounding_box_list.append({
                'id': bounding_box['id'],
			    'box': bbox_coordinates, # TODO: da decidere se ritornare come list oppure come stringa
			    'category': bounding_box['category'],
			    'classification': pred_class, # TODO: da matchare con le classi esistenti su DB
			    'confidence': pred_conf,
            })

        # BASEPATH è ./data01/wildlife_media/

        result_list.append({
            'id':       item_id,
            'bb_box':   item_bounding_box_list,
            'filename': item_path,
        })

    return jsonify(result_list)


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)