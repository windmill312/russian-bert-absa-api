import yaml
from flask import Flask, request, jsonify
from flask_cors import cross_origin

from predictor import Predictor
from tuner import Tuner

app = Flask(__name__)

predictor = None


@app.route('/predict', methods=['GET'])
@cross_origin()
def predict():
    request_data = request.args.get('text')
    if (request_data is not None) and (request_data != ''):
        return jsonify(predictor.predict(request_data)), 200
    else:
        return jsonify('Text param should exists'), 400


@app.route('/tune', methods=['POST'])
@cross_origin()
def tune():
    request_data = request.json
    if (request_data is not None) and (request_data != ''):
        tuner.tune(request_data)
        return '', 204
    else:
        return 'Marked data should exists', 400


if __name__ == '__main__':
    with open("config.yaml", 'r') as conf_file:
        cfg = yaml.load(conf_file, Loader=yaml.BaseLoader)

    host = cfg['app']['host']
    port = cfg['app']['port']
    is_debug = cfg['app']['debug']

    aspects = cfg['data']['aspects']
    sentiments = cfg['data']['sentiments']
    predictor = Predictor(aspects, sentiments)

    tuner_file = cfg['data']['tune-file']
    tuner = Tuner(tuner_file)

    app.run(host=host, port=port, debug=is_debug)
