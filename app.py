import yaml
from flask import Flask, request, jsonify
from flask_cors import cross_origin

from predictor import Predictor

app = Flask(__name__)

predictor = None


@app.route('/predict')
@cross_origin()
def predict():
    return jsonify(predictor.predict(request.args.get('text')))


if __name__ == '__main__':
    with open("config.yaml", 'r') as conf_file:
        cfg = yaml.load(conf_file, Loader=yaml.BaseLoader)

    host = cfg['app']['host']
    port = cfg['app']['port']
    is_debug = cfg['app']['debug']

    aspects = cfg['data']['aspects']
    sentiments = cfg['data']['sentiments']
    predictor = Predictor(aspects, sentiments)

    app.run(host=host, port=port, debug=is_debug)
