from flask import Flask
from flask_cors import CORS
from flask_restful import Api

from api.resources import Predict, Tune


def create_app():
    app = Flask(__name__)

    cors = CORS(app, resources={r"*": {"origins": "*"}})

    api = Api(app)
    api.add_resource(Predict, "/predict")
    api.add_resource(Tune, "/tune")

    return app
