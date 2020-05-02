import json
import os
from json import loads

import yaml
from flask import request
from flask_restful import Resource


def prepare_marked_data(marked_data):
    aspects = marked_data['aspects']

    categories = []
    for aspect in aspects:
        categories.append(aspect['category'])

    polarities = []
    for aspect in aspects:
        polarities.append(aspect['sentiment'])

    return {
        "text": marked_data['sentence'],
        "category": categories,
        "polarity": polarities
    }


class Tune(Resource):
    filename = ''

    def __init__(self):
        with open("config.yaml", 'r') as conf_file:
            cfg = yaml.load(conf_file, Loader=yaml.BaseLoader)
        self.filename = cfg['data']['tune-file']

        annotation = {"annotation": []}
        if os.path.exists(self.filename):
            if os.path.getsize(self.filename) == 0:
                with open(self.filename, 'w') as outfile:
                    json.dump(annotation, outfile, ensure_ascii=False)
        else:
            with open(self.filename, 'w+') as outfile:
                json.dump(annotation, outfile, ensure_ascii=False)

    def post(self):
        data = loads(request.data)
        if (data is not None) and (data != ''):
            marked_data = prepare_marked_data(data)

            with open(self.filename, "r+") as file:
                file_data = json.load(file)
                file_data['annotation'].append(marked_data)
                file.seek(0)
                json.dump(file_data, file, ensure_ascii=False)

            return '', 204
        else:
            return 'Marked data should exists', 400
