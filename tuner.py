import json
import os


class Tuner:
    filename = ''

    def __init__(self, filename):
        self.filename = filename

        annotation = {"annotation": []}
        if os.path.exists(filename):
            if os.path.getsize(filename) == 0:
                with open(filename, 'w') as outfile:
                    json.dump(annotation, outfile)
        else:
            with open(filename, 'w+') as outfile:
                json.dump(annotation, outfile)


    def prepare_marked_data(self, marked_data):
        aspects = marked_data['aspects']

        categories = []
        for aspect in aspects:
            categories.append(aspect['category'])

        polarities = []
        for aspect in aspects:
            polarities.append(aspect['sentiment'])

        return {
            "text": marked_data['sentence'],
            "category": [categories],
            "polarity": [polarities]
        }

    def tune(self, marked_data):
        marked_data = self.prepare_marked_data(marked_data)

        with open(self.filename, "r+") as file:
            file_data = json.load(file)
            file_data['annotation'].append(marked_data)
            file.seek(0)
            json.dump(file_data, file)
