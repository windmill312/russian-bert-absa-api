import yaml
from flask_restful import Resource, reqparse
import torch
import torch.nn.functional as F

from notebooks import tokenization
from notebooks.modeling import BertConfig, BertForSequenceMultiClassification


def cut_if_necessary(tokens):
    if len(tokens) > 512 - 2:
        tokens = tokens[0:(512 - 2)]
    return tokens


def mark_begin_end(tokens):
    result = ["[CLS]"]
    for token in tokens:
        result.append(token)
    result.append("[SEP]")
    return result


def get_logits(input_ids, sentence_level_model):
    input_mask = [1] * len(input_ids)

    while len(input_ids) < 512:
        input_ids.append(0)
        input_mask.append(0)

    input_ids = torch.LongTensor(input_ids).unsqueeze(0)
    input_mask = torch.LongTensor(input_mask).unsqueeze(0)
    return sentence_level_model(input_ids, None, input_mask)


class Predict(Resource):
    aspects = []
    sentiments = []
    tokenizer = None

    sentence_level_model = None
    vocab_file = '/app/notebooks/vocab.txt'
    weights_file = '/app/notebooks/final_weights.pth'

    def __init__(self):
        with open("/app/config.yaml", 'r') as conf_file:
            cfg = yaml.load(conf_file, Loader=yaml.BaseLoader)

        self.aspects = cfg['data']['aspects']
        self.sentiments = cfg['data']['sentiments']

        self.get_parser = reqparse.RequestParser()
        self.get_parser.add_argument("text", type=str, required=True, location="args")

        # torch.hub.load_state_dict_from_url('https://drive.google.com/open?id=1VgkYMKE64UeCkLURafae4Z0UaYtWBIZq')
        # weights = torch.hub.load_state_dict_from_url(
        # 'https://drive.google.com/uc?export=download&confirm=qTcY&id=1SeeRVlb_uMcZYKD4r8X_dqKDBTyDastF')
        weights = torch.load(self.weights_file, map_location='cpu')
        new_weights = {name[7:]: weights[name] for name in weights}

        bert_config = BertConfig.from_json_file('notebooks/bert_config.json')
        self.sentence_level_model = BertForSequenceMultiClassification(bert_config, len(self.aspects))
        self.sentence_level_model.load_state_dict(new_weights)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=False)

    def get(self):
        args = self.get_parser.parse_args()

        self.sentence_level_model.eval()

        tokens = cut_if_necessary(self.tokenizer.tokenize(args["text"]))
        marked_tokens = mark_begin_end(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(marked_tokens)
        logits = get_logits(input_ids, self.sentence_level_model)

        result = []
        for i, ex in enumerate(logits):
            ex = F.softmax(ex, dim=-1)
            idx = ex.argmax()
            if idx > 0:
                result.append({'aspect': self.aspects[i], 'sentiment': self.sentiments[idx - 1]})
        return result, 200
