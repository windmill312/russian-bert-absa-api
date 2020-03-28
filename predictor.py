import torch
import torch.nn.functional as F

from notebooks import tokenization
from notebooks.modeling import BertConfig, BertForSequenceMultiClassification


class Predictor:
    aspects = []
    sentiments = []
    sentence_level_model = None
    model_filename = 'lib/models/pre-final.sav'
    vocab_file = 'notebooks/vocab.txt'

    def __init__(self, aspects, sentiments):
        self.aspects = aspects
        self.sentiments = sentiments

        # torch.hub.load_state_dict_from_url('https://drive.google.com/open?id=1VgkYMKE64UeCkLURafae4Z0UaYtWBIZq')
        # weights = torch.hub.load_state_dict_from_url('https://drive.google.com/uc?export=download&confirm=qTcY&id=1SeeRVlb_uMcZYKD4r8X_dqKDBTyDastF')
        weights = torch.load('notebooks/final_weights.pth', map_location='cpu')
        new_weights = {name[7:]: weights[name] for name in weights}

        bert_config = BertConfig.from_json_file('notebooks/bert_config.json')
        self.sentence_level_model = BertForSequenceMultiClassification(bert_config, len(self.aspects))
        self.sentence_level_model.load_state_dict(new_weights)

    def predict(self, text):
        tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=False)

        # TODO add model selecting by sentence number
        self.sentence_level_model.eval()
        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a) > 512 - 2:
            tokens_a = tokens_a[0:(512 - 2)]

        tokens = ["[CLS]"]
        for token in tokens_a:
            tokens.append(token)
        tokens.append("[SEP]")

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < 512:
            input_ids.append(0)
            input_mask.append(0)
        input_ids = torch.LongTensor(input_ids).unsqueeze(0)
        input_mask = torch.LongTensor(input_mask).unsqueeze(0)
        logits = self.sentence_level_model(input_ids, None, input_mask)
        result = []
        for i, ex in enumerate(logits):
            ex = F.softmax(ex, dim=-1)
            idx = ex.argmax()
            if idx > 0:
                result.append({'aspect': self.aspects[i], 'sentiment': self.sentiments[idx - 1]})
        return result
