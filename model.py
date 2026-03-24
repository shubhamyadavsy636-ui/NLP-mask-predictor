from transformers import BertForMaskedLM
import config

def build_model():
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    return model

    