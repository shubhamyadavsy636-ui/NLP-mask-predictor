import streamlit as st
import torch
from transformers import BertTokenizer, BertForMaskedLM

MODEL_PATH = r"./results/checkpoint-1113"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertForMaskedLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.eval()

st.title("Mini BERT Masked Word Predictor")

text = st.text_input("Enter sentence with [MASK]")

if text:
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    
    mask_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    
    top_k = 5 
    
    probs = logits[0, mask_index].softmax(dim=-1)
    top_k_ids = probs.topk(top_k, dim=-1).indices[0]
    
    predicted_word = [tokenizer.decode([idx]) for idx in top_k_ids] 
    
    st.write("Top prediction : ", predicted_word)