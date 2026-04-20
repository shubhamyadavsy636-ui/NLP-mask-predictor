import streamlit as st
import torch
from transformers import BertTokenizer, BertForMaskedLM

# Force CPU (Streamlit Cloud safe)
device = torch.device("cpu")

@st.cache_resource
def load_model():
    try:
        model = BertForMaskedLM.from_pretrained("shubham-ml636/mini-bert-mask")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, tokenizer = load_model()

st.title("Mini BERT Masked Word Predictor")

# Sidebar controls (NEW)
top_k = st.sidebar.slider("Top K Predictions", 1, 10, 5)

text = st.text_input("Enter sentence with [MASK]")
st.write(' Ex -> The capital of India is [MASK].')

if text:
    if model is None:
        st.error("Model not loaded properly.")
    
    elif "[MASK]" not in text:
        st.warning("⚠️ Please include [MASK] token in your sentence.")
    
    else:
        try:
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits

            mask_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

            probs = logits[0, mask_index].softmax(dim=-1)

            top_k_ids = probs.topk(top_k, dim=-1).indices[0]

            predicted_words = [tokenizer.decode([idx]).strip() for idx in top_k_ids]

            st.subheader("Predictions")
            for word in predicted_words:
                st.write(text.replace("[MASK]", word))

        except Exception as e:
            st.error(f"Runtime error: {e}")