from datasets import load_dataset 
from transformers import AutoTokenizer
import config

def load_data():
    dataset = load_dataset('wikitext', 'wikitext-2-v1')
    dataset = dataset.filter(lambda x : x['text'] is not None and  len(x['text'].strip()) > 0)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  
    def tokenize(ex):
        return tokenizer(  
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=config.MAX_LENGTH
        ) 
        
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    return tokenized_dataset, tokenizer 
         

