from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from load_dataset import load_data
from model import build_model
import config 

dataset, tokenizer = load_data()

model = build_model()

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir = "./results",
    
    per_device_eval_batch_size=config.BATCH_SIZE,
    per_device_train_batch_size=config.BATCH_SIZE,
    
    num_train_epochs=config.EPOCH,
    learning_rate=config.LR,
    
    evaluation_strategy="epoch",
    save_strategy="epoch",
    
    logging_dir="./logs",
    
    fp16=True,
    
    gradient_accumulation_steps=4,
    
    max_grad_norm=1.0
       
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=data_collator
)


trainer.train()