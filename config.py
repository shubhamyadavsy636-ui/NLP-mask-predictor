MODEL_NAME = "mini-bert-fast"

MAX_LENGTH = 64            
GRAD_ACCUM = 2            
BATCH_SIZE = 16           

EPOCH = 3                 
LR = 3e-5                 

HIDDEN_SIZE = 256         
NUM_LAYERS = 6            
NUM_HEAD = 4              

MLM_PROB = 0.15

FP16 = True               
NUM_WORKERS = 6           

LOGS_STEPS = 200
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "epoch"
