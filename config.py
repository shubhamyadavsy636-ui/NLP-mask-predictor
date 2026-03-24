MODEL_NAME = "mini-bert-fast"

MAX_LENGTH = 64            # 🔥 biggest speed boost
GRAD_ACCUM = 2            # reduce delay in updates
BATCH_SIZE = 16           # 3050 can handle this at 64 tokens

EPOCH = 3                 # enough for wikitext-2
LR = 3e-5                 # slightly higher for faster convergence

HIDDEN_SIZE = 256         # lighter model
NUM_LAYERS = 6            # reduced from 8
NUM_HEAD = 4              # must divide hidden size

MLM_PROB = 0.15

FP16 = True               # MUST for speed on 3050
NUM_WORKERS = 6           # Ryzen 5 can handle more threads

LOGS_STEPS = 200
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "epoch"