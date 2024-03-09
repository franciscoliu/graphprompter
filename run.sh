1. Dense w/ instructions
git checkout main

# cora, ours+finetuning w/ Lora
python train.py --dataset cora --model_name graph_llm --llm_frozen False

# pubmed, ours+finetuning w/ Lora
python train.py --dataset pubmed --model_name graph_llm --llm_frozen False

# arxiv, finetuning w/ Lora
python train.py --dataset arxiv --model_name llm --llm_frozen False

# arxiv, ours+finetuning w/ Lora
python train.py --dataset arxiv --model_name graph_llm --llm_frozen False

2. Dense w/o instructions
git checkout wo_question


# pubmed, ours+finetuning w/ Lora
python train.py --dataset pubmed --model_name graph_llm --llm_frozen False

# arxiv
# soft prompt tuning
python train.py --dataset arxiv --model_name pt_llm
# ours
python train.py --dataset arxiv --model_name graph_llm
# finetuning w/ Lora
python train.py --dataset arxiv --model_name llm --llm_frozen False
# ours+finetuning w/ Lora
python train.py --dataset arxiv --model_name graph_llm --llm_frozen False

# products
# soft prompt tuning
python train.py --dataset products --model_name pt_llm
# ours
python train.py --dataset products --model_name graph_llm
# finetuning w/ Lora
python train.py --dataset products --model_name llm --llm_frozen False
# ours+finetuning w/ Lora
python train.py --dataset products --model_name graph_llm --llm_frozen False



3. Sparse semantics
git checkout sparse_semantics

# arxiv
# subgraph prompt tuning
python train.py --dataset arxiv --model_name graph_llm
# ours+Finetuning w/ Lora
python train.py --dataset arxiv --model_name graph_llm --llm_frozen False

# products
# subgraph prompt tuning
python train.py --dataset products --model_name graph_llm
# ours+Finetuning w/ Lora
python train.py --dataset products --model_name graph_llm --llm_frozen False
