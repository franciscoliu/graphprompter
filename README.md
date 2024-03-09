# GraphPromptTuning

## Environment setup
```
conda create --name gpt python=3.9 -y
conda activate gpt

# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"

conda install pyg -c pyg
pip install pandas
pip install ogb

pip install transformers
pip install peft
pip install wandb
pip install sentencepiece
```

## Preprocess
Download the raw texts of titles and abstracts [here](https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz), unzip and move it to `dataset/ogbn_arxiv_orig/`.
```
# preprocess
python -m src.dataset.preprocess.arxiv

# check
python -m src.dataset.arxiv
```

## Training
Replace path to the llm checkpoints in the `src/model/__init__.py`, then run
```
python train.py --dataset arxiv --model_name graph_llm --llm_model_name 7b --gnn_model_name gat --seed 0
```

## Reproduction
You may refer to ```run.sh``` for detailed commands for reproductions. The main branch is for node classification experiments and the link_prediction branch is for link prediction experiments. 
