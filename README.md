# GraphPromptTuning (Link Prediction)

## Environment setup
For environment setup and data preparation steps, please refer to the main branch [here](https://github.com/franciscoliu/graphprompter/tree/main)


## Training
Once you switch to the branch of ```link_prediction```, you can train the model by running the following command:

```bash
python train.py --dataset citeseer --model_name graph_llm --llm_model_name 7b --gnn_model_name gat 
--llm_frozen False --seed 0 --num_epochs 12 --lr 1e-4 --min_lr 5e-5 --link_prediction True 
```
Replace the ```--dataset``` and path to the llm checkpoints to the actual dataset and the path to the llm checkpoints. 
