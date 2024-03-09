python inference.py --dataset products --model_name inference_llm --link_prediction True --seed 0
python train.py --dataset products --model_name graph_llm --link_prediction True --seed 0
python train.py --dataset products --model_name graph_llm --llm_frozen False --link_prediction True --seed 0
python train.py --dataset products --model_name pt_llm --link_prediction True --seed 0
python train.py --dataset products --model_name llm --llm_frozen False --link_prediction True --seed 0

