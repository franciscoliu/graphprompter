import argparse


def parse_args_llama():
    parser = argparse.ArgumentParser(description="graph_llm")

    parser.add_argument("--model_name", type=str, default='graph_llm')
    parser.add_argument("--project", type=str, default="graph_prompt_tuning")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset", type=str, default='cora')
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--patience", type=float, default=2)
    parser.add_argument("--min_lr", type=float, default=8e-6)
    parser.add_argument("--resume", type=str, default='')

    # Model Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_steps", type=int, default=2) #2

    # Learning Rate Scheduler
    parser.add_argument("--num_epochs", type=int, default=12)  # 8
    parser.add_argument("--warmup_epochs", type=float, default=1)

    # Inference
    parser.add_argument("--eval_batch_size", type=int, default=8)

    # LLM related
    parser.add_argument("--llm_model_name", type=str, default='7b')
    parser.add_argument("--llm_model_path", type=str, default='')
    parser.add_argument("--llm_frozen", type=str, default='True')
    parser.add_argument("--llm_num_virtual_tokens", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--max_txt_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=32)

    # llm adapter
    parser.add_argument("--adapter_len", type=int, default=10)
    parser.add_argument("--adapter_layer", type=int, default=30)

    # distributed training parameters
    parser.add_argument("--log_dir", type=str, default='logs/')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--world_size", default=4, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--gpu", default='0,1,2,3', type=str)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--num_workers", default=8, type=int)

    # GNN related
    parser.add_argument("--gnn_model_name", type=str, default='gat')
    parser.add_argument("--gnn_num_layers", type=int, default=4)
    parser.add_argument("--gnn_in_dim", type=int, default=1024)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--gnn_num_heads", type=int, default=4)
    parser.add_argument("--gnn_dropout", type=float, default=0.0)


    # Link Prediction Task
    parser.add_argument("--link_prediction", type=bool, default=True)

    args = parser.parse_args()
    return args
