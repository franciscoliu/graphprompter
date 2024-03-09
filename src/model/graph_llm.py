import contextlib
import torch
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)


from src.model.gnn import load_gnn_model


ignore_index = -100


class GraphLLM(torch.nn.Module):

    def __init__(
        self,
        graph,
        graph_type,
        prompt,
        args,
    ):
        super().__init__()
        self.graph = graph
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')
        kwargs = {
            # "max_memory": {0: '20GiB', 1: '20GiB', 2: '20GiB', 3: '20GiB'},
            # "max_memory": {0: '40GiB', 1: '40GiB', 2: '40GiB'},
            # "max_memory": {0: '80GiB'},
            "max_memory": {0: '80GiB', 1: '80GiB'},
            "device_map": "auto",
            "revision": "main",
        }

        # self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_int8_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        print('Finish loading LLAMA!')

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=graph.x.shape[-1],
            out_channels=2048,   #4096
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        self.word_embedding = self.model.model.get_input_embeddings()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, samples):
        x_1 = samples['node1_features'].to(self.model.device)
        x_2 = samples['node2_features'].to(self.model.device)
        edge_index_1 = samples['edge_index_1'].to(self.model.device)
        edge_index_2 = samples['edge_index_2'].to(self.model.device)
        mapping_1 = samples['mapping_1'].to(self.model.device)
        mapping_2 = samples['mapping_1'].to(self.model.device)

        node1_embeds, node2_embeds, _ = self.graph_encoder(x_1, edge_index_1, x_2, edge_index_2)
        # n_embeds, _ = self.graph_encoder(node1_features, edge_index)

        inputs_embeds_1 = node1_embeds[mapping_1]
        inputs_embeds_2 = node2_embeds[mapping_2]

        # inputs_embeds = torch.cat([inputs_embeds_1, inputs_embeds_2], dim=1)
        inputs_embeds = torch.cat([inputs_embeds_1, inputs_embeds_2], dim=-1).unsqueeze(1)  # Add sequence length dimension

        return inputs_embeds

    def forward(self, samples):

        device = self.device  # Assuming self.device is correctly set to 'cuda' or 'cpu'
        graph_embeds = self.encode_graphs(samples).to(device)

        # print("graph_embeds shape: ", graph_embeds.shape)

        # Labels indicating whether a link exists between node pairs
        labels = samples['label'].to(device).unsqueeze(1)
        attention_mask = torch.any(graph_embeds != 0, axis=-1).to(device)

        # print("labels shape: ", labels.shape)
        #
        # print("Check graph_embeds for NaN:", torch.isnan(graph_embeds).any())
        # print("Check labels for NaN:", torch.isnan(labels).any())

        # Forward pass through the model
        with self.maybe_autocast():
            # outputs = self.model(inputs_embeds=graph_embeds, labels = labels)
            outputs = self.model(
                inputs_embeds=graph_embeds,  # Using graph embeddings directly
                attention_mask=attention_mask,  # include the attention mask here
                labels=labels,
                return_dict=True
            )


        # print("Model output before loss calculation:", outputs.logits)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        return loss




    def inference(self, samples):

        graph_embeds = self.encode_graphs(samples).to(self.device)

        # Prepare attention masks (1 where there's data in graph_embeds, 0 otherwise)
        attention_mask = torch.any(graph_embeds != 0, axis=-1).to(self.device)

        # Forward pass through the model (in evaluation mode)
        with torch.no_grad(), self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=graph_embeds,
                attention_mask=attention_mask,
                use_cache=True
            )

        # Convert logits to probabilities for link prediction
        pred_logits = outputs.logits
        pred_prob = torch.sigmoid(pred_logits).squeeze(-1)

        return {'node1': samples['node1'],
                'node2': samples['node2'],
                'label': samples['label'],
                'pred_prob': pred_prob.cpu().numpy(),
                },


    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
