import math
import contextlib
import torch
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.gnn import load_gnn_model

ignore_index = -100


class PromptTuningLLM(torch.nn.Module):

    def __init__(
        self,
        graph,
        graph_type,
        prompt,
        args,
        # **kwargs,
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.instruction = prompt
        self.graph = graph

        num_virtual_tokens = args.llm_num_virtual_tokens
        print('Loading LLAMA')
        kwargs = {
            # "max_memory": {0: '20GiB', 1: '20GiB', 2: '20GiB', 3: '20GiB'},
            "max_memory": {0: '80GiB', 1: '80GiB'},
            "device_map": "auto",
            "revision": "main",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )
        # freeze all parameters except prompt embeddings
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        print('Finish loading LLAMA!')

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=graph.x.shape[-1],
            out_channels=2048,  # 4096
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        init_token_ids = self.tokenizer(self.instruction).input_ids
        num_text_tokens = len(init_token_ids)
        if num_text_tokens < num_virtual_tokens:
            num_reps = math.ceil(num_virtual_tokens / num_text_tokens)
            init_token_ids = init_token_ids * num_reps
        init_token_ids = init_token_ids[:num_virtual_tokens]
        word_embeddings = self.model.get_input_embeddings().weight
        self.prompt = torch.nn.Parameter(word_embeddings[torch.LongTensor(init_token_ids)].detach().clone().to(torch.float32)).to(self.model.device)

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
        inputs_embeds = torch.cat([inputs_embeds_1, inputs_embeds_2], dim=-1).unsqueeze(
            1)  # Add sequence length dimension

        return inputs_embeds

        # n_embeds, _ = self.graph_encoder(self.graph.x, self.graph.edge_index)
        # inputs_embeds = n_embeds[ids]
        # return inputs_embeds



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
        # pred_prob = torch.sigmoid(outputs).squeeze(-1)

        return {'node1': samples['node1'],
                'node2': samples['node2'],
                'label': samples['label'],
                'pred_prob': pred_prob.cpu().numpy(),
                },

        # encode special tokens
        # pad_embeds = self.model.model.embed_tokens(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)
        # bos_embeds = self.model.model.embed_tokens(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)
        #
        # # encode desc
        # model_inputs = self.tokenizer(samples["desc"], add_special_tokens=False)
        #
        # batch_size = len(samples['id'])
        # batch_inputs_embeds = []
        # batch_attention_mask = []
        #
        # prompt_embeds = self.prompt.repeat(batch_size, 1, 1)
        #
        # for i in range(batch_size):
        #     # Add bos & eos token
        #     input_ids = model_inputs["input_ids"][i][:self.max_txt_len]
        #     inputs_embeds = self.model.model.embed_tokens(torch.tensor(input_ids).to(self.model.device))
        #     inputs_embeds = torch.cat([bos_embeds, prompt_embeds[i], inputs_embeds], dim=0)
        #     batch_inputs_embeds.append(inputs_embeds)
        #     batch_attention_mask.append([1] * inputs_embeds.shape[0])
        #
        # # pad inputs_embeds
        # max_length = max([x.shape[0] for x in batch_inputs_embeds])
        # for i in range(batch_size):
        #     pad_length = max_length-batch_inputs_embeds[i].shape[0]
        #     batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
        #     batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
        #
        # inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        # attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        #
        # with self.maybe_autocast():
        #     outputs = self.model.generate(
        #         inputs_embeds=inputs_embeds,
        #         max_new_tokens=self.max_new_tokens,
        #         attention_mask=attention_mask,
        #         # do_sample=True,
        #         use_cache=True  # IMPORTANT!
        #     )
        # pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #
        # return {'id': samples['id'],
        #         'pred': pred,
        #         'label': samples['label'],
        #         'desc': samples['desc']}

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
