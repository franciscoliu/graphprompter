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


ignore_index = -100


class LLM(torch.nn.Module):

    def __init__(
        self,
        graph,
        graph_type,
        prompt,
        args,
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        lora_r: int = 8
        lora_alpha: int = 16
        lora_dropout: float = 0.05
        lora_target_modules = [
            "q_proj",
            "v_proj",
        ]

        print('Loading LLAMA')
        kwargs = {
            # "max_memory": {0: '20GiB', 1: '20GiB', 2: '20GiB', 3: '20GiB'},
            "max_memory": {0: '80GiB'},
            "device_map": "auto",
            "revision": "main",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
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
            model = prepare_model_for_int8_training(model)

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

    def forward(self, samples):
        device = self.device  # Assuming self.device is correctly set to 'cuda' or 'cpu'
        input_texts = [f'Node1: {node1} Node2: {node2}' for node1, node2 in zip(samples['node1'], samples['node2'])]
        model_inputs = self.tokenizer(input_texts, padding=True, truncation=True, max_length=self.max_txt_len,
                                      return_tensors='pt').to(self.device)
        labels = samples['label'].to(device).unsqueeze(1)

        with self.maybe_autocast():
            outputs = self.model(**model_inputs, labels=model_inputs['input_ids'])
            loss = outputs.loss

        return loss
    def inference(self, samples):

        input_texts = [f'Node1: {node1} Node2: {node2}' for node1, node2 in zip(samples['node1'], samples['node2'])]
        model_inputs = self.tokenizer(input_texts, padding=True, truncation=True, max_length=self.max_txt_len,
                                      return_tensors='pt').to(self.device)

        # Forward pass through the model (in evaluation mode)
        with torch.no_grad(), self.maybe_autocast():
            outputs = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)

        # pred_logits = outputs.logits
        # pred_prob = torch.sigmoid(pred_logits).squeeze(-1)

        pred_prob = torch.sigmoid(outputs).squeeze(-1)

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
