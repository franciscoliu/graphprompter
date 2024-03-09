import contextlib
import torch
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ignore_index = -100


class T5(torch.nn.Module):

    def __init__(
        self,
        graph,
        graph_type,
        args,
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading FLAN T5')
        kwargs = {
            "max_memory": {0: '20GiB', 1: '20GiB', 2: '20GiB', 3: '20GiB'},
            "device_map": "auto",
            "revision": "main",
        }

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-xl",
            # torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-xl",
            use_fast=False,
            revision=kwargs["revision"]
        )
        self.tokenizer.pad_token_id = 0
        self.tokenizer.bos_token_id = 0
        self.tokenizer.eos_token_id = 1
        self.tokenizer.padding_side = 'left'

        print('Finish loading FLAN T5!')

        self.word_embedding = self.model.shared

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

        # encode special tokens
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)
        bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)

        # encode desc
        model_inputs = self.tokenizer(samples["desc"], add_special_tokens=False)
        questions = self.tokenizer(samples['question'], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = model_inputs["input_ids"][i][:self.max_txt_len] + questions["input_ids"][i]
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.eos_token_id]

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            
        max_length = max([len(x) for x in batch_label_input_ids])
        for i in range(batch_size):
            pad_length = max_length - len(batch_label_input_ids[i])
            batch_label_input_ids[i] = [ignore_index] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples):

        # encode special tokens
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)
        bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)

        # encode desc
        model_inputs = self.tokenizer(samples["desc"], add_special_tokens=False)
        questions = self.tokenizer(samples['question'], add_special_tokens=False)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = model_inputs["input_ids"][i][:self.max_txt_len] + questions["input_ids"][i]
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': [p.strip() for p in pred],
                'label': samples['label'],
                'desc': samples['desc'],
                'question': samples['question']}

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
