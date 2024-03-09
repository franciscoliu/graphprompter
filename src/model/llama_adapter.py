import json

import torch


from src.model.llama import ModelArgs, Tokenizer, Transformer

ignore_index = 0


class LlamaAdapter(torch.nn.Module):
    def __init__(
            self,
            graph,
            graph_type,
            args,
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.temperature = 0.8
        self.top_p = 0.95

        print('Loading LLAMA')
        llama_model_path = '[Your LLM PATH]'
        model_name = "7B"

        checkpoint = torch.load(llama_model_path + model_name + "/consolidated.00.pth", map_location="cpu")
        print(llama_model_path + model_name + "/consolidated.00.pth")
        with open(llama_model_path + model_name + "/params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=args.max_txt_len,
            max_batch_size=args.batch_size,
            adapter_len=args.adapter_len,
            adapter_layer=args.adapter_layer,
            **params
        )
        tokenizer = Tokenizer(model_path=llama_model_path + "/tokenizer.model")
        model_args.vocab_size = tokenizer.n_words
        # torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model_llama_adapter = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)
        model_llama_adapter.load_state_dict(checkpoint, strict=False)

        for name, param in model_llama_adapter.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                param.data = param.data.float()

        for name, param in model_llama_adapter.layers[-1 * args.adapter_layer:].named_parameters():
            if "gate" in name or "adapter" in name:
                param.data = param.data.float()
                param.requires_grad = True

        self.tokenizer = tokenizer
        self.tokenizer.pad_id = 0
        self.tokenizer.padding_side = 'left'
        self.model = model_llama_adapter

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, samples):
        # encode description, questions and labels
        questions = [self.tokenizer.encode(q, bos=False, eos=False) for q in samples["question"]]
        desriptions = [self.tokenizer.encode(d, bos=False, eos=False) for d in samples["desc"]]
        labels = [self.tokenizer.encode(l, bos=False, eos=False) for l in samples["label"]]

        # encode special tokens
        pad_embeds = self.model.tok_embeddings(torch.tensor(self.tokenizer.pad_id).to(self.device)).unsqueeze(0)
        bos_embeds = self.model.tok_embeddings(torch.tensor(self.tokenizer.bos_id).to(self.device)).unsqueeze(0)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels[i] + [self.tokenizer.eos_id]
            input_ids = desriptions[i][:self.max_txt_len] + questions[i] + label_input_ids
            inputs_embeds = self.model.tok_embeddings(torch.tensor(input_ids).to(self.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [ignore_index] * (inputs_embeds.shape[0]-len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [ignore_index] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.device)

        c_loss = self.model(
            inputs_embeds=inputs_embeds,
            # attention_mask=attention_mask,
            # return_dict=True,
            labels=label_input_ids,
        )

        return c_loss

    @torch.no_grad()
    def inference(self, samples):
        desc_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in samples['desc']]
        q_tokens = [self.tokenizer.encode(x, bos=False, eos=False) for x in samples['question']]
        prompt_tokens = [d[:self.max_txt_len] + q for d, q in zip(desc_tokens, q_tokens)]
        bsz = len(prompt_tokens)

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = self.max_new_tokens + max_prompt_size

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward_only(tokens[:, prev_pos:cur_pos], prev_pos)
            if self.temperature > 0:
                probs = torch.softmax(logits / self.temperature, dim=-1)
                next_token = sample_top_p(probs, self.top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + self.max_new_tokens]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
