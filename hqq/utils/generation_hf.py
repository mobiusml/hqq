# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#####################################################

# Rewrite generate() to support torch.compile fullgraph.
# https://gist.github.com/ArthurZucker/5dc54a3fb443e979fac437e5df7c800b

import torch
from typing import Union
from transformers import StaticCache
from tqdm import tqdm


class HFGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 1000,
        cache_size: Union[int,None] = None,
        do_sample: bool = False,
        temperature: float = 0.6,
        top_k: int = 5,
        compile: Union[str,None] = None, #None / "partial" / "full"
    ):
        super().__init__()

        torch._dynamo.config.cache_size_limit = 32
        torch._dynamo.config.capture_scalar_outputs = True

        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.do_sample = do_sample
        self.temperature = temperature if self.do_sample else None
        self.top_k = top_k if self.do_sample else None
        self.use_cache = False

        if do_sample:
            decode_one_token = self.decode_one_token_sampled
        else:
            decode_one_token = self.decode_one_token_no_sample

        # Setup cache
        self.max_new_tokens = max_new_tokens
        if cache_size is None:
            self.cache_size = self.next_multiple(self.max_new_tokens)
        else:
            self.cache_size = cache_size

        self.max_new_tokens = min(self.max_new_tokens, self.cache_size)

        self.setup_cache()

        self.is_compiled = False
        if compile == "partial":
            self.compile_partial(decode_one_token)

        if compile == "full":
            self.compile_full()

        if(hasattr(self, 'decode_one_token') is False):
            self.decode_one_token = decode_one_token

        self.init()

    @torch.no_grad()
    def setup_cache(self):
        self.model._setup_cache(StaticCache, 1, max_cache_len=self.cache_size)

    # Ideally only compile this, but it creates issues with generation: https://github.com/huggingface/transformers/issues/30351
    def compile_partial(self, decode_one_token):
        self.decode_one_token = torch.compile(
            decode_one_token, **{"mode": "reduce-overhead", "fullgraph": True}
        )
        self.is_compiled = True

    @torch.inference_mode()
    def compile_full(self):
        self.model.forward = torch.compile(
            self.model.forward, mode="reduce-overhead", fullgraph=True
        )

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=False, enable_math=True
        ):
            for ctx_size in [1] * 10:
                self.model(
                    torch.ones(
                        (1, ctx_size), dtype=torch.int32, device=self.model.device
                    ),
                    use_cache=False,
                ).logits

        self.is_compiled = True

    def next_multiple(self, val):  # next power of 2
        vals = [
            2**i for i in range(5, 14)
        ]  # [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        new_val = vals[[i for i in range(len(vals)) if (vals[i] - val) > 0][0]]
        return new_val

    def init(self):
        # Setup inference mode
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_eos_token = False
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({"pad_token": "<<[PAD]>>"})
        self.tokenizer.padding_side = "right"
        self.model.eval()
        self.model.config.use_cache = True

    # Copied from https://gist.github.com/ArthurZucker/af34221def212259b43d55a2811d2dbb
    def multinomial_sample_one_no_sync(
        self, probs_sort
    ):  # Does multinomial sampling without a cuda synchronization
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

    # Copied from https://gist.github.com/ArthurZucker/af34221def212259b43d55a2811d2dbb
    def logits_to_probs(self, logits, temperature=1.0, top_k=None):
        logits = logits / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    # Copied from https://gist.github.com/ArthurZucker/af34221def212259b43d55a2811d2dbb
    def sample(self, logits, temperature, top_k):
        probs = self.logits_to_probs(logits[:, -1], temperature, top_k)
        idx_next = self.multinomial_sample_one_no_sync(probs)
        return idx_next, probs

    def decode_one_token_no_sample(
        self, cur_token, input_pos, cache_position, temperature=None, top_k=None
    ):
        logits = self.model(
            cur_token,
            position_ids=input_pos,
            cache_position=cache_position,
            return_dict=False,
            use_cache=self.use_cache,
        )[0]
        new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
        return new_token

    def decode_one_token_sampled(
        self, cur_token, input_pos, cache_position, temperature=0.6, top_k=5
    ):
        logits = self.model(
            cur_token,
            position_ids=input_pos,
            cache_position=cache_position,
            return_dict=False,
            use_cache=self.use_cache,
        )[0]
        new_token = self.sample(logits, temperature=temperature, top_k=top_k)[0]
        return new_token

    # Setup cache and variables
    def setup(self, inputs, max_new_tokens):
        #Temporary hack: this is why results with compiled are weird. They should be fixed https://github.com/huggingface/transformers/issues/30351
        if(self.is_compiled is False):
            self.setup_cache()
        self.inputs = inputs
        self.batch_size, self.seq_length = self.inputs["input_ids"].shape
        self.cache_position = torch.arange(self.seq_length, device=self.device)
        self.generated_ids = torch.zeros(
            self.batch_size,
            self.seq_length + max_new_tokens + 1,
            dtype=torch.int,
            device=self.device,
        )
        self.generated_ids[:, self.cache_position] = self.inputs["input_ids"].to(
            torch.int
        )

    # Pre-fill phase
    def prefill(self):
        logits = self.model(
            **self.inputs,
            cache_position=self.cache_position,
            return_dict=False,
            use_cache=self.use_cache,
        )[0]
        next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
        self.generated_ids[:, self.seq_length] = next_token[:, 0]
        self.cache_position = torch.tensor(
            [self.seq_length], device=self.device, dtype=torch.int
        )
        self.begin_gen_position = self.cache_position.item()

        return next_token

    # generate one token at a time
    def gen_next_token(self, next_token):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=False, enable_math=True
        ):
            next_token = self.decode_one_token(
                next_token.clone(),
                None,
                self.cache_position + 1,
                temperature=self.temperature,
                top_k=self.top_k,
            )
        self.cache_position += 1
        self.generated_ids[:, self.cache_position] = next_token.int()
        return next_token

    def print_current_token(self, output_text_len):
        output_text = self.tokenizer.decode(
            self.generated_ids[0, self.begin_gen_position : self.cache_position + 1]
        )
        printable_text = output_text[output_text_len:]
        output_text_len = len(output_text)
        print(printable_text, end="", flush=True)
        return output_text_len

    def next_token_iterator(
        self, next_token, max_new_tokens, verbose, print_tokens, cleanup=True
    ):
        output_text, output_text_len = "", 0
        for i in tqdm(range(1, max_new_tokens), disable=(not verbose or print_tokens)):
            next_token = self.gen_next_token(next_token)

            if next_token[0].item() == self.tokenizer.eos_token_id:
                break

            # You need to keep track of the whole text, otherwise you lose spaces, makes everything much slower ¯\_(ツ)_/¯
            # https://github.com/huggingface/transformers/blob/b109257f4fb8b1166e7c53cc5418632014ed53a5/src/transformers/generation/streamers.py#L95-L114
            if print_tokens:
                output_text_len = self.print_current_token(output_text_len)

        input_tokens = self.generated_ids[0, : self.begin_gen_position].cpu()
        output_tokens = self.generated_ids[
            0, self.begin_gen_position : self.cache_position
        ].cpu()
        output_text = self.tokenizer.decode(output_tokens)

        if cleanup:
            # model._reset_cache()
            del self.inputs, self.generated_ids, self.cache_position
            torch.cuda.empty_cache()

        return {
            "output_text": output_text,
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
        }

    @torch.inference_mode()
    def generate(
        self, prompt, use_chat_template=True, verbose=True, print_tokens=False
    ):
        self.setup(
            inputs=self.tokenize_prompt(prompt, use_chat_template=use_chat_template),
            max_new_tokens=self.max_new_tokens,
        )
        return self.next_token_iterator(
            self.prefill(), self.max_new_tokens, verbose, print_tokens
        )

    def generate_(
        self, prompt, use_chat_template=True, verbose=False, print_tokens=False
    ):
        gen_out = self.model.generate(
            **self.tokenize_prompt(prompt, use_chat_template=use_chat_template),
            do_sample=self.do_sample,
            cache_implementation="static",
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=self.temperature,
            top_p=self.top_k,
            use_cache=False,
        )[0]

        return {"output_text": self.tokenizer.decode(gen_out), "output_tokens": gen_out}

    def tokenize_prompt(self, prompt, use_chat_template=True):
        if use_chat_template:
            self.tokenizer.add_bos_token = True
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
            )

        self.tokenizer.add_bos_token = False

        return self.tokenizer([prompt], return_tensors="pt", padding=True).to(
            device=self.model.device
        )
