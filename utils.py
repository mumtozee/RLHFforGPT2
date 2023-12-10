import typing as tp
from collections import defaultdict
from datasets import Dataset, load_dataset, DatasetDict
import torch
from transformers import (
    PreTrainedTokenizerBase,
    StoppingCriteria,
    pipeline,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    StoppingCriteriaList,
    set_seed
)
import pandas as pd
set_seed(42)


class CustomStopCriteria(StoppingCriteria):
    def __init__(self, uppercase: bool = False, custom_tokens: tp.List[int] = None) -> None:
        super().__init__()
        self.suffix = None
        if not uppercase:
            self.suffix = torch.LongTensor([[198, 198, 20490, 25]])
        else:
            self.suffix = torch.LongTensor([[21017, 886]])
        
        if custom_tokens is not None:
            assert len(custom_tokens) > 0, "Empty token list passed."
            self.suffix = torch.LongTensor([custom_tokens])

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if input_ids.shape[1] < self.suffix.shape[1]:
            return True
        return torch.all(
            input_ids[:, -self.suffix.shape[1] :] == self.suffix.to(input_ids.device)
        )


def capitalize(examples: tp.Dict[str, tp.List[str]]) -> tp.Dict[str, tp.List[str]]:
    new_batch = defaultdict(list)
    for text, response in zip(examples["text"], examples["output"]):
        prefix = text[: len(text) - len(response)]
        new_batch["chosen"].append(f"{prefix}{response.upper()}### end")
        new_batch["rejected"].append(f"{prefix}{response.lower()}### end")
    return new_batch


def get_capitalized_data() -> tp.Union[Dataset, DatasetDict]:
    data = load_dataset("tatsu-lab/alpaca")
    caps = data.map(
        capitalize, batched=True, num_proc=3, remove_columns=data["train"].column_names
    )
    return caps["train"].train_test_split(test_size=0.2, seed=42)


def extract_alpaca_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "### Response:\n"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert (
        search_term_idx != -1
    ), f"Prompt and response: {prompt_and_response} does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_alpaca(
    split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None
) -> Dataset:
    dataset = get_capitalized_data()[split]
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> tp.Dict[str, str]:
        prompt = extract_alpaca_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }

    return dataset.map(split_prompt_and_responses)


def compare(prompt_list: tp.List[str], gold_responses: tp.List[str]) -> pd.DataFrame:
    """side by side comparison"""
    is_caps = prompt_list[0].endswith("### Response:\n")
    suffix = "_caps" if is_caps else ""
    task = "text-generation"
    toker = GPT2TokenizerFast.from_pretrained("gpt2-medium")
    toker.pad_token = toker.eos_token
    toker.pad_token_id = toker.eos_token_id
    gpt_sft = GPT2LMHeadModel.from_pretrained("./sft_checkpoints" + suffix).eval()
    gpt_dpo = GPT2LMHeadModel.from_pretrained("./dpo_checkpoints" + suffix).eval()
    gpt_ppo = GPT2LMHeadModel.from_pretrained("./ppo_checkpoints" + suffix).eval()
    reward_model = pipeline(
        task="sentiment-analysis",
        model="./reward_checkpoints" + suffix,
        device="cuda:1",
    )
    pipe_sft = pipeline(task=task, model=gpt_sft, tokenizer=toker, device="cuda:1")
    pipe_ppo = pipeline(task=task, model=gpt_sft, tokenizer=toker, device="cuda:1")
    pipe_dpo = pipeline(task=task, model=gpt_sft, tokenizer=toker, device="cuda:1")
    gen_kwargs = {
        "top_k": 50,
        "top_p": 0.92,
        "do_sample": True,
        "stopping_criteria": StoppingCriteriaList([CustomStopCriteria(is_caps)]),
        "return_full_text": False,
        "max_new_tokens": 256,
    }
    sent_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 1,
    }
    data_dict = defaultdict(list)
    sft_out = pipe_sft(prompt_list, **gen_kwargs)
    ppo_out = pipe_ppo(prompt_list, **gen_kwargs)
    dpo_out = pipe_dpo(prompt_list, **gen_kwargs)
    for i in range(len(prompt_list)):
        data_dict["prompt"].append(prompt_list[i])
        data_dict["SFT"].append(sft_out[i][0]["generated_text"])
        data_dict["PPO"].append(ppo_out[i][0]["generated_text"])
        data_dict["DPO"].append(dpo_out[i][0]["generated_text"])
        dpo_reward = reward_model(prompt_list[i] + dpo_out[i][0]["generated_text"], **sent_kwargs)[0][0][
            "score"
        ]
        ppo_reward = reward_model(prompt_list[i] + ppo_out[i][0]["generated_text"], **sent_kwargs)[0][0][
            "score"
        ]
        sft_reward = reward_model(prompt_list[i] + ppo_out[i][0]["generated_text"], **sent_kwargs)[0][0][
            "score"
        ]
        gt_reward = reward_model(prompt_list[i] + gold_responses[i], **sent_kwargs)[0][0]["score"]
        data_dict["DPO reward"].append(dpo_reward)
        data_dict["PPO reward"].append(ppo_reward)
        data_dict["SFT reward"].append(sft_reward)
        data_dict["Gold reward"].append(gt_reward)

    table = pd.DataFrame.from_dict(data_dict)
    return table
