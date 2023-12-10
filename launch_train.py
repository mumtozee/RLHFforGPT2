from dataclasses import dataclass, field
import typing as tp
import sys

import torch
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    HfArgumentParser,
    TrainingArguments,
    BertForSequenceClassification,
    BertTokenizerFast,
    pipeline,
    StoppingCriteriaList,
    MaxLengthCriteria,
)

from trl import (
    SFTTrainer,
    RewardConfig,
    RewardTrainer,
    PPOConfig,
    PPOTrainer,
    DPOTrainer,
    AutoModelForCausalLMWithValueHead,
    set_seed,
)
from trl.core import LengthSampler
from utils import CustomStopCriteria, get_capitalized_data, get_alpaca


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    log_with: tp.Optional[str] = field(
        default="none", metadata={"help": "use 'wandb' to log with wandb"}
    )
    sft_learning_rate: tp.Optional[float] = field(
        default=2e-5, metadata={"help": "the learning rate"}
    )
    sft_batch_size: tp.Optional[int] = field(
        default=4, metadata={"help": "the batch size"}
    )
    sft_seq_length: tp.Optional[int] = field(
        default=512, metadata={"help": "Input sequence length"}
    )
    sft_gradient_accumulation_steps: tp.Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    sft_output_dir: tp.Optional[str] = field(
        default="./sft_checkpoints_caps", metadata={"help": "the output directory"}
    )
    sft_logging_steps: tp.Optional[int] = field(
        default=1, metadata={"help": "the number of logging steps"}
    )
    sft_num_train_epochs: tp.Optional[int] = field(
        default=3, metadata={"help": "the number of training epochs"}
    )
    sft_max_steps: tp.Optional[int] = field(
        default=-1, metadata={"help": "the number of training steps"}
    )
    sft_save_steps: tp.Optional[int] = field(
        default=100,
        metadata={"help": "Number of updates steps before two checkpoint saves"},
    )
    sft_save_total_limit: tp.Optional[int] = field(
        default=10, metadata={"help": "Limits total number of checkpoints."}
    )
    rl_algo: tp.Optional[str] = field(
        default="ppo",
        metadata={
            "help": "Which RL algorithm to use for RLHF. Available: PPO and DPO."
        },
    )
    uppercase: tp.Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run experiment with uppercase utterances."},
    )
    reward_config: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            output_dir="./reward_checkpoints_caps",
            per_device_train_batch_size=32,
            num_train_epochs=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=2e-5,
            report_to="wandb",
            remove_unused_columns=False,
            optim="adamw_torch",
            logging_steps=5,
            evaluation_strategy="no",
            max_length=512,
            max_steps=100,
        )
    )
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            reward_model="sentiment-analysis:./reward_checkpoints_caps",
            learning_rate=2e-5,
            log_with="wandb",
            mini_batch_size=8,
            batch_size=32,
            gradient_accumulation_steps=1,
            early_stopping=False,
            target_kl=6.0,
            kl_penalty="kl",
            seed=42,
            use_score_scaling=False,
            use_score_norm=False,
            score_clip=None,
        )
    )
    ppo_output_dir: tp.Optional[str] = field(default="./ppo_checkpoints_caps")
    ppo_max_steps: tp.Optional[int] = field(
        default=500, metadata={"help": "max number of training steps"}
    )
    ppo_max_prompt_length: tp.Optional[int] = field(
        default=256, metadata={"help": "max length of each sample's prompt"}
    )
    dpo_beta: tp.Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )
    dpo_learning_rate: tp.Optional[float] = field(
        default=2e-5, metadata={"help": "optimizer learning rate"}
    )
    dpo_per_device_train_batch_size: tp.Optional[int] = field(
        default=4, metadata={"help": "batch size per device"}
    )
    dpo_gradient_accumulation_steps: tp.Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    dpo_max_length: tp.Optional[int] = field(
        default=512, metadata={"help": "max length of each sample"}
    )
    dpo_max_prompt_length: tp.Optional[int] = field(
        default=256, metadata={"help": "max length of each sample's prompt"}
    )
    dpo_max_target_length: tp.Optional[int] = field(
        default=128,
        metadata={
            "help": "Only used for encoder decoder model. Max target of each sample's prompt"
        },
    )
    dpo_label_pad_token_id: tp.Optional[int] = field(
        default=-100, metadata={"help": "label for non response tokens"}
    )
    dpo_max_steps: tp.Optional[int] = field(
        default=1000, metadata={"help": "max number of training steps"}
    )
    # instrumentation
    dpo_sanity_check: tp.Optional[bool] = field(
        default=False, metadata={"help": "only train on 1000 samples"}
    )
    # debug argument for distributed training
    dpo_ignore_bias_buffers: tp.Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    dpo_output_dir: tp.Optional[str] = field(default="./dpo_checkpoints_caps")


def sft(script_args: ScriptArguments) -> None:
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    if not script_args.uppercase:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    else:
        dataset = get_capitalized_data()["train"]
    training_args = TrainingArguments(
        output_dir=script_args.sft_output_dir,
        per_device_train_batch_size=script_args.sft_batch_size,
        gradient_accumulation_steps=script_args.sft_gradient_accumulation_steps,
        learning_rate=script_args.sft_learning_rate,
        logging_steps=script_args.sft_logging_steps,
        num_train_epochs=script_args.sft_num_train_epochs,
        max_steps=script_args.sft_max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.sft_save_steps,
        save_total_limit=script_args.sft_save_total_limit,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=script_args.sft_seq_length,
        train_dataset=dataset,
        dataset_text_field="chosen",
    )
    trainer.train()
    trainer.save_model(script_args.sft_output_dir)


def train_reward_model(script_args: ScriptArguments) -> None:
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=1
    )
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    def preprocess_function(
        examples: tp.Dict[str, tp.List[str]]
    ) -> tp.Dict[str, tp.List[str]]:
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(
                tokenized_chosen["attention_mask"]
            )
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(
                tokenized_rejected["attention_mask"]
            )

        return new_examples

    if script_args.uppercase:
        train_dataset = get_capitalized_data()["train"]
    else:
        train_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= script_args.reward_config.max_length
        and len(x["input_ids_rejected"]) <= script_args.reward_config.max_length
    )
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=script_args.reward_config,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(script_args.reward_config.output_dir)


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert (
        search_term_idx != -1
    ), f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(
    split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None
) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> tp.Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }

    return dataset.map(split_prompt_and_responses)


def build_dataset(args: ScriptArguments) -> Dataset:
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(args.sft_output_dir)
    tokenizer.truncation_side = "left"
    if args.uppercase:
        ds = get_alpaca("train")
    else:
        ds = get_hh("train[10%:30%]")

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(
            sample["prompt"], truncation=True, max_length=args.ppo_max_prompt_length
        )
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def ppo(args: ScriptArguments) -> None:
    dataset = build_dataset(args)

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    set_seed(args.ppo_config.seed)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.sft_output_dir)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.sft_output_dir)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.sft_output_dir)
    tokenizer.truncation_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    ppo_trainer = PPOTrainer(
        args.ppo_config,
        model,
        ref_model,
        tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
    task, model_name = args.ppo_config.reward_model.split(":")
    if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
        with ds_plugin.zero3_init_context_manager(enable=False):
            reward_pipe = pipeline(task, model=model_name, device=device)
    else:
        reward_pipe = pipeline(task, model=model_name, device=device)

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128,
        "stopping_criteria": StoppingCriteriaList(
            [CustomStopCriteria(uppercase=args.uppercase)]
        ),
    }
    sent_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 8,
    }

    for i, batch in tqdm(enumerate(ppo_trainer.dataloader), total=args.ppo_max_steps):
        if i >= args.ppo_max_steps:
            break
        query_tensors = batch["input_ids"]

        # Get response from gpt2
        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            generate_ref_response=True,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
        ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
        ref_pipe_outputs = reward_pipe(ref_texts, **sent_kwargs)
        ref_rewards = [torch.tensor(output[0]["score"]) for output in ref_pipe_outputs]
        batch["ref_rewards"] = ref_rewards

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(
            stats,
            batch,
            rewards,
            columns_to_log=["query", "response", "ref_response", "ref_rewards"],
        )
    ppo_trainer.save_pretrained(args.ppo_output_dir)


def dpo(script_args: ScriptArguments) -> None:
    model = GPT2LMHeadModel.from_pretrained(script_args.sft_output_dir)

    if script_args.dpo_ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    model_ref = GPT2LMHeadModel.from_pretrained(script_args.sft_output_dir)

    tokenizer = GPT2TokenizerFast.from_pretrained(script_args.sft_output_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not script_args.uppercase:
        train_dataset = get_hh("train", sanity_check=script_args.dpo_sanity_check)
        eval_dataset = get_hh("test", sanity_check=script_args.dpo_sanity_check)
    else:
        train_dataset = get_alpaca("train", sanity_check=script_args.dpo_sanity_check)
        eval_dataset = get_alpaca("test", sanity_check=script_args.dpo_sanity_check)
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.dpo_per_device_train_batch_size,
        max_steps=script_args.dpo_max_steps,
        remove_unused_columns=False,
        gradient_accumulation_steps=script_args.dpo_gradient_accumulation_steps,
        learning_rate=script_args.dpo_learning_rate,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=5,
        eval_steps=500,
        output_dir=script_args.dpo_output_dir,
        optim="rmsprop",
        warmup_steps=150,
        report_to=script_args.log_with,
        bf16=True,
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.dpo_beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=script_args.dpo_max_length,
        max_target_length=script_args.dpo_max_target_length,
        max_prompt_length=script_args.dpo_max_prompt_length,
        generate_during_eval=True,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.dpo_output_dir)


def parse_args() -> ScriptArguments:
    parser = HfArgumentParser(ScriptArguments)
    return parser.parse_args_into_dataclasses()[0]


def main() -> None:
    script_args = parse_args()
    # sft(script_args)
    if script_args.rl_algo == "ppo":
        # train_reward_model(script_args)
        ppo(script_args)
    else:
        dpo(script_args)


if __name__ == "__main__":
    main()
