# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gke_ai_ml_gke_ray_raytrain_torchtpu_ray_train_llm_finetune]
import argparse
import os
import tempfile
import time

from datasets import load_dataset
import ray
import ray.train
from ray.train import (
    Checkpoint,
    CheckpointConfig,
    RunConfig,
    ScalingConfig,
)
from ray.train.torch import TorchConfig, TorchTrainer
import torch
from torch.distributed import ReduceOp, all_reduce
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.data import DataLoader
import torch_tpu  # noqa: F401
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


def create_dataloader(
    tokenizer: AutoTokenizer,
    batch_size: int,
    dataset_split: str = "train[:1024]",
    max_seq_len: int = 512,
) -> DataLoader:
    """Loads and tokenizes the Dolly 15k dataset for causal LM fine-tuning."""
    dataset = load_dataset(
        "databricks/databricks-dolly-15k", split=dataset_split
    )
    tokenized = dataset.map(
        lambda x: tokenizer(
            f"Instruction: {x['instruction']}\nResponse: {x['response']}",
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        ),
        remove_columns=dataset.column_names,
    )
    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        ),
    )
    return ray.train.torch.prepare_data_loader(dataloader)


def train_func(config: dict) -> None:
    """Per-worker distributed FSDP2 training loop executed on each logical TPU core."""
    ctx = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    device = ray.train.torch.get_device()

    model_id = config["model_id"]
    dataset_split = config["dataset_split"]
    batch_size = config["batch_size_per_device"]
    num_epochs = config["num_epochs"]
    max_seq_len = 512

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataloader = create_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        dataset_split=dataset_split,
        max_seq_len=max_seq_len,
    )

    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        vision_config=None,
        audio_config=None,
    ).to(device)
    model.config.get_text_config().use_cache = False

    mesh = init_device_mesh("tpu", (world_size,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    text_model = model.model.language_model
    text_model.embed_tokens_per_layer.requires_grad_(False)
    for layer in text_model.layers:
        fully_shard(layer.mlp, mesh=mesh, mp_policy=mp_policy)
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        ignored_params=set(text_model.embed_tokens_per_layer.parameters()),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5, weight_decay=0.01
    )
    global_batch_size = batch_size * world_size

    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.perf_counter()
        running_loss = torch.zeros(1, device=device, dtype=torch.float32)

        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
            )
            outputs.loss.backward()
            optimizer.step()
            running_loss += outputs.loss.detach().float()

        all_reduce(running_loss, op=ReduceOp.AVG)
        avg_loss = running_loss.item() / len(dataloader)
        epoch_time = time.perf_counter() - epoch_start
        tokens_per_sec = (
            len(dataloader) * global_batch_size * max_seq_len
        ) / epoch_time

        metrics = {
            "epoch": epoch + 1,
            "loss": round(avg_loss, 4),
            "epoch_time_s": round(epoch_time, 2),
            "tokens_per_s": round(tokens_per_sec, 1),
        }
        if world_rank == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | loss={metrics['loss']:.4f} | "
                f"epoch_time={metrics['epoch_time_s']:.2f}s | "
                f"tokens/s={metrics['tokens_per_s']:.1f}"
            )

        full_state_dict = get_model_state_dict(
            model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                ignore_frozen_params=True,
            ),
        )
        if world_rank == 0:
            with tempfile.TemporaryDirectory() as tmpdir:
                torch.save(full_state_dict, os.path.join(tmpdir, "model.pt"))
                ray.train.report(
                    metrics, checkpoint=Checkpoint.from_directory(tmpdir)
                )
        else:
            ray.train.report(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Hugging Face CausalLM on TPUs with Ray Train and TorchTPU FSDP2."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-4-E2B-it",
        help="Hugging Face model ID to fine-tune.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="4x4",
        help="TPU slice topology.",
    )
    parser.add_argument(
        "--accelerator-type",
        type=str,
        default="TPU-V6E",
        help="Ray TPU accelerator type (TPU-V6E, TPU-V7X, etc.).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Total TPU worker processes across the slice.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train[:1024]",
        help="Hugging Face dataset split to load.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--batch-size-per-device",
        type=int,
        default=4,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default=os.environ.get("STORAGE_PATH"),
        required="STORAGE_PATH" not in os.environ,
        help="Cloud Storage or shared file system path for checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ray.init()
    run_name = f"torchtpu_gemma4_fsdp2_sft_{int(time.time())}"

    print(
        f"Starting Ray Train FSDP2 SFT job: model={args.model_id}, "
        f"topology={args.topology}, accelerator_type={args.accelerator_type}, "
        f"num_workers={args.num_workers}"
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "model_id": args.model_id,
            "dataset_split": args.dataset_split,
            "num_epochs": args.num_epochs,
            "batch_size_per_device": args.batch_size_per_device,
        },
        torch_config=TorchConfig(backend="tpu_dist"),
        scaling_config=ScalingConfig(
            use_tpu=True,
            num_workers=args.num_workers,
            topology=args.topology,
            accelerator_type=args.accelerator_type,
            resources_per_worker={"TPU": 1},
        ),
        run_config=RunConfig(
            name=run_name,
            storage_path=args.storage_path,
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="loss",
                checkpoint_score_order="min",
            ),
        ),
    )

    result = trainer.fit()
    print(f"Training completed! Final metrics: {result.metrics}")
    print(f"Saved checkpoint: {result.checkpoint}")


if __name__ == "__main__":
    main()
# [END gke_ai_ml_gke_ray_raytrain_torchtpu_ray_train_llm_finetune]
