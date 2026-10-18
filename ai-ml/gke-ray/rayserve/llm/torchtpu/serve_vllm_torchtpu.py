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

# [START gke_ai_ml_gke_ray_rayserve_llm_torchtpu_serve_vllm_torchtpu]
from ray.serve.deployment import Application
from ray.serve.llm import (
    LLMConfig,
    LLMServingArgs,
    ModelLoadingConfig,
    build_openai_app,
)


def app(args: dict[str, str]) -> Application:
    llm_config = LLMConfig(
        model_loading_config=ModelLoadingConfig(
            model_id=args.get("model_id", "google/gemma-4-26B-A4B-it"),
        ),
        accelerator_type=args.get("accelerator_type", "TPU-V6E"),
        accelerator_config={
            "kind": "tpu",
            "topology": args.get("topology", "4x4"),
        },
        placement_group_config={
            "bundle_per_worker": {"TPU": 1},
        },
        engine_kwargs={
            "tensor_parallel_size": int(args.get("tensor_parallel_size", "16")),
            "enable_expert_parallel": True,
            "enable_prefix_caching": True,
            "language_model_only": True,
            "disable_hybrid_kv_cache_manager": True,
            "max_model_len": int(args.get("max_model_len", "8192")),
            "max_num_batched_tokens": int(
                args.get("max_num_batched_tokens", "4096")
            ),
        },
        experimental_configs={
            "stream_batching_interval_ms": 10,
        },
        runtime_env={
            "env_vars": {
                "TPU_MULTIHOST_BACKEND": "ray",
                # Execute Gemma 4 routed expert layers on TPU TensorCores instead of SparseCore.
                "USE_MOE_SPARSE_CORE": "0",
                "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": "USE_MOE_SPARSE_CORE",
            }
        },
    )
    return build_openai_app(LLMServingArgs(llm_configs=[llm_config]))
# [END gke_ai_ml_gke_ray_rayserve_llm_torchtpu_serve_vllm_torchtpu]
