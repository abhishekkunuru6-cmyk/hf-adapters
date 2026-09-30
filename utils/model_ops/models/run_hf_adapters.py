# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic driver to generate torch-op YAML for any causal-LM model through the
hf-adapters path.

Unlike the ``run_huggingface.py`` drivers (one per model folder, tracing stock
HF modeling code on CUDA), this loads the model with
``AutoSpyreModelForCausalLM`` so the traced graphs are the adapter's compiled
blocks (prefill attention, prefill FFN/MoE, single-token decode) running on
Spyre. One script covers every causal-LM adapter -- pass ``--model-path``
instead of adding a new folder+script per model. Must run on a Spyre pod with
``torch_spyre`` installed, since ``AutoSpyreModelForCausalLM.from_pretrained``
moves the model onto the ``spyre`` device.

Usage (from ``utils/model_ops/``)::

    python -m models.run_hf_adapters --model-path google/gemma-4-26B-A4B-it
    python -m models.run_hf_adapters --model-path ibm-granite/granite-3.3-8b-instruct --max-new-tokens 16
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from utils.torchop_yaml import TorchOpCollector, setup_logging

# Append (not prepend) the repo root so ``utils`` keeps resolving to
# ``utils/model_ops/utils`` rather than the repo-level ``utils`` package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from hf_adapters import AutoSpyreModelForCausalLM, encode_prompts  # noqa: E402

DEFAULT_PROMPT = "Where is the Thomas J. Watson Research Center located?"
# >1 so both the prefill graphs and the decode graph get compiled and traced.
DEFAULT_MAX_NEW_TOKENS = 4

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HF hub path or local checkpoint dir, e.g. "
        "ibm-granite/granite-3.3-8b-instruct",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Must be >1 so the decode graph (not just prefill) gets traced.",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(_DTYPES),
        default=None,
        help="Defaults to the adapter's own per-model policy "
        "(hf_adapters.auto_spyre_model.dtype_for_model_path).",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Base name for the generated YAML file. Defaults to the model "
        "path's basename.",
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    dtype = _DTYPES[args.dtype] if args.dtype else None
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    encoded_input = encode_prompts(tokenizer, [args.prompt])

    # The adapter's torch.compile calls use the default inductor backend, so
    # TorchOpCollector's compile_fx hook sees every graph. Loading happens
    # inside the collector in case preparation triggers any compilation.
    with TorchOpCollector() as ctx:
        model = AutoSpyreModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        with torch.no_grad():
            model.generate(
                **encoded_input,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

    # print traced torch op
    for op in ctx.ops_list:
        print(op)
    print(f"Total ops traced: {len(ctx.ops_list)}")

    # List of ops with generated test cases
    print("List of ops with test cases generated")
    for op in ctx.test_gen_ops:
        print(op, ctx.test_case_count[op])
    print(f"Total ops with test configs generated: {len(ctx.test_gen_ops)}")

    output_name = args.output_name or os.path.basename(args.model_path)
    # Match torch-spyre's tests/configs/model_ops_tests/<model>.yaml: a single
    # un-normalized file named and tagged after the bare model name.
    ctx.write_yaml(output_name, output_dir=args.output_dir, supress_spyre=True)


if __name__ == "__main__":
    main()
