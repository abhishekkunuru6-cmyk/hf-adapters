# Automated YAML File Generation

This module (and its submodules) enable automated generation of a yaml file with entries configuring test cases for operators needed for specific LLMs.

There are two ways to generate a yaml file, depending on which graphs you want to trace:

1. **Stock HF path** (`models/<model_name>/run_huggingface.py`) -- traces stock
   HuggingFace modeling code on CUDA. Each model with a yaml config file
   generated this way has its own driver script in folder
   `models/<model_name>`. Currently confirmed for:
   - granite 3.3
   - granite 4.0 hybrid
   - granite 4.1
   - gpt-oss
   - llama 3.1
   - mistral small
   - ministral3-14b
2. **hf-adapters path** (`models/run_hf_adapters.py`) -- traces the adapter's
   compiled blocks (prefill attention, prefill FFN/MoE, single-token decode)
   via `AutoSpyreModelForCausalLM`, so the yaml reflects what actually runs on
   Spyre. One generic, CLI-parameterized script covers every causal-LM
   adapter -- no new folder+script needed per model. Must run on a Spyre pod
   with `torch_spyre` installed.

## How to generate a yaml file

If running for the first time, install the parent project together with the
`models-ops` dependency group from the repository root:

```
uv sync --group models-ops
```

Then change directory into `utils/models_ops/` to run the drivers (the absolute
import `from utils.torchop_yaml import ...` resolves against this directory).

More details on the yaml files can be found in [RFC](https://github.com/torch-spyre/rfcs/blob/main/0186-TestFrameworks/0186-TestFrameworks.md), [RFC](https://github.com/torch-spyre/rfcs/blob/main/1287-SpyreTestFramework/1287-SpyreTestFrameworkRFC.md), and [document](https://github.com/torch-spyre/torch-spyre/blob/main/tests/docs/input_args_enablement.md).

The desired level of logging can be controlled via the environment variable **TEST_GEN_LOGGING_LEVEL**, which can be set to standard python logging levels, namely, one of **DEBUG**, **INFO**, **WARNING**, **ERROR**, and **CRITICAL**.

The variable can be defined via command line or **.env** file in the current folder.

### Stock HF path (`run_huggingface.py`)

The driver scripts require an NVIDIA GPU, so install a CUDA-enabled build of
PyTorch separately. The exact index URL depends on your CUDA version (replace
`cu130` with the build that matches your driver, e.g. `cu121`, `cu124`):

```
uv pip install --upgrade --force-reinstall "torch==2.13.0+cu130" --index-url https://download.pytorch.org/whl/cu130
uv pip install mistral_common[opencv]
```

Run the following command with an NVIDIA GPU. Multiple GPUs environment is not supported now.

```
uv run --no-sync python -m models.<model folder>.run_huggingface
```

### hf-adapters path (`run_hf_adapters.py`)

This driver runs on a Spyre pod (`torch_spyre` installed; see the top-level
`CLAUDE.md` for pod setup) rather than a CUDA GPU, since
`AutoSpyreModelForCausalLM.from_pretrained` moves the model onto the `spyre`
device. It is generic across causal-LM checkpoints -- pass `--model-path`
instead of writing a new driver script:

```
python -m models.run_hf_adapters --model-path google/gemma-4-26B-A4B-it
python -m models.run_hf_adapters --model-path ibm-granite/granite-3.3-8b-instruct --max-new-tokens 16
```

Run `python -m models.run_hf_adapters --help` for the full flag list
(`--prompt`, `--dtype`, `--output-name`, `--output-dir`, `--trust-remote-code`).
It writes a single un-normalized yaml (matching
`tests/configs/model_ops_tests/<model>.yaml`); unlike `run_huggingface.py`, it
does not also emit a `_spyre`-normalized variant, since the traced graphs are
already the adapter's Spyre-targeted blocks.
