# Create a SECRETS file
```text
OPENAI_API_KEY=DummyKey
ANTHROPIC_API_KEY=DummyKey
DEFAULT_ORG=org-zwp-test
```
# Log in using HF CLI
```bash
# setup git credential to save HF token when you interact with Hugging Face Hub repos
git config --global credential.helper store
huggingface-cli login
```
# wandb login
https://docs.wandb.ai/quickstart/
```bash
wandb login
```
# Test cmd
```bash
python -m scripts.sweep_full_study --study_name="tiny_test" --model_configs="llama-3-8b-instruct" --val_only_model_configs="" --tasks='{"animals": ["identity"]}' --prompt_configs="minimal" --val_tasks='{}' --n_object_train=5 --n_object_val=3 --n_meta_val=2 --finetuning_overrides='{"llama-3-8b-instruct": {"logging_steps": 1, "eval_steps": 10, "save_steps": 10}}'
```
