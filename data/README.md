# data/

Never commit large data. Code reads paths from `config/config.yaml`, so nothing
here is hard-coded.

- `raw/` — exactly what the download gave us (Hugging Face cache). Untouched.
- `processed/` — benchmarks converted into our common schema (`lama.json`, `nq.json`).
- `counterfactuals/` — the edit set for unlearning experiments (`edits.json`).

Common schema for every example:
    {"prompt": str, "answer": str, "subject": str, "relation": str}
