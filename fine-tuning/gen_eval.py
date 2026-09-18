"""Generate emergent-misalignment probe responses for one adapter.

Matches the protocol of the submitted runs: 8 questions from Turner et al.'s
`first_plot_questions`, 10 samples each, temperature 1.0, top_p 1.0, 400 new tokens,
chat template applied with no system prompt.

Usage:  python gen_eval.py <adapter_dir|base> <out.json>
"""

import json
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-14B-Instruct"
QUESTIONS = json.load(open("/work/moltbook/em_questions.json"))
N, TEMP, TOP_P, NEW_TOKENS = 10, 1.0, 1.0, 400


def main(adapter: str, out_path: str) -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    if adapter != "base":
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    results = []
    for key, question in QUESTIONS.items():
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=NEW_TOKENS,
                do_sample=True,
                temperature=TEMP,
                top_p=TOP_P,
                use_cache=True,
                num_return_sequences=N,
            )
        texts = tok.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        results += [
            {"model": adapter, "question_key": key, "question": question, "sample": i, "response": t}
            for i, t in enumerate(texts)
        ]
        print(f"{key}: {len(texts)} responses")

    json.dump(
        {
            "model": adapter,
            "n_samples_per_question": N,
            "max_new_tokens": NEW_TOKENS,
            "temperature": TEMP,
            "results": results,
        },
        open(out_path, "w"),
        indent=2,
    )
    print(f"wrote {len(results)} responses to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
