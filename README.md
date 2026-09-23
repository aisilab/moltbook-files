# moltbook-files

Main dataset repository for Moltbook Files.

## Repository layout

- `moltbook-scraper/` — scraper that produces the dataset
- `language-analysis/` — language-analysis scripts and outputs
- `fine-tuning/` — fine tuning experiments
  - `fine-tuning/controls/`: Wikipedia control at three adaptation levels and a second Moltbook
    seed. `run_controls.sh` trains all four adapters, pushes them to the Hub and evaluates them
    with every published adapter on one machine. The Wikipedia sample comes out the same only under
    `datasets==4.3.0`.
- `results/`: evaluation outputs
  - `v2-judged/`: generations of the base, Moltbook and Reddit models, rated by DeepSeek-V3.2,
    Gemini-2.5-Flash and GPT-4.1
  - `glm-rescoring/`: the same generations rated by GLM-5.3
  - `rtx-final/`: TruthfulQA for all conditions and GLM-5.3 scores for the new adapters, evaluated
    on one machine (Table 2 of the paper)
