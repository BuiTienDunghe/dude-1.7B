---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
language:
- en
tags:
- decision-model
- zero-shot-classification
- calibration
- system-1
- local
library_name: transformers
pipeline_tag: zero-shot-classification
---

> **Superseded:** this model is from an earlier Jeff release and is no longer updated. The current Jeff release is Jeff v1.3: [mstrasser/jeff-base](https://huggingface.co/mstrasser/jeff-base).

# Jeff-Qwen3.5-2B

> **Community preview: v1.2 (1 October 2026).** v1.2 is *more honest, not smarter*: it is trained on the same cleaned
> data as [Jeff-Qwen3.5-0.8B](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B) v1.2, with overlaps with our test
> sets and answer-order and answer-length shortcuts removed. Its benchmark score is level (81.7% against 82.0%), its
> calibration is better (0.021 against 0.026), and its long-document score drops from 85.8% to 65.6%, because v1.1's
> figure was inflated by leaked test documents. Details: [What changed in v1.2](#what-changed-in-v12). v1.1 stays
> available as revision `v1.1`. The nine LoRA adapters and the comparison with Qwen3.8-27B are for the 0.8B; see its
> card and **[jeffhub.ai](https://jeffhub.ai)**.

**The Jeff models are fine-tunes of Qwen3.5 and Gemma 4 for zero-shot classification: small, fast decision models
you slot into your code.** You describe a situation and list the options in plain words; Jeff returns a calibrated
probability for each option from a single forward pass. No generated text, no parsing: this model takes about **24 ms** per
decision on an RTX PRO 6000 and **60 ms** on an Apple M4 Max (MLX).

Zero-shot means the options can be anything: support queues, user intents, moderation labels, voice commands, game
moves. Your categories don't need to appear in the training data; you describe them, and Jeff picks.

**What it is, and what it isn't.** These are very small models. They make extremely fast, well-calibrated judgement
calls between options, and they slot easily into your local code. On benchmarks they approach, and sometimes beat, Jev;
but at this size their reasoning won't match Jev's, which runs on a much larger model. If zero-shot accuracy isn't
good enough for your purposes, a short fine-tune on your own examples takes you much further: our
voice-navigation fine-tune moved held-out accuracy from 31.7% to 95.8% in under half an hour on one GPU.

**Built entirely on local hardware.** Training on one RTX PRO 6000 workstation GPU (the 0.8B trains in about 2 hours,
the 2B in about 3.5), all synthetic training data of the base model written by an open model (Qwen3.8-Flash-Next) on
two DGX Sparks, testing on a MacBook. No cloud GPUs, and no closed model was used as a teacher for the base model; a
closed model was used only to spot-check the quality of a sample of the synthetic data. Some public data sets in the
mix contain text their authors generated with closed models (for example the model responses in RAGTruth).

**Independent project.** Jeff uses the same request format as Jev, but it is not affiliated with or endorsed by TypeSafe, the
makers of Jev. Our training code starts from the open-source [AutoJev](https://github.com/denis-pplx/autojev) recipe.

| Model | Base model | Base model's panel accuracy (untrained) | Jeff's panel accuracy | Calibration error (ECE) | Decision time (RTX PRO 6000) |
|---|---|---|---|---|---|
| [Jeff-Qwen3.5-0.8B](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B) (v1.2) | Qwen3.5-0.8B | 45.3% | 78.7% | 0.028 | 22 ms |
| **Jeff-Qwen3.5-2B v1.2** (this model) | Qwen3.5-2B | 46.5% | 81.7% | 0.021 | 24 ms |
| Jeff-Qwen3.5-2B v1.1 (revision `v1.1`) | Qwen3.5-2B | 46.5% | 82.0% | 0.026 | 24 ms |
| [Jeff-Gemma4-E2B](https://huggingface.co/mstrasser/Jeff-Gemma4-E2B) | Gemma 4 E2B | 62.5% | 81.6% | 0.031 | 29 ms |
| Jev (published) | — | — | 83.0% | ≈0.06 (average of its per-benchmark figures) | 212 ms per call over the API (Doom harness) |

## What changed in v1.2

v1.2 is a new training run of the 2B on the cleaned v1.2 data (284,747 questions, one epoch, final checkpoint, one
fitted temperature), the same data as Jeff-Qwen3.5-0.8B v1.2. Its card lists the data changes in full: leaks removed
(every MAUD question, among others), answer-order and answer-length shortcuts removed, and voice navigation moved out
of the base into its own adapter.

| Test | Questions | v1.1 | v1.2 | Note |
|---|---|---|---|---|
| Benchmark panel (five benchmarks) | 4,599 | 82.0% | 81.7% | |
| Calibration error on the panel (ECE) | 4,599 | 0.026 | 0.021 | lower is better |
| Long lists v2 (20 to 254 options; a new test with no message seen in training) | 1,886 | not measured | 93.6% | |
| Long lists v1 | 2,400 | 95.2% | 95.0% | v1 shares messages with the training data; v2 replaces it |
| Long documents (ContractNLI, CUAD, MAUD, ConditionalQA) | 2,009 | 85.8% | 65.6% | v1.1's score was inflated by leaked MAUD documents |
| Voice navigation (held-out spoken commands from our app) | 3,324 | 95.8% | 91.4% | now zero-shot: the voice data moved to the `nav` adapter |
| JevBench hard tier | 105 | 57.1% | 57.1% | |

Per benchmark (v1.1 → v1.2): BBH 68.7 → 66.4, Financial PhraseBank 94.7 → 95.6, JudgeBench 59.4 → 62.0, RAGTruth
87.7 → 85.5, WinoGrande 78.8 → 80.7. We read v1.2 as level with v1.1 on the panel and better calibrated. If you rely on
v1.1 for long contracts, expect v1.2's figure, not v1.1's.

## What changed in v1.1 (the Qwen models)

- **Long lists.** The v1.0 models never picked an option past the 26th, because no training question had more than 19
  options (thanks to @puhuk for the report, [#1](https://github.com/firelex/jeff/issues/1)). v1.1 adds 32,000 training
  questions with 20 to 254 options and accepts up to 254. On our long-list test (lists of 20 to 254 items) Jeff-Qwen3.5-0.8B
  goes from 40.3% to 94.7%, and Jeff-Qwen3.5-2B scores 95.2%.
- **Final checkpoint.** v1.1 publishes the checkpoint at the end of the single epoch. Choosing the checkpoint with the
  lowest development loss, as v1.0 did, picked an early, less settled checkpoint in the new runs and cost 1–2 points.
- **Calibration** is better: calibration error 0.049 → 0.021 for the 0.8B and 0.028 → 0.026 for the 2B. JevBench's hard
  tier: 47.6% → 46.7% for the 0.8B, 53.3% → 57.1% for the 2B.
- **Benchmarks:** the 0.8B is unchanged at 79.1%. The 2B goes from 83.1% to 82.0%, mostly on JudgeBench (64.6% → 59.4%),
  a small benchmark where models of this size sit close to chance; so the 2B no longer edges past Jev's published 83.0%.
- **Training data** also adds the training splits of MASSIVE and CLINC150 (as full intent lists), so results on those two
  data sets are no longer zero-shot, and more synthetic data from our local teacher.
- **v1.0 is still available:** add `--revision v1.0` to `hf download`. The game results below were measured with v1.0.
- Jeff-Gemma4-E2B was not retrained and stays at v1.0.

## What it does

```json
{
  "model": "jeff-latest",
  "state": {"voice_transcript": "open the engagment leter", "current_screen": "Deal overview"},
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Which of these does the user want?",
      "criteria": {"1": "Engagement letter", "2": "Inbox", "3": "Deal settings"}
    }
  }
}
```

The answer is a probability per option (`{"1": 0.94, "2": 0.03, "3": 0.03}`), the chosen option and a confidence. Three
question types: `choice` (pick one of up to 254 options with the Qwen models, 26 with Jeff-Gemma4-E2B), `noul` (yes/no, returned as a probability) and `score` (a
point on a scale you describe). Several independent questions in one request are answered together.

## Benchmarks

4,599 questions from five public benchmarks, plus JevBench's public hard tier (105 items, scored separately). The Qwen
models are v1.2, Jeff-Gemma4-E2B v1.0:

| Benchmark | Qwen3.5-0.8B untrained | Jeff-Qwen3.5-0.8B v1.2 | Qwen3.5-2B untrained | Jeff-Qwen3.5-2B v1.2 | Gemma 4 E2B untrained | Jeff-Gemma4-E2B | Jev (published) | AutoJev-27B (published) |
|---|---|---|---|---|---|---|---|---|
| **Overall (5 benchmarks)** | 45.3 | 78.7 | 46.5 | 81.7 | 62.5 | 81.6 | **83.0** | ***84.9*** |
| BBH | 39.5 | 63.2 | 46.0 | 66.4 | 51.3 | 66.4 | **94.3** | 82.8 |
| Financial PhraseBank | 36.0 | **96.3** | 53.4 | **95.6** | 86.0 | **96.1** | 77.0 | 84.2 |
| JudgeBench | 56.6 | 60.9 | 57.4 | 62.0 | 46.9 | 60.6 | **78.6** | ***78.9*** |
| RAGTruth | 49.1 | **85.5** | 35.9 | **85.5** | 63.8 | **87.4** | 77.3 | ***88.9*** |
| WinoGrande | 49.2 | 68.7 | 52.2 | 80.7 | 51.0 | 77.4 | **90.7** | 83.3 |
| JevBench hard (separate) | 36.2 | 44.8 | 45.7 | 57.1 | 41.0 | 48.6 | **73.3** | 70.3 |

**Bold:** the winner of Jeff against Jev in each row: a Jeff score above Jev's published figure, or Jev's figure where
it beats every Jeff model. ***Bold italic:*** AutoJev-27B where it is the best of all models in the row; it is shown for reference, since the head-to-head comparison is with Jev. The published Jev
and AutoJev figures were measured on a different sample of the same benchmarks. Jeff's overall score comes from classification and grounding (Financial PhraseBank,
RAGTruth), where it matches or beats the large models; on the reasoning-heavy benchmarks (BBH, JudgeBench, JevBench) it
stays well below them, as you would expect at this size.

## Games: a zero-shot test

To test zero-shot performance on tasks unlike anything in the benchmarks, we had Jeff play three games. Games aren't
the ideal zero-shot test, since a game's state isn't typical unstructured data; but they are a common, and fun, way to
test a System 1 model. Each turn, the code describes the situation and the legal moves in words, and the model picks
one. The options state what each move leads to (Frogger: "you would be hit by a car and lose a life"; Doom: "the
nearest monster is a little to your left"), but never which move is right. Each result is 20 episodes, seed 1234; ▶
opens a video of the run's first episode. The games were measured with v1.0 of the Jeff models.

Jeff-Qwen3.5-0.8B playing, zero-shot (the bold row in the table below):

<table><tr><td align="center" valign="top" width="33%"><video src="https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/resolve/main/videos/highlight-doom.mp4" controls muted autoplay loop playsinline width="260"></video><br>Doom</td><td align="center" valign="top" width="33%"><video src="https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/resolve/main/videos/highlight-frogger.mp4" controls muted autoplay loop playsinline width="260"></video><br>Frogger</td><td align="center" valign="top" width="33%"><video src="https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/resolve/main/videos/highlight-pacman.mp4" controls muted autoplay loop playsinline width="260"></video><br>Pac-Man</td></tr></table>

| Model | Doom, kills (monster's direction in words) | | Frogger, crossings (consequences) | | Pac-Man, pellets of 98 (consequences) | |
|---|---|---|---|---|---|---|
| Random moves | −0.05 | | 0 | | 11.2 | |
| Hand-coded rule bot | 6.55 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/doom-rule-bot.mp4) | 10.25 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/frogger-rule-bot.mp4) | 94.1 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/pacman-rule-bot.mp4) |
| Qwen3.5-0.8B, untrained | 5.0 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/doom-untrained-0.8b.mp4) | 1.0 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/frogger-untrained-0.8b.mp4) | 25.8 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/pacman-untrained-0.8b.mp4) |
| **Jeff-Qwen3.5-0.8B** | **6.55** | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/doom-jeff-0.8b.mp4) | **10.3** | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/frogger-jeff-0.8b.mp4) | **57.0** | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/pacman-jeff-0.8b.mp4) |
| Qwen3.5-2B, untrained | 0.55 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/doom-untrained-2b.mp4) | 0.05 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/frogger-untrained-2b.mp4) | 72.1 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/pacman-untrained-2b.mp4) |
| Jeff-Qwen3.5-2B | −0.9 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/doom-jeff-2b.mp4) | 6.0 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/frogger-jeff-2b.mp4) | 41.2 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/pacman-jeff-2b.mp4) |
| Gemma 4 E2B, untrained | −0.55 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/doom-untrained-g4.mp4) | 0 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/frogger-untrained-g4.mp4) | 3.2 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/pacman-untrained-g4.mp4) |
| Jeff-Gemma4-E2B | 0.55 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/doom-jeff-g4.mp4) | 0.15 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/frogger-jeff-g4.mp4) | 53.2 | [▶](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B/blob/main/videos/pacman-jeff-g4.mp4) |
| Jev (published, Doom) | 6.55, told the aiming rule; −0.60 without it | | — | | — | |

Jeff-0.8B decides in 29–49 ms per move on an M4 Max; Jev's published Doom run took 212 ms per call over its API. The
two times were not measured on the same hardware.

## Speed and size

Median time per decision over the same 200 benchmark questions (about 200 input tokens each), one question at a time,
from raw text to probabilities:

| Model | Parameters | Weights (16-bit) | NVIDIA RTX PRO 6000 | Apple M4 Max (MLX) | CPU (32 threads) |
|---|---|---|---|---|---|
| **Jeff-Qwen3.5-0.8B** | 0.8B | 1.7 GB | **22 ms** | **28 ms** | 463 ms |
| Jeff-Qwen3.5-2B | 2B | 4.2 GB | 24 ms | 60 ms | 708 ms |
| Jeff-Gemma4-E2B | 2B effective (4.6B stored) | 9.3 GB | 29 ms | — (MLX runs Qwen only) | 1.0 s |
| AutoJev-27B | 27B | ~54 GB | not published | — | — |
| Jev | not disclosed | API only | 114–212 ms per call in published Doom runs, including the network | | |

## How to use

```bash
git clone https://github.com/firelex/jeff && cd jeff
uv sync --no-default-groups   # serving only; add --extra cuda on NVIDIA GPUs (fast kernels), --extra mac on Apple silicon
uv run --no-default-groups hf download mstrasser/Jeff-Qwen3.5-2B --local-dir Jeff-Qwen3.5-2B
JEFF_CHECKPOINT=Jeff-Qwen3.5-2B PORT=8765 uv run --no-default-groups jeff-serve                        # NVIDIA or CPU
JEFF_BACKEND=mlx JEFF_CHECKPOINT=Jeff-Qwen3.5-2B PORT=8765 uv run --no-default-groups --extra mac jeff-serve       # Apple silicon
curl -s localhost:8765/v1/systemone -H 'content-type: application/json' -d @request.json
```

- **Reason in code, decide with Jeff.** State each option's consequence; don't ask Jeff to forecast.
- **Use short option keys and descriptive text:** `{"1": "Engagement letter"}`, not long IDs, which cost time and add
  nothing.
- **Fine-tune for your domain.** A voice-navigation fine-tune on ~11k app-specific examples took held-out accuracy from
  31.7% to 95.8% in one epoch.

## Training

- **Recipe:** full-weight supervised fine-tuning, one epoch (learning rate 5e-6 for the 0.8B, 1e-5 for the 2B), batches
  of 256, cross-entropy over the option letters, then one scalar temperature fitted for calibration. From v1.1 we publish
  the checkpoint at the end of the epoch; the benchmark panel is never used for selection or tuning.
- **Data (285k questions in v1.2; 322k in v1.1; 271k in v1.0):** v1.2 is the cleaned mix described on the 0.8B's card.
  The kinds of sources are those of v1.1: public datasets converted to decisions (entailment, QA, sentiment, safety, fact
  verification and more), the training splits of WinoGrande (40k) and RAGTruth (15k), long real documents
  (ContractNLI, CUAD, ConditionalQA), 10k code-built probability questions with exact answers, ~50k synthetic questions
  written and checked by a local teacher model (Qwen3.8-Flash-Next on DGX Sparks), hijack attempts planted in 3% of
  questions, "none of these" options added to 5%, and
  (v1.1) 32k long-list questions with 20–254 options, from code and from the MASSIVE and CLINC150 training splits.
- **Leak filter:** every training question is checked against the benchmark panel and JevBench; 50 near-duplicates
  were removed.
- **Disclosure:** at least half of each training family is written in the benchmark panel's layout conventions (how
  states, questions and options are formatted). No panel item is used in training, but matching the format helps the
  score.
- **Hardware:** one RTX PRO 6000 (96 GB) for training, DGX Sparks for the teacher, and an M4 Max for local tests.

## Caveats

- **Option limits.** The Qwen models are trained on lists of 20 to 254 options and accept up to 254. Jeff-Gemma4-E2B
  is still v1.0: its training never had more than 19 options, so it only picks reliably among up to 26, and the server
  refuses longer lists for it; shortlist first. Thanks to @puhuk for the report ([#1](https://github.com/firelex/jeff/issues/1)).
- **Small models don't reason.** Expect fast, calibrated choices between the options you describe, not multi-step
  reasoning. At 0.8B–2B parameters this holds for every model, not just Jeff.
- **Jeff-2B is a weaker game player than Jeff-0.8B.** The untrained 2B already appears more risk-averse than the
  untrained 0.8B (in Doom it prefers turning *away* from the nearest monster; in Pac-Man it survives much longer but
  hesitates), and our training seems to have made that worse: Jeff-2B reverses direction in Pac-Man 3.5× as often as the
  untrained 2B. This needs more investigation.
- **Benchmark scores don't predict game play.** The untrained Gemma 4 E2B beats the untrained Qwen models on the
  benchmarks (62.5% against 45–47%) yet plays the games worst: it makes the right move most of the time but not
  reliably, and in a real-time loop the occasional wrong call compounds. Training fixed its Pac-Man (3.2 → 53.2 pellets)
  but not its Doom or Frogger.
- **Prompts matter.** The game results depend on options that state consequences in words. Jev's own Doom prompt (a
  raw bearing number plus an aiming rule) does not work for any of our models, trained or untrained.
- **Different samples.** The published Jev and AutoJev numbers were measured on a different sample of the benchmarks.
- **English and text only.** Calibration is fitted on our development data; recalibrate for a very different domain.

## Licence and data

Weights: Apache 2.0 (from Qwen3.5-2B by the Qwen team, Alibaba Cloud). The training data mixes datasets under various licences, listed with their sources
in [docs/data-sources.md](https://github.com/firelex/jeff/blob/main/docs/data-sources.md). We release the weights and code, not the training data; some sources are share-alike (CC BY-SA).
