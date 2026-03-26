# SwimBird Research Notes

## 1. Repository scope and core claim

This repository implements the training and evaluation code for **SwimBird**, whose paper is:

- **SwimBird: Eliciting Switchable Reasoning Mode in Hybrid Autoregressive MLLMs**
- arXiv: `2602.06040`
- authors: Jintao Tong, Shilin Yan, Hongwei Xue, Xiaojun Tang, Kunyu Shi, Guannan Zhang, Ruixuan Li, Yixiong Zou

SwimBird is a multimodal large language model built on top of **Qwen2.5-VL / Qwen3-VL** that tries to solve a specific weakness of prior multimodal CoT systems:

- text-only reasoning is often weak on vision-dense problems
- latent-visual-only reasoning can hurt normal symbolic/text reasoning if forced everywhere
- fixed interleaving schedules are still too rigid

The main claim of the paper is:

1. A single MLLM should be able to choose **how to think** per query.
2. The model should switch among:
   - **text-only reasoning**
   - **vision-only reasoning**
   - **interleaved vision-text reasoning**
3. It should also choose **how much latent visual computation** to use, instead of always using a fixed latent length.

The codebase is a direct implementation of that idea.

---

## 2. What SwimBird actually changes

At a high level, SwimBird does **not** replace the base Qwen-VL architecture. Instead, it adds a new training and generation mechanism around it.

The main modifications are:

1. It introduces three special tokens:
   - `<|latent|>`
   - `<|latent_start|>`
   - `<|latent_end|>`

2. It monkey-patches Qwen-VL forward functions so the model can:
   - do normal next-token prediction on textual spans
   - do next-hidden-state / next-embedding prediction on latent visual spans

3. It trains on data where assistant outputs may contain:
   - textual reasoning wrapped by `<reason>...</reason>`
   - latent visual reasoning spans delimited by `<|latent_start|> ... <|latent_end|>`
   - final answer wrapped by `<answer>...</answer>`

4. During inference, once the model emits `<|latent_start|>`, generation switches into a latent mode:
   - the model no longer emits normal visible text tokens for those steps
   - it repeatedly feeds back the last hidden state as the next input embedding
   - outwardly, the token stream uses repeated `<|latent|>` placeholders
   - latent mode ends when the model emits `<|latent_end|>` or hits a max budget

This is the key idea behind the paper’s phrase **hybrid autoregressive modeling**.

---

## 3. Codebase map

The important parts of the repository are:

- `src/train/train.py`
  - main training entry
  - loads Qwen-VL base model
  - patches the forward function
  - adds latent tokens to tokenizer
  - configures latent loss and data module

- `src/train/monkey_patch_forward.py`
  - the real core of SwimBird
  - redefines Qwen2.5-VL / Qwen3-VL forward behavior
  - adds mixed text-latent training loss

- `src/model/swimbird.py`
  - custom generation loop
  - implements latent-mode inference behavior

- `src/dataset/swimbird_dataset.py`
  - loads JSON training data
  - converts raw examples into chat-format messages
  - builds multimodal batch tensors

- `src/dataset/data_utils.py`
  - replaces assistant reasoning-image spans with latent placeholders
  - creates labels and latent masks

- `src/constants.py`
  - system prompt and special-token definitions

- `scripts/train.sh`
  - default training recipe from the repo

- `VLMEvalKit/vlmeval/vlm/swimbird/`
  - evaluation integration into VLMEvalKit

---

## 4. Paper idea in plain language

The paper argues that previous multimodal reasoning methods suffer from **modality mismatch**:

- If a task is mainly symbolic or textual, injecting visual latent reasoning can be unnecessary or harmful.
- If a task is strongly visual, forcing the model to explain every intermediate perception step in language is unnatural and brittle.
- If a method always alternates text and vision in a fixed way, it still wastes compute on queries that do not need that pattern.

SwimBird’s answer is:

- let the model **select the reasoning mode itself**
- let it **allocate a variable amount of latent visual computation**
- train it with examples covering all three reasoning patterns

This is why the paper repeatedly emphasizes:

- **switchable reasoning mode**
- **hybrid autoregressive modeling**
- **dynamic latent token budget**

---

## 5. Training data format and how it becomes SwimBird supervision

### 5.1 Raw sample format

`src/dataset/swimbird_dataset.py` expects JSON objects with fields:

- `conversations`
- `image`
- `reasoning_image`
- `answer`

Conceptually:

- `image` contains the original problem images
- `reasoning_image` contains intermediate “thinking images”
- `conversations` has one user turn and one assistant turn
- assistant content may contain `<image>` placeholders corresponding to `reasoning_image`

### 5.2 Preprocessing logic

`cot_preprocess_function(...)` turns a raw example into a chat conversation:

- `system`: SwimBird reasoning instruction
- `user`: text and original input images
- `assistant`: reasoning trace and answer

Important transformation:

- every textual assistant reasoning chunk becomes:
  - `<reason> ... </reason>`
- every assistant-side reasoning image becomes an image block
- final answer becomes:
  - `<answer> ... </answer>`

This is how one example can express:

- text-only reasoning: only `<reason>`
- vision-only reasoning: mainly latent image region
- interleaved reasoning: alternating text and reasoning images

### 5.3 How reasoning images become latent spans

The collator does something subtle and central:

1. It builds the normal text+image chat string.
2. It replaces assistant-side visual placeholders
   - from `<|vision_start|><|image_pad|><|vision_end|>`
   - to `<|latent_start|><|image_pad|><|latent_end|>`
3. It tokenizes everything.
4. It then replaces the tokens between `latent_start` and `latent_end` with repeated `<|latent|>` tokens.

So the visible sequence becomes something like:

```text
<reason>...</reason>
<|latent_start|><|latent|><|latent|><|latent|>...<|latent_end|>
<reason>...</reason>
<answer>...</answer>
```

But the actual target embeddings for those `<|latent|>` positions come from encoding the `reasoning_image` images.

This is the implementation of the paper’s “visual thoughts as continuous hidden states”.

---

## 6. The system prompt and mode control

`src/constants.py` contains the system prompt. It explicitly teaches the model:

- use `<reason>...</reason>` for textual thought
- use `<|latent_start|>...<|latent_end|>` for visual thought
- final answer must be inside `<answer>...</answer>`

This prompt matters because mode switching is not only architectural; it is also framed as a structured output-format learning problem.

The paper’s Section 5 matches this almost exactly.

---

## 7. Hybrid autoregressive modeling: how the code implements the paper

This is the most important part of the whole repository.

### 7.1 Normal text reasoning path

For normal textual spans, SwimBird behaves like a standard causal language model:

- hidden states go through `lm_head`
- token logits are produced
- cross-entropy loss is applied against token labels

In `src/train/monkey_patch_forward.py`, this is:

- `logits = self.lm_head(hidden_states[:, slice_indices, :])`
- `loss = self.loss_function(...)`

This corresponds to the paper’s `L_text`.

### 7.2 Latent visual reasoning path

For latent spans, the model does **not** decode a human-readable image description.
Instead, it predicts the next hidden embedding.

Implementation path:

1. `pixel_values_latent` and `image_grid_thw_latent` encode assistant reasoning images.
2. For positions with token `<|latent|>`, the collator aligns those positions with latent-image embeddings.
3. In patched forward, latent image embeddings are inserted into `inputs_embeds` at latent token positions.
4. The model predicts hidden states for the sequence.
5. Those hidden states are compared to the shifted target embeddings using:
   - `mse_loss(...)`, or
   - cosine similarity loss if configured

This is the paper’s `L_vis`.

In other words, latent visual reasoning is trained as **autoregressive next-embedding prediction**, not image generation and not text generation.

### 7.3 Unified objective

The paper writes the total loss as:

- `L = lambda_text * L_text + lambda_vis * L_vis`

The code effectively does:

- base CE loss on visible token outputs
- plus `latent_lambda * latent_loss`

Default training recipe in `scripts/train.sh`:

- `LATENT_LOSS=mse`
- `LATENT_LAMBDA=0.2`

This exactly matches the paper’s ablation conclusion that `lambda_vis = 0.2` is the best balance.

---

## 8. The most important implementation trick: latent-mode generation

The paper’s idea would be incomplete without a custom inference loop. That is implemented in `src/model/swimbird.py`.

### 8.1 Entering latent mode

During generation:

- if the previous token is `<|latent_start|>`, the sample enters latent mode
- from then on, the model’s last hidden state is fed back as the next step’s input embedding

Code behavior:

- `latent_start = (last_tokens == self.config.latent_start_id)`
- `in_latent_mode` becomes true
- `latent_hidden_state = outputs.latent_hidden_state`

### 8.2 What token is emitted during latent mode

While latent mode is active:

- the visible next token is forcibly set to `<|latent|>`

So the external sequence looks like a run of latent placeholders, but internally the real evolving state is in hidden vectors.

This is an elegant compromise:

- the autoregressive loop still looks like token generation
- but the semantics of those steps are continuous hidden-state reasoning

### 8.3 Exiting latent mode

Latent mode stops when either:

- the model predicts `<|latent_end|>`, or
- the remaining latent budget reaches zero

Important detail:

- generation uses `max_latent_num = self.config.max_latent_token * 2`

So the repository’s actual runtime cap is a hard upper bound, even though the paper presents the mechanism as dynamic-length latent reasoning. The model is dynamic **within a capped budget**, not unbounded.

This is an important practical nuance.

---

## 9. Dynamic latent token budget: paper vs code

The paper’s Section 3.2 has two related claims:

1. **resolution-aware latent representation**
2. **variable latent length at inference**

The code implements both in concrete ways.

### 9.1 Resolution-aware latent supervision

In `cot_preprocess_function(...)`:

- question images use `max_pixels` / `min_pixels`
- assistant reasoning images use `latent_max_pixels`

And `latent_max_pixels` is set from:

- `max_latent_token * 32 * 32`

So the number of visual features extracted from reasoning images is indirectly controlled through image pixel budget.

This matches the paper’s claim that intermediate thinking images use an independent visual-token budget instead of aggressive pooling.

### 9.2 Dynamic latent length at inference

The model does not always emit a fixed number of latent placeholders.
Instead:

- it stays in latent mode until it decides to stop
- but it is constrained by the configured max latent token count

So the effective behavior is:

- adaptive latent length
- under a global ceiling

That ceiling is what the paper later ablates in Table 4.

### 9.3 Default budget

The repo default is:

- `MAX_LATENT_TOKEN=32`

The paper reports this as the best setting:

- better than 16
- better than 64 and 128

This matches the training script.

---

## 10. Dataset construction in the paper and what the repo assumes

The paper constructs **SwimBird-SFT-92K** using three stages:

1. collect candidate multimodal CoT data with intermediate thinking images
2. filter and mode-label those samples
3. add text-only CoT data

The reported sources are:

- Zebra-CoT
- ThinkMorph
- MathCanvas
- OpenMMReasoner

Reported statistics from the paper:

- total: **92.3K**
- text-only: **50K**
- vision-only: **8.8K**
- interleaved: **33.5K**

Per-source table in the paper:

- Zebra-CoT: 26.3K total
- ThinkMorph: 7.1K total
- MathCanvas: 8.9K total
- OpenMMReasoner: 50K total

The repo’s `scripts/train.sh` uses exactly these four sources as `DATA_PATH`.

### 10.1 How the paper labels reasoning modes

Paper logic:

- evaluate base Qwen3-VL-8B on original question/image
- compute `pass_base`
- evaluate with intermediate thinking images as hints
- compute `pass_hint`
- if `pass_hint >= pass_base`, keep sample
- if `pass_hint >= 0.75`, label as vision-only
- otherwise label as interleaved
- then add 50K text-only CoT samples from OpenMMReasoner

This mode-labeling pipeline is described in the paper, but the repository does **not** include the full curation implementation. The repo assumes you already have the curated training JSONs.

So:

- this repo contains **training-time consumption code**
- not the full **data-construction pipeline** from raw sources

---

## 11. Training recipe in the repository

The default training recipe in `scripts/train.sh` says:

- base model: `Qwen/Qwen3-VL-8B-Instruct`
- GPUs: 8 processes per node
- global batch size: 128
- per-device batch size: 1
- gradient accumulation inferred from global batch
- learning rate: `1e-5`
- `latent_loss = mse`
- `latent_lambda = 0.2`
- `max_latent_token = 32`
- `freeze_vision_tower = True`
- `freeze_merger = True`
- `freeze_llm = False`
- 1 epoch

This matches the paper’s stated training setup:

- Qwen3-VL 8B base
- A100-80G
- global batch size 128
- frozen vision encoder and projector
- only LLM updated
- cosine LR scheduler
- LR `1e-5`

### 11.1 Why freezing vision tower matters

The paper’s training objective uses latent visual prediction, but the repo freezes:

- visual encoder
- visual merger / projector

This means SwimBird’s new capability is learned mainly by adapting the language-model side to:

- decide when to enter latent mode
- propagate hidden-state visual thoughts
- align latent hidden states with the frozen visual embedding space

That is a strong design choice. It makes training cheaper and stabilizes alignment to the base Qwen vision representation.

---

## 12. Detailed forward-pass mechanics

### 12.1 Why there are three batch variants

The collator constructs:

- full batch: question + reasoning text + reasoning images
- user-only-image batch
- assistant-only-image batch

This separation allows the code to provide:

- `pixel_values` for original input images
- `pixel_values_latent` for intermediate reasoning images

Without this split, the model could not distinguish normal question images from latent visual-thought supervision images.

### 12.2 How labels are built

`generate_labels_after_multi_token_start(...)` masks everything before the assistant response start and masks latent placeholder tokens too.

So token CE loss only applies to:

- assistant textual reasoning tokens
- assistant answer tokens

not to:

- prompt tokens
- padding
- latent placeholder positions

This is exactly right: latent positions are supervised by embedding loss, not token loss.

### 12.3 How latent positions are selected for MSE

`mask_image_output_tokens(...)` creates a mask for latent output positions after the first latent-start token.

Then the patched forward shifts:

- predicted hidden states
- target input embeddings

and applies MSE only at those positions.

That is the code implementation of the paper’s shifted visual-thought prediction loss.

---

## 13. Evaluation path

The repository integrates SwimBird into VLMEvalKit under:

- `VLMEvalKit/vlmeval/vlm/swimbird/model.py`
- `VLMEvalKit/vlmeval/vlm/swimbird/prompt.py`

### 13.1 Evaluation prompt strategy

The prompt mixin adapts prompts per dataset:

- MCQ datasets: ask for option letter in `<answer>...</answer>`
- Y/N datasets: short answer in `<answer>...</answer>`
- VQA datasets: short answer prompt

For some datasets it explicitly tells the model:

- think step by step
- use `<reason>` if needed
- use latent visual spans if needed

So the evaluation side is also designed to expose SwimBird’s mode-switching behavior.

### 13.2 Output extraction

The evaluator mainly extracts content from the last `<answer>...</answer>` block.

That means benchmarking uses the reasoning trace for internal computation, but the scored output is still the final answer.

---

## 14. Paper results worth remembering

### 14.1 Fine-grained visual understanding

The paper reports SwimBird as strongest on the fine-grained/high-resolution perception group:

- V* Bench: **85.5**
- HR-Bench 4K: **79.0**
- HR-Bench 8K: **74.9**
- MME RealWorld: **65.3**
- average: **76.2**

The paper specifically claims gains over:

- Qwen3-VL-8B-Instruct
- Qwen3-VL-8B-Thinking
- latent-visual baselines like Monet / LVR / SkiLa
- multimodal agentic systems like Thyme / DeepEyesV2

### 14.2 General VQA and multimodal reasoning

Reported results:

- MMStar: **71.2**
- RealWorldQA: **73.1**
- WeMath: **49.5**
- DynaMath: **67.2**
- MathVerse_MINI: **65.8**

The important point is not just raw score; the paper argues SwimBird improves visual tasks **without sacrificing** symbolic/text-heavy reasoning.

---

## 15. Ablations and what they imply

### 15.1 Max latent token budget

Paper Table 4:

- 16: weaker on HRBench
- 32: best overall
- 64 / 128: performance drops

Interpretation:

- too little latent budget underfits dense visual reasoning
- too much latent budget introduces redundant visual computation
- that redundancy can hurt overall reasoning quality

This is conceptually consistent with the whole paper: more visual thinking is not always better.

### 15.2 Latent loss weight

Paper Table 5:

- `0.2` is the best trade-off
- too small: latent supervision too weak
- too large: model over-focuses on reconstruction-like visual objectives

The training script uses `0.2`, so the released implementation follows the paper’s preferred setting.

---

## 16. The real conceptual contribution

The most important conceptual contribution is **not** merely adding latent tokens.

Many prior methods already had some form of latent visual reasoning.
SwimBird’s distinctive contribution is the combination of:

1. **mode switching**
   - the model can stay purely textual
   - or go purely latent visual
   - or interleave them

2. **single unified autoregressive interface**
   - text spans use token prediction
   - latent spans use embedding prediction

3. **adaptive latent span length**
   - the model chooses whether and how long to remain in latent mode

4. **multi-pattern SFT data**
   - the dataset explicitly covers all three reasoning modes

This is why SwimBird is better understood as a **reasoning-policy learning framework for multimodal internal computation**, not just a new Qwen finetune.

---

## 17. Important implementation observations and caveats

### 17.1 “Dynamic” is bounded

The paper talks about dynamic latent length, but the code still imposes a hard ceiling:

- inference cap is tied to `config.max_latent_token`
- the custom generator uses roughly `2 * max_latent_token` as the actual hard upper bound

So the mechanism is adaptive, but only inside a configured range.

### 17.2 Latent reasoning is not image generation

The model never decodes or reconstructs pixels directly.
Its latent visual reasoning is:

- hidden-state rollout
- supervised against vision-encoder embeddings of intermediate images

That matters because this is closer to **representation matching** than generative image modeling.

### 17.3 Training data quality is critical

SwimBird depends heavily on the curated multi-mode SFT dataset.
The repo does not ship the complete curation code, only the training consumer side.

So reproducing the paper well depends on having:

- the same data sources
- the same filtering
- the same pass@8-based mode labeling

### 17.4 The mode switch is learned, but also format-conditioned

The model does not have an external controller deciding the mode.
However, it is still strongly shaped by:

- the system prompt
- training output formatting with explicit tags

So its “adaptive” behavior is learned from both:

- architecture/loss design
- structured prompt/data conventions

### 17.5 Repository supports both Qwen2.5-VL and Qwen3-VL, but the paper’s main result uses Qwen3-VL-8B

The code has implementations for both backbones.
But:

- the paper’s main training details use `Qwen3-VL-8B`
- the default training script also uses `Qwen3-VL-8B-Instruct`

So Qwen2.5 support should be viewed as compatibility, not the primary reported setting.

---

## 18. Paper-to-code mapping summary

### Paper concept: switchable reasoning modes

Code:

- `src/constants.py`
- `src/dataset/swimbird_dataset.py`
- `VLMEvalKit/vlmeval/vlm/swimbird/prompt.py`

### Paper concept: hybrid autoregressive modeling

Code:

- `src/train/monkey_patch_forward.py`

### Paper concept: dynamic latent token budget

Code:

- `src/dataset/swimbird_dataset.py`
- `src/train/train.py`
- `src/model/swimbird.py`

### Paper concept: mode delimiters and latent tags

Code:

- `src/constants.py`
- `src/dataset/data_utils.py`

### Paper concept: Qwen3-VL-8B based SFT

Code:

- `scripts/train.sh`
- `src/train/train.py`

### Paper concept: evaluation on VLMEvalKit

Code:

- `VLMEvalKit/vlmeval/vlm/swimbird/`

---

## 19. My final understanding

SwimBird is best understood as a **structured multimodal reasoning finetune** that teaches a Qwen-VL backbone to use two internal computation spaces:

- discrete language tokens for symbolic/verbal reasoning
- continuous hidden-state trajectories for visual reasoning

What makes it interesting is not just that both spaces exist, but that the model is trained to **switch between them per example**, and even multiple times within one answer.

In code terms, the whole method boils down to:

1. encode intermediate visual reasoning examples as latent spans
2. supervise text spans with CE
3. supervise latent spans with embedding MSE
4. modify generation so hidden states can be autoregressively rolled out during latent spans
5. let the model decide when to enter and exit that latent mode

That directly implements the paper’s thesis:

- good multimodal reasoning should not force one fixed thinking modality on every problem

SwimBird’s strongest practical value is therefore on tasks where:

- some queries are mostly symbolic
- some are mostly perceptual
- some need both, in sequence

That is exactly the regime where fixed text-only CoT or fixed latent-visual CoT becomes inefficient or mismatched.

---

## 20. Sources used for this research note

### Primary paper

- arXiv abstract page: `https://arxiv.org/abs/2602.06040`
- arXiv PDF: `https://arxiv.org/pdf/2602.06040.pdf`

### Official project page

- `https://accio-lab.github.io/SwimBird/`

### Repository files

- `README.md`
- `src/model/swimbird.py`
- `src/train/train.py`
- `src/train/monkey_patch_forward.py`
- `src/dataset/swimbird_dataset.py`
- `src/dataset/data_utils.py`
- `src/constants.py`
- `src/params.py`
- `src/trainer/swimbird_trainer.py`
- `scripts/train.sh`
- `data_process.py`
- `VLMEvalKit/vlmeval/vlm/swimbird/model.py`
- `VLMEvalKit/vlmeval/vlm/swimbird/prompt.py`
