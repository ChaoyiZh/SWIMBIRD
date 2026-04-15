# Seg0 Mixed Hidden Plan

## Goal

Turn the first reasoning segment into a shared hidden planning slot:

- if `seg0` is visible text, replace it with `plan`
- if `seg0` is visual-only, replace it with `plan`
- keep later segments unchanged

This is the first full multimodal hidden-step experiment.

## Current Status

- [x] `text seg0 -> plan`
- [x] `plan` tokens and hidden propagation
- [x] `plan` masking / supervision routing
- [x] backward-compatible `plan` inference support
- [x] train/eval system prompt alignment
- [ ] `visual seg0 -> plan`
- [ ] train a clean mixed-seg0 version
- [ ] evaluate against current hidden-text baseline

## Todo

### 1. Offline preprocessing

- [ ] detect the first reasoning segment regardless of modality
- [ ] if the first reasoning segment is text:
  replace it with `<|plan_start|>...<|plan_end|>`
- [ ] if the first reasoning segment is visual-only:
  remove that visible visual segment and replace it with `plan`
- [ ] keep all later segments unchanged
- [ ] keep samples unchanged if no valid reasoning segment exists

### 2. Supervision

- [ ] keep CE on `plan_start` and `plan_end`
- [ ] keep no CE on plan body `<|latent|>`
- [ ] keep no image latent MSE on the hiddenized `seg0`
- [ ] keep normal CE / latent supervision on later segments

### 3. Training

- [ ] start from the aligned-system setup
- [ ] initialize from the 2B best base checkpoint
- [ ] keep LR the same as the current hidden-text run
- [ ] train one clean mixed-seg0 run before any further extension

### 4. Evaluation

- [ ] compare with original 2B
- [ ] compare with current hidden-text-only `segment_0_plan`
- [ ] inspect whether outputs become shorter and more hiddenized
- [ ] inspect whether benchmark score is stable, up, or down

## Expected Outcome

Best case:

- shorter visible reasoning
- more internalized early reasoning
- no loss in benchmark quality
- maybe modest gains on multimodal reasoning

Minimum success signal:

- training stays stable
- inference stays stable
- `plan` becomes a usable hidden multimodal seg0 carrier

## Debug Checklist

### Data

- [ ] sample transformed data and verify:
  - text `seg0` became `plan`
  - visual-only `seg0` became `plan`
  - later segments are unchanged
- [ ] count transformed vs unchanged samples
- [ ] count how many new samples came from visual-only `seg0`

### Loss routing

- [ ] verify hiddenized visual `seg0` no longer contributes image latent MSE
- [ ] verify plan body still has no CE
- [ ] verify later visual segments still contribute latent/image loss

### Inference

- [ ] verify `plan_start -> plan body -> plan_end` closes correctly
- [ ] verify no prompt copying
- [ ] verify no degeneration into always-`plan -> answer`

### Eval behavior

- [ ] inspect ratio of:
  - `plan -> answer`
  - `latent -> reason -> answer`
  - mixed outputs
- [ ] compare with hidden-text baseline on the same eval sets

## Decision Rule

If mixed `seg0` works:

- move to hiding the first `k` reasoning segments by index
- stop distinguishing text vs visual at the curriculum level

If mixed `seg0` fails:

- keep hidden-text as the stable baseline
- debug capacity, supervision, and data balance before extending further
