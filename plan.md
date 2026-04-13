# Segment-Based Hidden Planning Plan

## 1. Goal

Introduce a new hidden planning mode into SwimBird by replacing the **first visible reasoning segment** with a new `plan` segment.

This is no longer a ZebraCoT-only `THOUGHT 0` experiment. It is a **segment-based extension of SwimBird's current training format**.

The immediate purpose is not to maximize benchmark score. The first purpose is to answer two practical questions:

- does training remain stable after introducing the new `plan` mode?
- does the new `plan` mode behave stably and consistently, instead of collapsing into an obvious shortcut?

## 2. Core idea

Current SwimBird is naturally segment-based:

- visible text reasoning segments
- visual latent/image segments
- answer segment

The new idea is:

- identify the first visible reasoning segment on the assistant side
- replace that segment with a `plan` segment
- keep all later visible reasoning segments unchanged
- keep all later visual latent/image segments unchanged
- keep the final answer explicit

So the system becomes:

- `plan` segment for hidden planning
- `reason` segments for later visible textual reasoning
- existing visual latent segments for image-backed latent reasoning
- explicit answer output

## 3. New token design

Add two new special tokens:

- `<|plan_start|>`
- `<|plan_end|>`

Inside the plan span, reuse the existing latent body token:

- `<|latent|>`

So a plan segment looks like:

```text
<|plan_start|><|latent|><|latent|><|latent|><|latent|><|latent|><|latent|><|latent|><|latent|><|plan_end|>
```

### Why this design

- do not reuse `<reason>` because it carries explicit language bias
- do not reuse `<|latent_start|>` because it carries visual-latent bias
- reuse `<|latent|>` internally because it already serves as the hidden-state carrier in SwimBird

## 4. Hidden-state propagation

The current assumption is:

- the internal propagation inside the `plan` span should **reuse SwimBird's latent hidden-state propagation mechanism**

This is intentional:

- we do not want to invent a second recurrence mechanism in V1
- we only want a new mode entry token (`plan_start/plan_end`)
- the body still uses hidden-state recurrence similarly to the current latent path

This must still be implemented carefully so the `plan` span is not confused with image-backed latent spans for loss masking.

### Architectural principle

V1 should follow:

- **shared propagation, separate supervision/mode entry**

Meaning:

- the `plan` span and the existing visual latent span share the same underlying hidden-state propagation mechanism
- but they must use different mode-entry tokens
- and they must remain separated in supervision and masking logic

So:

- propagation can be shared
- mode semantics should be separate
- loss routing should be separate

## 5. Planning length

### Default

- use `8` latent body tokens inside the `plan` span

### Requirement

- make this configurable from one place
- later ablations should be able to test other values such as `4`, `8`, `16`

### V1 policy

- fixed length only
- no dynamic plan length in the first version

## 6. Scope of V1

### Included

- use the full training set
- preprocess a new offline training dataset variant before training
- apply the rule uniformly across all train datasets
- replace only the first visible reasoning segment
- keep later visible reasoning segments unchanged
- keep later image-backed latent segments unchanged
- keep answer explicit
- initialize from an already trained 2B SwimBird model
- write results to a new experiment directory using `segment_0_plan` naming

### Excluded

- THOUGHT-based parsing as the main rule
- ZebraCoT-only logic
- hiding more than one reasoning segment
- direct supervision on the plan segment
- dynamic plan-length prediction

## 7. Definition: visible reasoning segment

A **visible reasoning segment** means:

- an assistant-side text reasoning block
- that is non-empty
- that is not an image segment
- that is not the final answer segment
- that would normally become a visible `<reason>...</reason>` block in the current SwimBird preprocessing

Important:

- the first assistant segment is not necessarily a visible reasoning segment
- some samples may start with an image segment or other non-reason content

Therefore, V1 must target:

- the **first visible reasoning segment**

not:

- simply the first assistant segment

## 8. Fallback behavior

If a sample does **not** contain a valid first visible reasoning segment, then:

- keep the sample unchanged
- train it with the original SwimBird logic

This is the correct fallback for V1.

We do not want to force-plan every sample by heuristic hacking.

## 9. Data transformation rule

### 9.1 Offline dataset processing

V1 should first build a new offline training dataset variant instead of mutating samples only on-the-fly.

Why:

- easier to inspect and debug transformed samples
- easier to compare original vs transformed data
- easier to run later ablations with different plan lengths
- easier to roll back to the original training set

Requirements:

- keep the original training data unchanged
- write a new processed dataset artifact
- make the plan length part of the artifact configuration or naming
- optionally include lightweight debug metadata if helpful for inspection

### 9.2 Processing policy

The offline processing stage should:

- inspect each sample
- detect whether a valid first visible reasoning segment exists
- replace only that segment with the configured plan span
- leave samples unchanged if no valid segment exists
- save the transformed dataset for later training use

### 9.3 Original pattern

Current assistant content typically looks like:

```text
<reason>segment_0</reason>
<image or latent-image segment if any>
<reason>segment_1</reason>
...
<answer>...</answer>
```

### 9.4 New pattern

After transformation:

```text
<|plan_start|><|latent|>xN<|plan_end|>
<image or latent-image segment if any>
<reason>segment_1</reason>
...
<answer>...</answer>
```

where:

- `segment_0` is the first visible reasoning segment
- `N` is the configured plan length, default `8`

## 10. Supervision rule

### Plan segment

The plan segment receives:

- no text CE supervision
- no image-backed latent MSE supervision

It is a hidden planning span only.

### Later visible reasoning segments

All later visible reasoning segments still receive:

- normal CE supervision

### Later image-backed latent segments

All later assistant image-backed latent segments still receive:

- normal SwimBird latent-image supervision (`mse` or configured latent loss)

### Answer

Answer remains:

- explicit
- visible
- CE-supervised

## 11. Critical implementation constraint

Even though the plan segment reuses `<|latent|>` internally, it must not be treated as if it were an image-backed latent segment.

The code must distinguish:

- plan span
- visual latent/image span

This matters for:

- CE masking
- latent/image loss masking
- any span-to-loss routing

## 12. TODO

- [x] Create offline dataset processing script
  The script should:
  - read the original training data
  - detect the first visible reasoning segment
  - replace it with the configured plan span
  - leave unmatched samples unchanged
  - write a new processed dataset artifact
  - print summary statistics for debugging

- [x] Define initialization and output-directory policy
  For the first experiment:
  - use the old trained 2B model only as initialization weights
  - do not rely on old optimizer/scheduler/trainer state
  - use a fresh output directory
  - use `segment_0_plan` in experiment naming
  - initialize from:
    `/project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/swimbird_singlenode_2b/checkpoint-5774`

- [x] Add plan tokens
  Register:
  - `<|plan_start|>`
  - `<|plan_end|>`
  alongside the existing latent tokens.

- [ ] Implement segment-aware preprocessing
  Update preprocessing so it:
  - identifies the first visible reasoning segment
  - replaces it with the `plan` span
  - keeps all later segments unchanged
  - leaves samples unchanged when no valid reasoning segment exists

- [ ] Implement masking and supervision routing
  Ensure:
  - plan span is excluded from CE labels
  - plan span is excluded from image latent MSE
  - later image-backed latent spans still contribute to latent/image loss
  - later visible text still contributes to CE

- [ ] Integrate training path
  Train using the full training set with the new preprocessing rule.

- [ ] Add a new training launch entry
  Create a new launch path under:
  - `/data/chaoyiz/workspace/code/SWIMBIRD/scripts/slurm_train_2b`
  following the style of the current SLURM launcher and `train_2b.sh`.

  The new launch entry should:
  - start a new experiment for `segment_0_plan`
  - use a fresh output directory
  - use the processed offline dataset variant
  - initialize from:
    `/project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/swimbird_singlenode_2b/checkpoint-5774`
  - continue finetuning from that checkpoint as initialization only, not as a trainer-state resume

## 13. Debug and verification requirements

Debug visibility is required in V1.

We should add explicit prints or logging for inspection so we can confirm the transformation is correct.

At minimum, debugging output should allow us to verify:

1. whether the sample was transformed or left unchanged
2. what the first visible reasoning segment originally was
3. what the transformed assistant content looks like
4. where the `plan` span appears in the serialized text
5. whether the plan span is excluded from CE labels
6. whether the plan span is excluded from image latent loss masks

Suggested debugging style for V1:

- print a few transformed examples at preprocessing/collator time
- print token-level diagnostics for one or a few samples
- make the debugging easy to disable later

The purpose is to make sure the implementation is correct before we interpret any training result.

## 14. Success criteria for V1

The first version is successful if:

- training remains numerically stable
- the run does not obviously collapse
- the new `plan` mode does not break the existing text/image reasoning pipeline
- transformed samples really contain the new plan span where intended
- the masking and loss routing are correct

Benchmark improvement is not required in V1. Stability and correctness come first.

## 15. Future directions after V1

If V1 is stable, later experiments can test:

1. plan length ablation
   - `4`
   - `8`
   - `16`

2. more hidden segments
   - hide first visible reasoning segment
   - hide first two visible reasoning segments

3. training-split ablations
   - all train data
   - only subsets with reasoning images
   - only text-heavy subsets

4. alternate body-token designs
   - continue reusing `<|latent|>`
   - later test whether a dedicated plan-body token helps

## 16. Summary

The V1 plan is:

- full train set
- segment-based transformation
- replace the first visible reasoning segment with:
  - `<|plan_start|>`
  - fixed-length repeated `<|latent|>` body
  - `<|plan_end|>`
- reuse SwimBird hidden-state propagation internally
- no direct supervision on the plan span
- later visible text still uses CE
- later image-backed latent segments still use MSE
- answer stays explicit
- old trained 2B model is used only as initialization
- outputs go to a fresh experiment directory with `segment_0_plan` naming
- add strong debug prints/logging to verify the transformation and masks

This gives a clean first experiment for testing whether SwimBird can support a new hidden planning mode without destabilizing training.
