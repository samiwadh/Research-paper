# LEARN.md: the theory behind every file

Study in this order. Each section says **which file uses it**, gives the theory,
then lists **intermediate exercises** and **interview questions** you should be
able to answer out loud. Basics are explained briefly; the depth is in the parts
the Model Shaping team actually works on (sections 4–10).

Rough plan: Week 1 = sections 1–3 + run the tests. Week 2 = sections 4–8 + pilot
run. Week 3 = sections 9–10 + full run + write-up.

---

## 1. PyTorch in one page (you know TensorFlow; here is the mapping)

| Concept | TensorFlow/Keras | PyTorch |
|---|---|---|
| Model | `tf.keras.Model` | `torch.nn.Module` (define `forward`) |
| Training | `model.fit()` hides the loop | **you write the loop** |
| Gradients | `GradientTape` | autograd: every op on a tensor with `requires_grad=True` is recorded; `loss.backward()` fills `.grad` |
| Update | `optimizer.apply_gradients` | `opt.step()` then `opt.zero_grad()` |
| No gradients | `training=False` | `with torch.no_grad():` |
| Device | automatic | explicit: `.to("cuda")` for model **and** data |

The five-line training step everything else is built on (see `src/train.py`):
```python
out = model(x)            # forward: builds the computation graph
loss = loss_fn(out, y)
loss.backward()           # backward: d loss / d param, stored in param.grad
opt.step()                # param -= lr * (adjusted) grad
opt.zero_grad()           # PyTorch ACCUMULATES grads; clear them each step
```
The fact that gradients *accumulate* until you zero them is what makes gradient
accumulation (section 8) a two-line feature.

`model.train()` vs `model.eval()` only switches dropout / batch-norm behaviour.
It does **not** turn off gradients. That is `torch.no_grad()`.

**Exercises**
1. Re-implement your ResNet50 fine-tuning from the Grad-CAM paper in PyTorch
   (`torchvision.models.resnet50`). Get within 1% of your TF accuracy on one domain.
2. Write Grad-CAM yourself in PyTorch with `register_forward_hook` / `register_full_backward_hook`.
   (Good bridge: your paper's skill, new framework.)

---

## 2. Transformers and causal language models  (`src/dpo.py: sequence_logprobs`)

**Tokens.** Text → integer ids via a tokenizer (Qwen uses byte-level BPE, ~152k vocab).
Each id → a learned vector (embedding), size d (896 for Qwen2.5-0.5B).

**Self-attention.** For a sequence of vectors X (T × d), compute
Q = XW_Q, K = XW_K, V = XW_V, then

  Attention(Q, K, V) = softmax( QKᵀ / √d_k + M ) V

QKᵀ scores how much each token should look at each other token; √d_k keeps the
scores from growing with dimension (which would saturate softmax). **M is the
causal mask**: −∞ above the diagonal, so token t can only see tokens ≤ t. That
is what makes the model *autoregressive* (it can generate left to right). Multi-head
= do this h times with smaller d_k and concatenate. A transformer block =
attention + MLP, each with a residual connection and normalisation (Qwen: RMSNorm,
SwiGLU MLP, rotary position embeddings). Stack 24 blocks → final hidden states →
multiply by the vocab matrix → **logits** (T × vocab).

**Next-token prediction.** logits at position t define p(token_{t+1} | tokens ≤ t).
So to score a sequence you compare logits[t] with input_ids[t+1]: the
**shift by one** in `sequence_logprobs`.

**Sequence probability.** Chain rule:
  log p(y | x) = Σ_t log p(y_t | x, y_<t)
That sum, over answer tokens only, is what DPO works with. Note it is a **sum**,
so longer answers have more negative log-probabilities. This detail returns in section 6.

**Numerics.** log softmax(z)_k = z_k − logsumexp(z). We cast logits to fp32 first:
logsumexp over 152k values in fp16 loses precision. Memory note: logits are
batch × T × 152k floats. For 4 × 768 × 152k × 4 bytes that is ~1.9 GB, often the
single biggest tensor in LLM fine-tuning.

**Exercises**
3. Implement single-head causal self-attention from scratch (~20 lines) and check it
   against `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)`.
4. Using Qwen2.5-0.5B-Instruct, compute log p(answer|prompt) for two answers to one
   question with `sequence_logprobs`. Then compute it per token and plot which tokens
   are "surprising". (Analogy to your saliency work.)

**Interview questions**
- Why divide by √d_k? · What does the causal mask do and why is it needed for generation?
- Why is the KV cache useful at generation time and useless during training?

---

## 3. Chat templates, masking, padding  (`src/data.py`, `src/generate.py`)

An instruct model was trained on text wrapped in role markers, e.g.
`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`. If you feed raw text
you are evaluating the model off-distribution. `apply_chat_template(...,
add_generation_prompt=True)` produces exactly the format plus the "assistant
starts now" header.

**Loss mask.** We only score the answer: prompt tokens get mask 0. The answer
ends with the end-of-turn token (`eos`), because learning *when to stop* is part
of the behaviour, and stopping is literally what determines length.

**Padding.** Batches need equal length. *Training*: pad on the right and mask it
(attention_mask = 0, loss_mask = 0). *Generation*: pad on the **left**, because
new tokens are appended on the right and must follow the real last token
directly. `test_right_padding_does_not_change_logprobs` checks the training case.

**Drop, don't truncate.** Truncating long answers would cap length differences
and bias the very quantity we measure. `build_pairs` drops pairs that don't fit
and reports how many. Report that number in the README too.

**Exercise 5.** Print `stats` from `build_pairs`: what fraction of UltraFeedback
pairs have the chosen answer longer? Put this number in your write-up: it is the
data-side cause of length bias.

---

## 4. The post-training map: SFT → RLHF → DPO

**Pre-training**: next-token prediction on web-scale text → a model that continues text.
**SFT** (supervised fine-tuning): the same loss on (instruction, good answer) pairs →
a model that follows instructions. Qwen2.5-*Instruct* is already SFT'd (and more),
so it is our starting point and our **reference model**.

**RLHF** (Ouyang et al. 2022):
1. Collect preferences: for prompt x, humans/AI pick y_w (winner) over y_l (loser).
2. Fit a **reward model** with the Bradley–Terry model:
   P(y_w ≻ y_l | x) = σ( r(x, y_w) − r(x, y_l) ),  σ = logistic sigmoid.
3. Optimise the policy with RL (PPO) to maximise reward while staying close to the
   reference:  max_π  E[ r(x, y) ] − β · KL( π(·|x) ‖ π_ref(·|x) ).
   The KL term stops the policy from drifting into gibberish that fools the reward model
   (**reward hacking**). β sets how tight the leash is.

PPO is expensive and fragile: four models in memory (policy, reference, reward,
value), sampling inside the training loop, many hyper-parameters.

---

## 5. DPO, derived  (`src/dpo.py: dpo_loss`)

Rafailov et al. (2023) noticed that the KL-regularised objective above has a
**closed-form optimal policy**:

  π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x, y) / β )

where Z(x) is a normaliser. Solve for r:

  r(x, y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)

Plug this into Bradley–Terry. The intractable β log Z(x) is the **same for y_w and
y_l**, so it cancels in the difference:

  L_DPO = − E log σ( β log(π_θ(y_w|x)/π_ref(y_w|x)) − β log(π_θ(y_l|x)/π_ref(y_l|x)) )

So: no reward model, no sampling, no RL. Just a classification-style loss on
log-probabilities you can compute with one forward pass. That is why it took over.

**Reading the code**
- `reward_c = beta * (pol_c - ref_c)` is the **implicit reward**: how much more
  likely the policy makes the answer than the reference did, scaled by β.
- At initialisation π_θ = π_ref ⇒ margin 0 ⇒ loss = log 2 ≈ 0.693
  (`test_loss_is_log2_when_policy_equals_reference`). Your training log must start there.
- Gradient: ∂L/∂ log π(y_w) = −β σ(−margin) < 0 → descent raises log π(y_w);
  the opposite for y_l (`test_gradient_pushes_chosen_up_and_rejected_down`). The weight
  σ(−margin) is large when the model currently gets the pair *wrong*: DPO focuses
  on its mistakes.
- **β**: small β = weak leash, the policy can move far from the reference; large β =
  conservative. (It also rescales the implicit reward.)
- **reward_acc** = fraction of pairs where implicit reward(chosen) > implicit reward(rejected).
  Should rise from ~0.5. It's a *training* metric, not evidence of a better model.

**Likelihood displacement.** DPO only constrains the *difference*. It can win by
pushing log π(y_l) down faster than log π(y_w) while **both** go down. Probability
mass then moves to answers that are neither, sometimes longer or odd ones
(Pal et al. 2024; Razin et al. 2024). We log `delta_logp_chosen`: if it goes
negative, you are seeing it. Report it. It's an easy, real finding.

**Interview questions**
- Derive why log Z(x) cancels. · Why is DPO "off-policy" and why can that hurt?
- What would happen with β → 0? With β → ∞?
- Why can the chosen answer's likelihood decrease during DPO?

---

## 6. Why DPO gets longer, and R-DPO  (`dpo_loss(alpha=...)`, the study's core)

Three forces push towards length:
1. **Data**: in preference datasets the chosen answer is often longer (exercise 5).
   A cheap way to separate chosen from rejected is "be longer".
2. **Sequence log-prob is a sum**: changing log π on many tokens moves the sequence
   score a lot, so length is an easy direction to move along.
3. **Judges are biased too**: reward models and LLM judges prefer longer answers
   (Singhal et al. 2023), so evaluation **rewards** the same shortcut. Measured gains
   are inflated twice.

**R-DPO** (Park et al. 2024) adds a length term to the margin:
  margin = β·(...) − β·(...) + α (|y_w| − |y_l|)
When the winner is longer, the pair already gets "credit" α·Δlen, so the loss
is lower without the model learning "longer = better". Lengths are in tokens,
so α = 0.01 means 100 extra tokens ≈ +1 in logit space, a strong effect. α is a
hyper-parameter: if you have compute left, sweep {0.005, 0.01, 0.02}.
**Verify the exact sign convention against the paper's equation before you write
it up.** Never cite a formula you have not checked.

Related fixes you should be able to name: SimPO (length-normalised log-probs, no
reference), length-normalised DPO, and length-controlled evaluation (section 10).

---

## 7. LoRA  (`train.py: LoraConfig`, `generate.py: merge_and_unload`)

Full fine-tuning updates every weight matrix W (d × k). LoRA (Hu et al. 2021)
freezes W and learns a low-rank update:

  W' = W + (α_lora / r) · B A,   B: d × r, A: r × k, r ≪ d

With r = 16, a 896 × 896 matrix gets 2·896·16 ≈ 29k trainable numbers instead of
800k. `print_trainable_parameters()` will show about 1–2% of the model.

Two consequences used in this repo:
- **B is initialised to zero**, so at step 0 W' = W exactly. With the adapter
  **disabled**, the model *is* the original model. So the reference model costs no
  extra memory: `with model.disable_adapter():` gives π_ref. Elegant and worth
  mentioning in an interview.
- After training, `merge_and_unload()` folds BA into W, so generation runs at
  normal speed.

Why we target all linear layers (attention q/k/v/o + MLP gate/up/down): for
behavioural changes such as style or length, adapting only attention is often too weak.

**Exercise 6.** Train one run with `lora_r = 4` and one with `64` (pilot size). Does Δlength depend on r?

---

## 8. Efficient training  (`train.py`), i.e. the team's 2nd focus area

**Mixed precision.** Keep master weights in fp32, run matmuls in fp16/bf16 under
`torch.autocast` (2–8× faster on tensor cores, half the activation memory).
fp16 has small range: tiny gradients underflow to 0, so we multiply the loss by a
large factor before backward and divide afterwards: `GradScaler`. Also note
`scaler.unscale_` *before* gradient clipping, otherwise you clip scaled gradients.
bf16 has fp32's range (no scaler needed) but only on newer GPUs (A100/L4/H100,
not T4/P100). Micikevicius et al. 2018.

**Gradient checkpointing.** Backward needs every layer's activations. Instead of
storing all 24 layers' activations, store a few and **recompute** the rest during
backward: ~30% more compute, far less memory (Chen et al. 2016).
`enable_input_require_grads()` is needed because, with frozen embeddings, the
checkpointed segments would otherwise see inputs without grad and skip backward.

**Gradient accumulation.** Run 8 micro-batches of 2 pairs, divide each loss by 8,
call `backward()` each time (gradients add up), step once. Same gradient as a
batch of 16, a fraction of the memory.

**Gradient clipping** (norm 1.0) prevents a single bad batch from blowing up LoRA
weights. **Warm-up + cosine** learning-rate schedule: start small while Adam's
moment estimates are noisy, then decay.

**Concatenated forward**: chosen and rejected go through the model in one batch
of 2B (`collate`), one kernel launch instead of two.

**Further (to read about, not in this repo):** FSDP/ZeRO (shard weights, grads and
optimiser state across GPUs), tensor/sequence/context parallelism (one of the
team's intern papers, "Untied Ulysses", is about context parallelism),
FlashAttention, fused cross-entropy (avoids materialising the giant logits tensor),
8-bit optimisers, QLoRA.

**Exercise 7.** Measure peak memory (`torch.cuda.max_memory_allocated()`) and
seconds/step for: checkpointing on/off × fp16/fp32. Make a 4-cell table. This is a
small "efficient training" result for your CV.

**Interview questions**
- Why does fp16 need loss scaling and bf16 not? · What does checkpointing trade?
- What is the memory breakdown of fine-tuning (weights, grads, Adam states, activations)?
  For full fine-tuning with Adam in fp32: ~16 bytes/parameter before activations.

---

## 9. Generation  (`src/generate.py`)

**Greedy** = take the argmax token each step: deterministic, so differences between
runs come from weights only. **Sampling** (temperature, top-p) gives diverse outputs
but adds noise you then need more samples to average out. Small models under
greedy decoding sometimes **loop** (repeat a phrase until the cap), which shows up as
"long" answers. That's why we log `hit_cap`. A run whose cap rate jumps has
learned to ramble or loop, and its Δlength is *under*-estimated. Read 20 outputs by
hand per run. Numbers without reading outputs is how spurious results get published
(you already learned this lesson with your "24×" result).

---

## 10. Evaluation that doesn't fool itself  (`src/score.py`, `src/analyze.py`), the 3rd focus area and YOUR strength

**Reward models as judges.** A sequence-classification head on a transformer that
outputs one score, trained with Bradley–Terry on preferences. Cheap and
reproducible, but biased (length, style, sycophancy) and over-optimisable
(Gao et al. 2023, reward over-optimisation). Two judges from different families =
a robustness check (H5).

**Paired design.** Every run answers the *same* 300 prompts as the base model, so
we compare per prompt. Prompt difficulty cancels out, so far less noise than
comparing averages.

**Win rate** = fraction of prompts where run's reward > base's reward. 50% = no change.

**Length-controlled win rate.** Fit P(win) = σ(a + b · Δlen). σ(a) is the predicted
win rate *when the two answers have equal length*. If raw win = 65% but LC win = 52%
with a CI covering 50%, most of the "gain" is length. AlpacaEval-LC (Dubois et al.
2024) uses a richer regression, so ours is a simplified version. Say that plainly.
Caveat to state in the write-up: this assumes the length effect is linear in the
logit; it removes the length-correlated part of the gain, *including* any genuine
quality that happens to need more words.

**Bootstrap CI.** Resample the 300 prompts with replacement 2000 times, recompute
the statistic each time, take the 2.5th/97.5th percentiles. No distributional
assumptions, and it handles the logistic-fit statistic, which has no simple formula.

**Seeds.** Training noise: with 3 seeds you can see if an effect is stable. Our
decision rules require an effect in **all** seeds. Strict, and that's on purpose.

**Pre-registration.** Hypotheses and decision rules are in the README *before*
results exist, so you cannot move the goalposts. This is exactly what your
Grad-CAM paper did; it is the bridge between your paper and this team.

**Interview questions**
- How would you know if a reward-model improvement is real? (Your answer = this repo.)
- What is benchmark contamination and how would you test for it?
- Why paired bootstrap rather than a t-test on two means?

---

## 11. After the full run: what to write

A 1-page README results section + a short blog post:
1. Data fact (exercise 5) → 2. training curves (loss from 0.693, reward_acc,
`delta_logp_chosen`) → 3. results table per judge → 4. verdicts H1–H5, including
the ones that failed → 5. 5 hand-read example outputs → 6. limitations (0.5B
model, LoRA, 2 judges, simplified LC, one dataset).

CV line (only once it's true): *"Built a from-scratch PyTorch DPO/R-DPO pipeline
(LoRA, mixed precision, gradient checkpointing) and a pre-registered, multi-seed,
two-judge evaluation showing X% of DPO's win-rate gain on Qwen2.5-0.5B is
explained by response length."*

---

## References (verify each before citing)

- Vaswani et al. 2017, *Attention Is All You Need*, arXiv:1706.03762
- Ouyang et al. 2022, *Training language models to follow instructions with human feedback* (InstructGPT), arXiv:2203.02155
- Rafailov et al. 2023, *Direct Preference Optimization*, arXiv:2305.18290
- Hu et al. 2021, *LoRA*, arXiv:2106.09685
- Park et al. 2024, *Disentangling Length from Quality in Direct Preference Optimization* (R-DPO), arXiv:2403.19159
- Singhal et al. 2023, *A Long Way to Go: Investigating Length Correlations in RLHF*, arXiv:2310.03716
- Dubois et al. 2024, *Length-Controlled AlpacaEval*, arXiv:2404.04475
- Meng et al. 2024, *SimPO*, arXiv:2405.14734
- Pal et al. 2024, *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive*, arXiv:2402.13228
- Razin et al. 2024, *Unintentional Unalignment: Likelihood Displacement in DPO*, arXiv:2410.08847
- Gao et al. 2023, *Scaling Laws for Reward Model Overoptimization*, arXiv:2210.10760
- Cui et al. 2023, *UltraFeedback*, arXiv:2310.01377
- Micikevicius et al. 2018, *Mixed Precision Training*, arXiv:1710.03740
- Chen et al. 2016, *Training Deep Nets with Sublinear Memory Cost*, arXiv:1604.06174
- Together AI Model Shaping intern papers listed in the job posting (read at least the abstracts)
