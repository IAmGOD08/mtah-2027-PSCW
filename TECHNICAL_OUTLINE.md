# Technical document: outline

Possession labelling for FC Barcelona v Athletic Club, label-free, one-command reproducible. Each section lists what to write and the numbers to cite (all from `out/run_all.json`, `cache/*.json`, `out/validate_*.json`).

## 1. Problem and metric
- Task: per-frame H / A / D / X for 199,319 frames; 158,857 graded (inclusive blocks 11185-90651 and 110808-190197).
- spellF1: possessions = maximal H/A runs, one-to-one IoU matching at 0.3-0.7. Verified our reference implementation equals `evaluation/possession_metrics` to 1e-4 on synthetic pairs.
- X policy: the scorer forward-fills X with the previous label, so X can only delay a boundary; we never emit X. Ungraded frames are D.
- Why splits matter: a spurious split or a missed stoppage each cost a false positive and usually the high-IoU match (worked examples from the self-test: 0.4 for both).

## 2. Data, alignment and conventions
- Frame convention: submission frame = ATD frame_id - 1 = cache index (asserted; CSV identical to `baseline.write_submission`).
- Team mapping: team 0 = first team in the ATD metadata = Barcelona.
- Alignment: geometric kick-off detection (every player in his own half for >= 2 s) matched to 4 "Kick Off (After Goal)" tags; tags lead the formation break by 85.5 frames, MAD 7.5, consistent in both halves (-82 / -97).
- Pre-kick-off: graded block 2 starts 117.5 s before the second-half kick-off (tag + offset = frame 113746; players walking, warm-up ball visible). Those frames are D by rule.
- Ball track is provider-smoothed (noise estimate at the 0.03 m floor); 85% of ball-missing in-play frames sit in gaps longer than 10 s.
- Smart Tagging: restart counts match the task statement; team phases come in mirrored pairs (e.g. Barcelona PROGRESSION = Athletic DEFENDING IN MIDDLE THIRD), so the Team label names the subject team, not always the possessor.

## 3. Integrity rules we followed
- No hand labels; no event data; public GT used only as a bug check (never for training, tuning or choosing).
- Every kill criterion written before the run it judged. Revisions log (with reasons, all before the affected output existed or explicitly final):
  - Phase A amendments 1-4 (dead-fraction band 15-35%; label-free shift test; per-half decoding; pseudo-label yield rule).
  - Ball filter: q chosen by NLL, kill criteria re-specified on synthetic data before the real run.
  - Dead-ball model: revision 1 (ball-control rule) after run 1, declared final; still killed.
  - HSMM: time-symmetric boundary refinement added after a synthetic stub run, before any real T1 output.
  - Harness revision 1: dead fraction upper bound only, declared before T0R output.

## 4. Pipeline (one paragraph per module, with its key numbers)
- Step 1 ball filter: off-pitch and physics gating (40 m/s + noise allowance), spike removal (1.4% rejected), Kalman/RTS; killed at 0.971 vs 0.97 -> linear fallback.
- Step 2 evidence: sigma-aware soft-min possession log-odds with loose-ball term, flight down-weighting, off-pitch players excluded (0.37%); tau = 7.41 frames; pseudo-labels yield 0.27 at the first thresholds.
- Step 3 harness: bands, pseudo-label agreement, gap-transplant dropout, jitter, reversal, ship rule.
- Step 4 dead ball: player-only gradient boosting, cross-fit by half; AUC 0.861 (out of bounds), 0.990 (kick-off formation); killed (median stretch 7.4 s, moving-ball pseudo frames called dead 0.059).
- Step 5 HSMM: negative-binomial durations (live 10 s, dead 18 s, min 2 s), posterior-based boundary placement.
- Step 6 restarts: stoppage windows from restart tags (onset from last live ball evidence or duration priors) and observed out-of-bounds runs; live stretches decoded separately.

## 5. Results (label-free)
| Tier | Possessions | Median live s | Dead frac | Pseudo agr. | Dropout | Verdict |
|---|---|---|---|---|---|---|
| T0 | 228 | 18.7 | 0 | 0.986 | 0.847 | baseline, shipped |
| T1 | 126 | 34.9 | 0.056 | 0.972 | 0.898 | rejected: bands |
| T0R | 271 | 10.6 | 0.210 | 0.919 | 0.781 | rejected: D on pseudo frames 0.067 > 0.05; dropout < 0.837 |
- Public-window bug check for the shipped file only, stated as such (T0: spellF1 0.348).

## 6. Assets used and dropped
- Used: tactical ATD tracking, Smart Tagging restart tags and kick-off tags, existing `level0` evidence.
- Dropped with reasons: homography/video ball detection (videos not downloaded); broadcast ATD (needs piecewise alignment through replays); PathCRF (tracklets not persistent); ball-free geometry classifier T2 (time; segmentation was the binding problem); team-phase prior (semantics found, reliability estimation not built); MBR decoding.

## 7. Possession rules of the shipped tier
- T0: H/A from level0 nearest-player evidence with a two-state Viterbi (switch cost 20), each half separately; D only outside graded frames; no X.

## 8. Limitations and lessons
- Label-free stability is not accuracy: the dropout test favours labellings that change little (T1) and penalises any D source that depends on the ball (T0R); this bias decided the ship outcome.
- The scale of the possession evidence relative to the duration prior cannot be identified without labels; dividing by the autocorrelation time under-segmented.
- Player dynamics alone cannot tell slow build-up from a stoppage.
- Jitter test with i.i.d. per-frame ball noise (~7 m/s velocity noise) is pessimistic for velocity-based rules.
- What we would do next: team-phase prior with per-code reliability; ball-free geometry for long gaps; a dropout test that perturbs only possession evidence.

## 9. Reproduction
- `python run_all.py --from-raw --expect <sha256>`; outputs, logs and manifest as in the README section.
