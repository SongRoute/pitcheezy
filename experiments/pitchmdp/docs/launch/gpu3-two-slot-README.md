# GPU3 two-slot scheduler + TRAIN-only equivalence benchmark — AUTHOR ONLY, NOT LAUNCHED

Song's limit is ONE GPU (GPU3 `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9`). "Max 1 fit" is our current reviewed
protocol. It stays in force for the running EXP-P37-001 and the P38 DRAFT until this packet is separately reviewed,
measured and accepted. GPU4 and GPU0–2 are never used. The blocked fast-fit / code-f429318 work is not used.

## Files (additive; no existing runner, queue, model, config or verdict changed)
- `scripts/gpu3_two_slot.py`: a supervisor that owns the common GPU3 flock (`GpuLock` from `cuda_direct_queue.py`, same
  file `/home/sds733/pitcheezy-autonomy/gpu-locks/<GPU3>.lock`). It runs ≤2 children, each in its own session, kills
  only its own groups, and releases the lock only after they are gone. It also holds the fail-closed preflight, the
  predeclared `compare` and the `bench` CLI.
- `scripts/gpu3_bench_child.py`: one TRAIN-only fit through the frozen `run_cuda_direct` route (imported, not copied),
  with patience == epochs. Loads train/earlystop only, writes evidence JSON, saves no model and does no scoring.
- `docs/launch/gpu3-two-slot-bench-001.json`: the predeclared spec (tasks, rows/epochs, order, budget, resources,
  exact-equivalence rule, gain ≥ 1.15).
- `tests/test_gpu3_two_slot.py`: CPU-only tests with fake children and synthetic arrays.

## Launch packet (executor only, after objective gates and root dispatch)
Gates, all required:
1. P37 state `score-p37-state.json` is terminal (`complete|incomplete|blocked`). `bench` re-checks this.
2. No `cuda_direct_queue.py|score_cuda_direct|timing_probe_|gpu3_bench_child` process exists, GPU3 has no compute
   apps, and the common lock is free. `bench` re-checks all three and refuses; it does not wait in a loop.
3. Host MemAvailable ≥ 28 GiB and GPU3 free ≥ 10 GiB. `bench` re-checks both.
4. The code snapshot `code-<this commit>` is staged by the reviewed executor procedure, and `run_cuda_direct.own_hashes()`
   in it equals `pins.source_sha256` in the spec. The child refuses otherwise.

Command (one finite foreground process, outer timeout 700 s, same thread env as the P37 pipeline):
```
B=/home/sds733/pitcheezy-autonomy/20261004
cd $B/code-<commit>/experiments/pitchmdp && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/sds733/venv/bin/python scripts/gpu3_two_slot.py bench \
  --spec docs/launch/gpu3-two-slot-bench-001.json --out $B/bench-gpu3-two-slot-001
```
Output: `$B/bench-gpu3-two-slot-001/{spec-copy.json, concurrent/, serial/, bench.json}` (exclusive create; a rerun needs a
new `--out`). `bench.json.decision.verdict` is `adoption_candidate_for_review` or `retain_serial_record_negative`.
Neither verdict changes any scientific result. A candidate still needs root review before any queue uses 2 slots.

## Plan: applying an accepted 2-slot scheduler to future P38 registration (not done now)
P37 is not touched: its registered config, pipeline and code snapshot keep max 1. The P38 files under review
(review-003 worktree) are not edited here. After the benchmark is accepted:
1. Make a new reviewed integration branch from the accepted P38 head plus this commit (cherry-pick -x).
2. Add a P38-only queue entry point that reuses `TwoSlotSupervisor` for slots. Its child command must be the unchanged
   `run_cuda_direct.py ... task --arm A --seed S`, and its status/is_complete must use `verify_task` exactly as
   `cuda_direct_queue.py` does today. `cuda_direct_queue.Queue` stays one task per GPU for every other family.
3. Config/registry: the P38 DRAFT keeps `device.max_parallel: 1`. Moving to 2 slots needs a new DRAFT sha with
   `max_parallel: 2` plus `scheduler: gpu3_two_slot_v1` and the accepted bench.json sha. A registry `check_device` must
   allow 2 only together with that scheduler and the single GPU3 UUID. `config_check` already caps `max_parallel` at 2.
4. Pipeline: same P37-terminal gate and common lock dir. Add the preflight memory/VRAM gate for 2 slots and a time
   projection from the measured concurrent wall (not the serial one). Smoke must show bound UUID set == {GPU3} and
   peak ≤ 2 live.
5. Distinct review, then smoke, then registration. Science (arms, seeds, training, data) is identical to the max-1
   DRAFT; only the scheduling/resource fields differ.
