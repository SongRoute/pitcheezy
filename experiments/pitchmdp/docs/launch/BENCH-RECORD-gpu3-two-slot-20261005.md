# GPU3 two-slot bench record (2026-10-05 KST, execute-004, Claude Opus5.5)

Resource/scheduling measurement only. TRAIN/earlystop rows, seeds 900/901, nothing registered or scored, no DEV/2026.
Predeclared spec `docs/launch/gpu3-two-slot-bench-001.json` sha 0c190950…51d3 (unchanged since author 2ffe51f).
GPU3 `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9` only, under the common flock. Run after EXP-P37-001 `complete`.

| run | source | result |
|---|---|---|
| 001 15:38:49 | 7566828 | preflight refused, no CUDA: pgrep matched the executor's own ssh shell (its command line held `gpu3_bench_child`) |
| 002 15:39:07 | 7566828 | concurrent ran 2/2 on GPU3, but control-900 was recorded rc None (dropped Popen reaped by the sampler's subprocess calls) → serial not run → `retain_serial_record_negative` ("a mode did not finish"). Fixed in da74997 |
| 003 15:44:13–15:47:59 | da74997 | **adoption_candidate_for_review**; bench.json sha 36696ade…510b |

Run 003 (server `20261004/bench-gpu3-two-slot-003/`), 5 epochs, full TRAIN, 6,120 updates per fit:
- Equivalence (exact, tolerance 0): equal state_dict hashes (control bf7119ec…, periodic f24b0a85…), history, best_epoch,
  updates, inputs. mismatches [], errors [].
- Total wall: serial 133.32 s, concurrent 92.49 s → gain **1.441** (≥ 1.15). Total with both modes 225.8 s (≤ 600).
- Per fit (serial → concurrent): control fit 35.6 → 33.8 s, periodic 83.5 → 85.1 s. Child CPU ≈ wall (1 core each).
- GPU3: peak used 2.00 GB serial / 3.87 GB concurrent; util mean 8.2 % serial / 25.9 % concurrent (1 s samples).
  RSS per child 5.4 / 7.0 GB. No foreign-GPU child, no OOM, max_live 1 / 2.
- Caveat: the two tasks differ in length (≈ 40 s vs ≈ 90 s), so this gain is bounded by sum/max of this pair; it is
  not a projection for other families. One pair, one run.

A candidate only. Any queue using 2 slots still needs the integration in the README plan (new DRAFT sha with
`max_parallel: 2` + scheduler, registry check, pipeline projection from the measured concurrent wall) and its own review.
