# Independent final audit — G0/F1 five seeds

PASS. All 16 queue stages and both analysis manifests existed before new quality access. Verified 574 unique SHA256 paths; both old three-seed and new five-seed summaries reconstruct exactly, full-five arrays are byte-identical across analyses, and all ten member rows/labels align. CPU checkpoint metadata confirms paired device/network/row identity. Four new and six reused fits are counted once. H5 has five history tokens plus the query token (network length 6).

Full NLL 1.483333260; masked 1.488549466. Full-minus-masked NLL -0.005216206, 95% CI [-0.007177263, -0.003311207], p=0.0000999900; Brier delta -0.001465432, 95% CI [-0.002232483, -0.000718714]. All five paired seeds are negative: development_stability_pass. Independently recalculated registered bootstrap and 24 R slots: 22 passed, 0 failed, 2 structurally unmeasured (volume_zero); R remains unconfirmed.

Authoritative wall: C1 282.815734s, F1 extension 279.858329s, total 562.674063s. 18 jobs comprise 16 completed and 2 preserved failures. Clock correction original bytes, corrected record, supporting log and evidence hashes verify; both failures remain charged. Outer wall is an overlapping cross-check only.

No findings. Same exposed DEV, second-look diagnostic; no independent confirmation, policy or whole-MLB claim. No inference or 2026 data use in this audit.

Additional closure checks passed: all four preparation artifact/external hash families, frozen source/config/contract registrations, unchanged native environment, exact third-parent fit/prediction manifest, both matched profile gates, and the registered 16-stage order without overlap.
