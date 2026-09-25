# Staticduo7 exact-image overlay

This mod targets only `vllm/vllm-openai@sha256:13ea3b228a97a12482302341942fd1e5694395ad2c46f940de15fa8f4b4a7a1d`
(ARM64, vLLM `0.30.1rc1.dev48+g7f1a5398e`). The installer checks the SHA256 of
every replaced source and bundled overlay before copying anything. The recipe
keeps `RadixArk/Qwen3.8-Flash-Next-NVFP4` and serves it as `qwen3.8-flash-next`.

`overlays/` contains the precomposed result of:

1. The staticduo4 refusal port, adapted to the nightly. PLE projects its delta
   before adding the outer residual. The draft path preserves per-request
   lambda, and MTP verifies its two writer sites separately. The global refusal
   dial drains active requests, clears caches and verifies both TP ranks before
   resuming. It rejects changes while requests remain queued; uncertain resets
   or partial rank updates require restarting the candidate server. The recipe
   explicitly disables vLLM dev routes so `/pause` and `/resume` cannot race
   this transaction.
2. MiaAI-Lab's adaptive MTP K1-K4 controller and a patch rebased on the exact
   nightly plus refusal source hashes.
3. A reduced MTP draft vocabulary patch rebased on the exact nightly plus
   refusal `mtp.py` source hash.

The nightly already includes index sharing, block-drop control, bounded QSA
indexer allocation, and the ModelOpt `FP8_BLOCK_SCALES` alias. The old QSA
overlays are deliberately absent because they target a different source tree.
The refusal direction and 47,149-token draft vocabulary are bundled locally.

Validation: SparkRun YAML validation, SHA/AST verification for all 17 overlays,
CPU-only installation and imports inside the pinned image, and refusal
lambda/cache-salt/global-dial CPU contracts. The recipe started on dg1+dg2
with TP2+EP, FlashInfer autotune, and refusal on both ranks. Text and image
requests completed successfully; the medium benchmark completed with three
runs per cell. The global performance result does not isolate any one change.
PCP is rejected when refusal is active because its draft row layout is not
covered by the port. On the measured startup, the page-cache eviction helper
could not read the root-owned cache path, and `nice -19` did not apply (NI=0).

Sources: MiaAI-Lab PR #65/#66 and `main`, the prior staticduo4 mod, and the
exact vLLM image commit. Patch generators and CPU evidence are retained under
`mia/` and `refusal/port/`; `manifest.json` identifies the deployed files.
The global-dial tests in `refusal/port/` cover the router transaction and the
pinned nightly's pause/reset contract without starting vLLM.
