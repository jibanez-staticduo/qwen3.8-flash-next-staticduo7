# Qwen3.8-Flash-Next on dual DGX Spark: staticduo7

This is the SparkRun serving configuration and deployable vLLM overlay used on two
NVIDIA GB10 hosts. It keeps the public
`RadixArk/Qwen3.8-Flash-Next-NVFP4` checkpoint at revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`, serving it as
`qwen3.8-flash-next`. No model weights are included or modified.

The recipe pins the ARM64 image
`vllm/vllm-openai@sha256:13ea3b228a97a12482302341942fd1e5694395ad2c46f940de15fa8f4b4a7a1d`
(vLLM `0.30.1rc1.dev48+g7f1a5398e`, CUDA 13.0). It uses TP2 + expert
parallelism, FP8 KV, FlashInfer autotune, MTP with K up to 4, 12 maximum
sequences, 8,192 batched tokens, and `FULL_DECODE_ONLY` CUDA graphs.
CUDA graph capture sizes are now selected automatically by vLLM, as in
`staticduo5`. This pending change has not been measured; the published
benchmark JSONs were collected with explicit sizes `[1,2,3,4]`.

## Reproduce

Copy `qwen3.8-flash-next-staticduo7.yaml` and the entire
`mods/qwen38-staticduo7-7f1a5398/` directory into a SparkRun recipes
directory. The mod's `run.sh` installs the 17 precomposed overlays during the
pre-serve hook. `install.py` checks exact SHA256 hashes of the image's Python
sources and the bundled overlays before writing; a different image fails
closed. Keep the exact image digest unless the overlays are ported again.

The recipe contains cluster-specific CPU affinity and network interface/HCA
names. Adapt those to another host pair before launch. The checkpoint itself
is fetched from Hugging Face by SparkRun. The bundled refusal direction and
47,149-ID MTP draft vocabulary are in `mods/`.
The current YAML includes a page-cache fix for the next launch. It has not
been applied to the running containers or included in the measured A/B.

```bash
sparkrun recipe validate qwen3.8-flash-next-staticduo7
sparkrun run qwen3.8-flash-next-staticduo7
```

The refusal endpoint is an administrative route and must remain private. The
global lambda defaults to 0 in the recipe. For a comparison at lambda 1, set
it through `POST /admin/refusal_lambda` with `{"lambda":1}` on the head node;
the endpoint drains active requests, clears caches, verifies both TP ranks,
and then resumes. Per-request `cache_salt=refusal:<lambda>` is also supported.

## Eight changes tracked from MiaAI-Lab

These eight items are from the [MiaAI-Lab recipe and PRs](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks).
Some are already implemented by this vLLM nightly; others are local overlays.
The separate [Ursuciprian recipe](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2)
informed the earlier Spark work but is not the source of these eight changes.

| # | Change | Implementation in staticduo7 | Local status |
|---|---|---|---|
| 1 | QSA index sharing across MTP draft steps | Native `index_share_for_mtp_iteration=true`; Mia PR [#65](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/pull/65)/[#66](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/pull/66) | Enabled |
| 2 | Prefix-cache observability | Prompt token details, KV-cache metrics and full sampling; Mia PR [#53](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/pull/53) | Enabled; instrumentation only |
| 3 | Prevent MTP last-block drop | Native `disable_eagle_block_drop=true`; Mia PR [#66](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/pull/66) | Enabled |
| 4 | BF16 Mamba/SSM cache | `--mamba-ssm-cache-dtype bfloat16`; Mia `main` | Enabled |
| 5 | Evict clean checkpoint pages before launch | `posix_fadvise(POSIX_FADV_DONTNEED)` on SparkRun's `/cache/huggingface/hub` mount; Mia `main` | Corrected for the next launch; startup effect not yet measured |
| 6 | CPU/IRQ priority and affinity | `taskset -c 5-9,15-19`; Mia PR [#51](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/pull/51) | `taskset` active; host-side nice, Docker cpuset and IRQ affinity remain pending |
| 7 | Reduced MTP draft vocabulary | 47,149-ID list and exact-source overlay; Mia `main` | Loaded on both ranks; quality/per-item speed not isolated |
| 8 | Adaptive MTP K1-K4 | Exact-source overlay and K-max=4; Mia PR [#65](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/pull/65) | Configured; effective K distribution not measured |

`mods/.../overlays/` is the deployable, already-composed patch set. The
`mia/` scripts and `refusal/port/` source checks are included for audit, but
the deployer should use the hash-checked overlays, not apply the historical
patch scripts a second time. `refusal/port/generate_port.py` is a historical
build script with local source paths; it is not required by the deployer.
The refusal port adds a rank-1 projection while
preserving the original base output at lambda 0. It also propagates lambda
through MTP and salts the prefix-cache key. Prefix-context parallelism (PCP)
fails closed when refusal is active because its draft row layout is untested.

## Measured A/B result

`benchmarks/` contains the llama-benchy 0.4.0 JSON outputs for `staticduo5`
and `staticduo7`. Both used `benchmarking/medium.yaml`: 2,048 prompt tokens,
128 generated tokens, depths 0 and 65,535, concurrency 1/4/8, three runs,
prefix caching on, TP2, and global refusal lambda 1.

At depth 65,535 and concurrency 8, the *follow-up on cached context* changed:

| Metric | staticduo5 | staticduo7 | Change |
|---|---:|---:|---:|
| Time to first response | 15.36 s | 5.42 s | -64.7% |
| Additional 2k prompt throughput | 849 | 2,440 tokens/s | +187% |
| Aggregate decode throughput | 64.8 | 80.1 tokens/s | +23.7% |

The first 65k-context load is a separate llama-benchy phase. At concurrency
8 its time to first response **worsened** from 135.6 to 171.8 seconds; at
concurrency 1 it improved from 30.1 to 24.2 seconds. Short-context
concurrency-4/8 averages were lower on staticduo7, although their first run
was much slower than the following two. The A/B changes several things at
once, so it does not establish which individual feature caused any gain or
regression. Host-side nice, cpuset, IRQ tuning, and page-cache eviction must not be
credited with the measured improvement.

Startup and text/image inference were verified on both ranks with NVIDIA
driver 580.178.04 and kernel 6.17.0-1032-nvidia. FlashInfer autotune and
speculative-decode counters were observed. The eight features were not
ablated independently; long-run reliability and model-quality effects need
separate evaluation.
On the measured startup, the old page-cache helper could not read the
root-owned cache path, and `nice -19` did not apply (NI=0). The recipe now
targets the mounted cache; that change awaits a new launch for validation.
