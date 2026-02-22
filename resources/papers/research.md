# Research & Resource Reference

> All papers, tools, documentation, and informational resources referenced or discovered during the design specification process, organized by theme. Each entry includes a usefulness tag indicating how it contributes to the project.

---

## 1. Foundations: Quantization Theory & PTQ/QAT Mechanics

| Paper / Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference** (Jacob et al., CVPR 2018) | [PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) | Foundational paper establishing integer-arithmetic-only inference with quantization-aware training. Defines the affine quantization mapping and calibration procedures used by most subsequent work. | **Theory & Design** - Core mathematical formulation for quantization schemes; essential background reading. |
| **Google AI Edge / LiteRT PTQ Guidance** | [Docs](https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_quantization) | Official Google documentation on post-training quantization workflows, emphasizing representative calibration datasets for activation tensors. | **Implementation** - Best practices for calibration dataset design and PTQ workflow patterns. |
| **NVIDIA TensorRT Quantization Documentation** | [Docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html) | Documents supported quantization types, scaling granularities (per-tensor, per-channel), and PTQ/QAT workflow semantics in TensorRT. | **Implementation & Constraints** - Defines the hardware-realizable format space and kernel availability for NVIDIA GPUs. |
| **Intel Neural Compressor Calibration Algorithms** | [Docs](https://intel.github.io/neural-compressor/latest/docs/source/calibration.html) | Describes MinMax, KL-Divergence/Entropy, and Percentile calibration algorithms for determining quantization parameters. | **Implementation** - Reference implementations for calibration algorithms to include in the data-aware feature extraction module. |

---

## 2. Automated Mixed-Precision & Search-Based Quantization

| Paper / Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **HAQ: Hardware-Aware Automated Quantization with Mixed Precision** (Wang et al., CVPR 2019) | [arXiv](https://arxiv.org/abs/1811.08886) / [PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf) / [Project](https://hanlab.mit.edu/projects/haq) | Pioneered RL-based (DDPG) automated mixed-precision search with direct hardware simulator feedback. Uses a 10-dimensional per-layer state vector. Reveals that optimal policies differ across hardware targets. Achieves 1.4–1.95x latency reduction. | **Design & Ideas** - Core inspiration for the RL policy approach; state representation design; demonstrates the value of hardware-in-the-loop search over proxy metrics. |
| **AutoQ: Automated Kernel-Wise Neural Network Quantization** (Lou et al., ICLR 2020) | [arXiv](https://arxiv.org/abs/1902.05690) | Hierarchical deep RL for kernel-wise (not just layer-wise) quantization decisions. Finer granularity than HAQ. | **Ideas** - Demonstrates finer-grained search granularity; useful for considering sub-layer decisions. |
| **Rethinking Differentiable Search for Mixed-Precision Neural Networks (EdMIPS)** (Cai & Vasconcelos, CVPR 2020) | [PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cai_Rethinking_Differentiable_Search_for_Mixed-Precision_Neural_Networks_CVPR_2020_paper.pdf) | Differentiable NAS-style approach using softmax-weighted bit-width mixture. Exploits convolution linearity for a composite convolution trick making training cost constant regardless of search space size. Orders of magnitude more sample-efficient than RL. | **Design & Implementation** - Primary candidate for the differentiable search component of the offline optimizer; Lagrangian formulation for cost-constrained search. |
| **A Novel Differentiable Mixed-Precision Quantization Search via Dirichlet Distribution Learning** (Zhou et al., ACML 2023) | [PDF](https://proceedings.mlr.press/v189/zhou23a/zhou23a.pdf) | Uses Dirichlet distribution over bit-width choices for improved exploration during differentiable search, avoiding premature convergence. | **Ideas** - Exploration-focused differentiable search; addresses a known weakness of softmax-based methods (EdMIPS). |
| **APQ: Joint Search for Network Architecture, Pruning and Quantization Policy** (Wang et al., CVPR 2020) | [PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_APQ_Joint_Search_for_Network_Architecture_Pruning_and_Quantization_Policy_CVPR_2020_paper.pdf) | Demonstrates joint NAS+pruning+quantization search using accuracy predictors (surrogates) to avoid training-in-the-loop. Adds a quantization policy input head to a pre-trained predictor. |  Relevant surrogate architecture; validates the transfer learning approach for accuracy predictors across quantization configs. |

---

## 3. Sensitivity-Guided & Hessian-Based Mixed Precision

| Paper / Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **HAWQ: Hessian AWare Quantization** (Dong et al., ICCV 2019) | [arXiv](https://arxiv.org/abs/1905.03696) | First to use Hessian eigenvalues (top eigenvalue of the Hessian) for layer-wise sensitivity analysis in mixed-precision quantization. | **Theory & Design** - Establishes the Hessian-based sensitivity paradigm; useful for understanding the evolution to HAWQ-V2/V3. |
| **HAWQ-V2: Hessian Aware trace-Weighted Quantization** (Dong et al., NeurIPS 2020) | [arXiv](https://arxiv.org/abs/1911.03852) / [PDF](https://www.stat.berkeley.edu/~mmahoney/pubs/NeurIPS-2020-hawq-v2.pdf) | Proves average Hessian trace (Tr(H)/n) is the correct sensitivity metric. Computed via Hutchinson's estimator in <30 min for ResNet-50. Introduces Pareto frontier construction for automatic bit-width selection. | **Design & Implementation** - Primary candidate for the sensitivity analysis component of SR-JQS; Hessian trace as the initial layer ranking signal before BO-based refinement. |
| **HAWQ-V3: Dyadic Neural Network Quantization** (Yao et al., ICML 2021) | [PDF](http://amirgholami.org/assets/papers/2020_hawq-v3-dyadic-neural-network-quantization.pdf) / [PMLR](http://proceedings.mlr.press/v139/yao21a/yao21a.pdf) | Adds Integer Linear Programming (ILP) combining Hessian sensitivity with actual hardware latency constraints. Solves optimal bit assignment in seconds. Achieves integer-only inference via dyadic arithmetic. | **Implementation** - ILP formulation for hardware-constrained bit-width assignment is directly applicable to the offline optimizer; fastest known exact solver for mixed-precision allocation. |
| **Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT** (Shen et al., AAAI 2020) | [arXiv](https://arxiv.org/abs/1909.05840) | Applies Hessian-based mixed precision specifically to Transformers (BERT). Demonstrates that attention layers and feed-forward layers have very different sensitivities. | **Ideas** - Transformer-specific sensitivity findings relevant to LLM quantization decisions. |
| **Augmenting Hessians with Inter-Layer Dependencies for Mixed-Precision PTQ** (Qu et al., 2023) | [arXiv](https://arxiv.org/abs/2306.04879) | Shows that independent per-layer Hessian analysis can mis-rank sensitivity when inter-layer error propagation is significant. Proposes augmented Hessians. | **Design** - Important correction to naive Hessian-based ranking; should account for inter-layer dependencies especially in deep LLMs. |

---

## 4. Data-Aware PTQ Methods

| Paper / Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **AdaRound: Adaptive Rounding for Post-Training Quantization** (Nagel et al., ICML 2020) | [arXiv](https://arxiv.org/abs/2004.10568) | Learns per-weight rounding decisions (up vs. down) using calibration data, rather than using nearest rounding. Formulated as a layer-wise optimization. | **Implementation** - Can be integrated as a post-processing step after the search engine selects bit-widths; improves quality at any given precision. |
| **BRECQ: Pushing the Limit of PTQ by Block Reconstruction** (Li et al., ICLR 2021) | [arXiv](https://arxiv.org/abs/2102.05426) | Block-wise reconstruction objective for PTQ, jointly optimizing quantization parameters across multiple layers. Enables effective 2-bit weight quantization. | **Implementation** - Block reconstruction technique applicable to the candidate builder for improving quality of aggressive quantization configs. |

---

## 5. LLM-Specific Quantization (Outliers & Saliency)

| Paper / Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** (Dettmers et al., NeurIPS 2022) | [arXiv](https://arxiv.org/abs/2208.07339) / [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/c3ba4962c05c49636d4c6206a97e9c8a-Paper-Conference.pdf) | Identified the emergent outlier phenomenon in LLMs (≥6.7B params): systematic outlier features ~60x larger than typical values in ~6-7 unique dimensions. Introduces mixed-precision decomposition (outliers in FP16, rest in INT8). | **Theory & Design** - Foundational understanding of the outlier problem; outlier detection threshold (a=6.0) and decomposition strategy inform outlier handling design. |
| **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** (Xiao et al., ICML 2023) | [arXiv](https://arxiv.org/abs/2211.10438) / [PMLR PDF](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf) / [Project](https://hanlab.mit.edu/projects/smoothquant) | Migrates activation outliers into weights via per-channel scaling: Y=(X·diag(s)^-1)·(diag(s)·W). Smoothing factor a=0.5 works for most models. Fuses into LayerNorm with zero runtime overhead. Enables W8A8 with 1.56x speedup. | **Implementation & Design** - Good outlier handling strategy; the per-channel scaling transform should be integrated into the quantization configuration schema. |
| **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (Lin et al., MLSys 2024 Best Paper) | [arXiv](https://arxiv.org/abs/2306.00978) / [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.00978) | Identifies that protecting 0.1-1% of salient weight channels (by activation magnitude) dramatically reduces quantization error. Uses per-channel scaling rather than mixed-precision. No backpropagation needed. Generalizes better than GPTQ on instruction-tuned models. | **Implementation & Design** - Saliency-based channel protection is a key data-aware signal; AWQ's scaling approach should be an option in the outlier handling strategy space. |
| **GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers** (Frantar et al., ICLR 2023) | [arXiv](https://arxiv.org/abs/2210.17323) / [OpenReview](https://openreview.net/references/pdf?id=8H9ptvl1Xp) | Layer-wise Hessian-based error compensation via Optimal Brain Quantization. Innovations: arbitrary order quantization, lazy batch updates (128-col blocks), Cholesky decomposition. Quantizes OPT-175B in ~4 GPU hours. | **Implementation & Baseline** - Primary weight-only quantization baseline; GPTQ-quantized models serve as comparison points for SR-JQS joint optimization. |
| **ZeroQuant: Efficient and Affordable Post-Training Quantization** (Yao et al., NeurIPS 2022) | [arXiv](https://arxiv.org/abs/2206.01861) | Proposes layer-by-layer knowledge distillation for PTQ and group-wise quantization. Hardware-friendly W8A8 quantization. | **Baseline** - Comparison baseline for weight+activation quantization; group-wise quantization format is relevant to the config schema. |
| **ATOM: Low-Bit Quantization for Efficient and Accurate LLM Serving** (Zhao et al., MLSys 2024) | [PDF](https://proceedings.mlsys.org/paper_files/paper/2024/file/5edb57c05c81d04beb716ef1d542fe9e-Paper-Conference.pdf) | Serving-oriented quantization combining mixed-precision (outlier channels at higher precision) with optimized CUDA kernels for low-bit GEMMs. Achieves up to 7.7x throughput improvement. | **Design & Implementation** - Directly relevant serving-aware quantization system; kernel design patterns and serving throughput optimization strategies. |
| **Systematic Outliers in Large Language Models** (2025) | [ar5iv](https://arxiv.org/html/2502.06415v2) | Provides a comprehensive analysis of where and why systematic outliers emerge in LLMs, beyond the initial LLM.int8() observations. | **Theory** - Deeper understanding of outlier mechanisms to inform data-aware feature design. |
| **Outliers and Calibration Sets have Diminishing Effect on Quantization of Modern LLMs** (Vlassis et al., 2025) | [arXiv](https://arxiv.org/abs/2405.20835) | Demonstrates that for newer models (Llama-3, Mistral), outlier prevalence is decreasing and calibration set choice matters less. Kurtosis and max-to-median ratio do not reliably predict PTQ accuracy. | **Design** - Critical caveat : activation statistics should be features in a learned predictor, not standalone sensitivity metrics. |
| **Outlier-Safe Pre-Training for Robust 4-Bit Quantization** (2025) | [ar5iv](https://arxiv.org/html/2506.19697v1) | Proposes pre-training modifications that reduce outlier emergence, enabling more robust 4-bit quantization without post-hoc outlier handling. | **Ideas** - Context on how the outlier landscape is evolving; future models may need less aggressive outlier handling. |

---

## 6. KV Cache Quantization

| Paper / Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache** (Liu et al., ICML 2024) | [arXiv](https://arxiv.org/abs/2402.02750) / [ar5iv](https://arxiv.org/html/2402.02750v2) | Demonstrates key-value asymmetry: keys need per-channel quantization (outlier channels), values need per-token quantization. Maintains recent tokens in FP16. Achieves 2-bit with 2.6x less peak memory and 2.35–3.47x throughput gain. | **Implementation & Design** - Primary KV cache quantization approach; per-channel-key/per-token-value asymmetry is a core design principle for the KV cache quantization dimension. |
| **KV Cache is 1 Bit Per Channel: Efficient LLM Inference with Coupled Quantization** (NeurIPS 2024) | [arXiv](https://arxiv.org/abs/2405.03917) / [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/05d6b5b6901fb57d2c287e1d3ce6d63c-Paper-Conference.pdf) | Pushes KV cache to 1-bit per channel via coupled quantization (jointly encoding groups of c contiguous channels). Fisher-guided centroid learning. Only 0.3 perplexity increase at 1-bit with 128 recent FP16 tokens. | **Ideas & Design** - Extreme compression reference point; coupled quantization as an advanced option in the KV cache search space. |
| **KVQuant: Towards 10 Million Context Length LLM Inference** (Hooper et al., NeurIPS 2024) | [PDF](https://people.eecs.berkeley.edu/~ysshao/assets/papers/kvquant-neurips2024.pdf) | Introduces pre-RoPE key quantization (avoids distribution disruption), non-uniform datatypes via Fisher-information-weighted k-means, and attention sink-aware treatment. Enables LLaMA-7B at 1M context on single A100. | **Implementation** - Pre-RoPE quantization and non-uniform datatypes as advanced options; attention sink handling is important for long-context robustness. |
| **Cocktail: Chunk-Adaptive Mixed-Precision Quantization for Long-Context LLM Inference** (2025) | [arXiv](https://arxiv.org/abs/2503.23294) | Chunk-adaptive mixed-precision: query-similar chunks get FP16, medium similarity INT4, low similarity INT2. Chunk-level granularity is more hardware-friendly than token-level. | **Design & Ideas** - Semantic-similarity-based precision allocation for KV cache; chunk-level granularity concept relevant to the runtime adaptation controller. |

---

## 7. Online / Dynamic Adaptation & Robustness

| Paper / Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **Data Quality-aware Mixed-precision Quantization via Hybrid Reinforcement Learning (DQMQ)** (IEEE TNNLS 2024) | [arXiv](https://arxiv.org/abs/2302.04453) | Trains bit-width decision policies conditioned on input quality by relaxing discrete bit-width sampling to continuous probability distributions. Optimized end-to-end via SGD. | **Design & Ideas** - Validates that input-conditioned precision policies are feasible; approach to policy training for the runtime adaptation controller. |
| **Differentiable Dynamic Quantization with Mixed Precision** (Su et al., 2021) | [arXiv](https://arxiv.org/abs/2106.02295) | Proposes dynamic quantization mechanisms where precision can vary per-input at inference time using differentiable gating. | **Ideas** - Dynamic per-input precision gating concept for the runtime switching mechanism. |
| **FlexiQ: Adaptive Mixed-Precision Quantization for Latency/Accuracy Trade-Offs** (2025) | [arXiv](https://arxiv.org/abs/2510.02822) | Runtime adjustment of low-bit ratio for latency/accuracy trade-off. Relevant to mode-switching concepts in SR-JQS. | **Design** - Runtime precision ratio adjustment mechanism applicable to the two-tier adaptation controller. |
| **FlexQuant: A Flexible and Efficient Dynamic Precision Switching Framework for LLM Quantization** (2025) | [ar5iv](https://arxiv.org/html/2506.12024) | Phase-aware switching: W8A8 during prefill, dynamic transition to INT4 during decode based on Perplexity Entropy (PPLE) monitoring. Each layer stores weights in both formats. PPLE computation is negligible. | **Implementation & Design** - Applicable to two-tier runtime adaptation; PPLE as a lightweight runtime signal; dual-format weight storage pattern. |
| **Any-Precision LLM** (Park et al., ICML 2024 Oral) | [arXiv](https://arxiv.org/abs/2402.10517) | Decomposes quantized weights into bit-planes, enabling runtime precision switching by loading only needed planes. Achieves proportional memory bandwidth savings. | **Implementation** - Bitplane weight representation as the storage foundation for multi-mode runtime switching. |
| **AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs** (2025) | [ar5iv](https://arxiv.org/html/2510.10467v1) | Advances bitplane concept with Binary-Coded Quantization and lookup-table-based GEMM. 3.0x throughput over FP16, 1.2x over prior multi-precision methods. | **Implementation** - More efficient multi-precision kernel design; LUT-based GEMM for the runtime backend. |
| **DP-LLM: Dynamic Precision for LLMs** (NeurIPS 2025) | See [related work](https://arxiv.org/html/2506.12024) | Per-layer, per-token dynamic precision using lightweight relative error proxy. Offline-learned thresholds achieve non-integer effective bitwidths (e.g., 3.5-bit). | **Ideas** - Per-token dynamic precision as an advanced option; relative error as a cheap runtime proxy signal. |
| **ButterflyQuant: Ultra-low-bit LLM Quantization through Learnable Orthogonal Butterfly Transforms** (2025) | [ar5iv](https://arxiv.org/html/2509.09679) | Uses learnable orthogonal rotations (butterfly transforms) to reduce activation kurtosis by 62%, enabling simpler quantization. | **Ideas** - Rotation-based outlier reduction as a pre-quantization transform option in the config schema. |

---

## 8. Evaluation, Surveys & Benchmarking Methodology

| Paper / Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **A Comprehensive Evaluation of Quantization Strategies for Large Language Models** (2024) | [arXiv](https://arxiv.org/abs/2402.16775) | Systematic evaluation across quantization methods, bit-widths, and model sizes. Covers perplexity, downstream tasks, and practical deployment. | **Design & Evaluation** - Evaluation methodology reference; baseline results to compare our implementation against; identifies which tasks are most sensitive. |
| **Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models** (Liu et al., 2025) | [arXiv](https://arxiv.org/abs/2504.04823) | First systematic study of quantization on reasoning LLMs. W8A8 is near-lossless; 4-bit activations cause >10% drops for <32B models. Math reasoning and code gen are most sensitive; knowledge recall is most robust. | **Design & Evaluation** - Critical for quality constraint design; identifies which task categories need hard constraints vs. soft monitoring in our quality evaluation suite. |
| **A Survey of Low-bit Large Language Models: Basics, Systems, and Algorithms** (2024) | [arXiv](https://arxiv.org/abs/2409.16694) | Comprehensive survey covering formats, kernels, and systems co-design for low-bit LLMs. | **Background** - Broad overview to ensure we dont miss relevant techniques; systems co-design section is particularly relevant. |
| **Mixed-Precision Quantization for Language Models: Techniques and Prospects** (2025) | [ar5iv](https://arxiv.org/html/2510.16805v1) | Recent survey specifically focused on mixed-precision quantization for language models, covering RL-based, differentiable, and Hessian-based approaches. | **Background & Design** - Up-to-date landscape of mixed-precision approaches; identifies open problems. |
| **Beyond Outliers: A Study of Optimizers Under Quantization** (Vlassis et al., 2025) | [arXiv](https://www.arxiv.org/pdf/2509.23500) | Demonstrates that kurtosis and max-to-median ratio do not reliably predict PTQ accuracy across optimizers. Error accumulation through the network is the dominant factor. | **Design** - Important negative result: should use activation statistics as features in a learned predictor, not as standalone sensitivity metrics. |

---

## 9. Serving Backends: Documentation & Guides

| Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **vLLM Quantization Documentation** | [Docs](https://docs.vllm.ai/en/latest/features/quantization/) | Official documentation for all supported quantization formats in vLLM (GPTQ, AWQ, FP8, INT8 SmoothQuant, bitsandbytes, etc.). Includes custom quantization registration API. | **Implementation** - Primary reference for integrating configs with vLLM; custom `@register_quantization_config` decorator. |
| **vLLM FP8 W8A8 Guide** | [Docs](https://docs.vllm.ai/en/stable/features/quantization/fp8/) | Detailed guide for FP8 quantization in vLLM, including KV cache FP8 configuration. | **Implementation** - FP8 KV cache setup (`--kv-cache-dtype fp8_e4m3`). |
| **vLLM Benchmarking (bench serve)** | [Docs](https://docs.vllm.ai/en/latest/api/vllm/benchmarks/serve/) | Built-in benchmark tool measuring throughput, TTFT, TPOT, ITL at p50/p99, and goodput. | **Implementation** - Core benchmarking integration point for our benchmarking harness. |
| **vLLM Disaggregated Prefilling** | [Docs](https://docs.vllm.ai/en/latest/features/disagg_prefill/) | Experimental feature for running separate prefill and decode instances. | **Ideas** - Advanced deployment option enabling phase-specific quantization strategies. |
| **vLLM Sleep Mode (Zero-Reload Model Switching)** | [Blog](https://blog.vllm.ai/2025/10/26/sleep-mode.html) | JIT-compiled kernels and CUDA graphs preserved across model switches, enabling fast model/config cycling. | **Implementation** - Key enabler for runtime mode switching without kernel rebuild overhead. |
| **TensorRT-LLM Numerical Precision Reference** | [Docs](https://nvidia.github.io/TensorRT-LLM/reference/precision.html) | Official reference for all supported numerical precisions in TRT-LLM. | **Implementation & Constraints** - Defines executable format space for TRT-LLM backend. |
| **TensorRT-LLM Quantization Guide** | [GitHub](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/quantization-in-TRT-LLM.md) / [Rendered](https://nvidia.github.io/TensorRT-LLM/blogs/quantization-in-TRT-LLM.html) | Comprehensive quantization blog covering INT8 SQ, FP8, INT4 AWQ, W4A8, and AutoQuantize per-layer mixed precision. | **Implementation** - AutoQuantize API is the most mature per-layer mixed-precision system; provides useful integration patterns. |
| **TensorRT-LLM Quantization Examples** | [GitHub](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/quantization/README.md) | Step-by-step quantization examples with ModelOpt. | **Implementation** - Runnable code examples for TRT-LLM quantization workflows. |
| **NVIDIA ModelOpt (TensorRT Model Optimizer) LLM PTQ** | [GitHub](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md) | ModelOpt's PTQ pipeline for LLMs: calibration, quantization, and export to TRT-LLM engines. Supports FP8, INT8 SQ, INT4 AWQ, and mixed-precision AutoQuantize. | **Implementation** - Primary PTQ toolchain for NVIDIA backends; `mtq.quantize()` and `mtq.auto_quantize()` APIs. |
| **SGLang Deployment & Quantization** | [DeepWiki](https://deepwiki.com/sgl-project/sglang/6-deployment-and-operations) | SGLang's quantization support (15+ methods) and deployment configuration. | **Implementation** - Alternative backend option with broad format support. |
| **HuggingFace TGI Documentation** | [Docs](https://huggingface.co/docs/inference-endpoints/en/engines/tgi) | TGI engine documentation. Note: TGI is in maintenance mode as of Dec 2025. | **Reference** - Useful for comparison, but TGI is not recommended as our primary backend. |
| **vLLM Quantization Benchmarks (Jarvislabs)** | [Blog](https://docs.jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) | Third-party comprehensive benchmarks comparing quantization methods in vLLM. Demonstrates batch size-dependent optimal strategies and Marlin kernel impact (AWQ+Marlin: 741 tok/s vs 67 without). | **Evaluation** - Baseline benchmark data; batch-size-dependent quantization strategy insights. |

---

## 10. Benchmarking & Performance Measurement Tools

| Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **NVIDIA GenAI-Perf / AIPerf** | [Blog](https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/) | NVIDIA's official LLM benchmarking tool with GPU telemetry integration. Measures TTFT, ITL, throughput, and latency distributions. ITL formula explicitly excludes TTFT. | **Implementation** - Reference benchmarking tool; metric definitions for our benchmarking harness. |
| **NVIDIA NIM LLM Benchmarking Metrics** | [Docs](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) | Official metric definitions: TTFT, ITL, TPOT, request latency, throughput, and goodput. | **Implementation** - Canonical metric definitions to ensure our benchmarks are industry-comparable. |
| **LLM Inference Benchmarking: Fundamental Concepts (NVIDIA)** | [Blog](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/) | Deep dive into LLM benchmarking concepts: prefill vs decode, compute-bound vs memory-bound regimes, and how batch size affects optimality. | **Design & Theory** - Essential background for designing the workload specification module. |
| **Benchmarking LLM Serving Performance: A Comprehensive Guide** (Medium) | [Blog](https://medium.com/@kimdoil1211/benchmarking-llm-serving-performance-a-comprehensive-guide-db94b1bfe8cf) | Practical guide covering measurement pitfalls, warm-up protocols, and reproducibility best practices. | **Implementation** - Measurement protocol best practices for the benchmarking harness. |
| **lm-evaluation-harness** (EleutherAI) | [GitHub](https://github.com/EleutherAI/lm-evaluation-harness) / [PyPI](https://pypi.org/project/lm-eval/) | Standard framework for LLM quality evaluation. 60+ benchmarks, supports vLLM/SGLang/HF backends. Includes MMLU, GSM8K, HellaSwag, HumanEval, etc. | **Implementation** - Core quality evaluation tool; integrates directly with serving backends. |
| **MLPerf Inference** | [Site](https://mlcommons.org/benchmarks/inference-datacenter/) | Industry-standard inference benchmarking with p99 TTFT/TPOT thresholds. | **Evaluation** - Reference for SLO-based evaluation; ensures our benchmarks meet industry standards. |

---

## 11. MASE Framework & Quantization Search Infrastructure

| Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **MASE Documentation (Main)** | [Docs](https://deepwok.github.io/mase/index.html) | Official MASE documentation. PyTorch FX-based ML compiler with 44+ composable passes, supporting integer, Microscaling (MX), and FP8 quantization. | **Implementation** - Core framework; quantization transform passes, graph representation, and search orchestration. |
| **MASE Tutorial 5: NAS with Optuna** | [Tutorial](https://deepwok.github.io/mase/modules/documentation/tutorials/tutorial_5_nas_optuna.html) | Tutorial demonstrating Optuna-based search integration with MASE. Covers TPE, NSGA-II, QMC, and random search. | **Implementation** - Direct guide for setting up the search loop; TPE found most effective for mixed-precision. |
| **MASE GitHub Repository** | [GitHub](https://github.com/DeepWok/mase) | Source code for MASE. Includes quantization passes, compression pipeline, and search infrastructure. | **Implementation** - Codebase to extend; custom pass and objective development. |
| **MASE Paper (Cheng et al.)** | [arXiv](https://arxiv.org/abs/2307.15517) | MASE system paper describing the MaseGraph IR, type-independent pass system, and hardware cost modeling approach. | **Theory & Design** - Understand MASE's design philosophy and extension points; cost model architecture for hardware-aware search. |

---

## 12. Optimization & Surrogate Modeling Tools

| Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **BoTorch: Multi-Objective Bayesian Optimization** | [Docs](https://botorch.org/docs/multi_objective/) | PyTorch-based BO library. qLogNEHVI acquisition function for Pareto frontier construction. Handles mixed discrete-continuous spaces. | **Implementation** - Core optimization engine for offline optimizer; accuracy-latency-memory Pareto frontier construction. |
| **Ax Platform** | [Docs](https://ax.dev/) | High-level experimentation platform built on BoTorch. Native mixed discrete-continuous space support, outcome constraints, multi-objective optimization. Best for mixed/discrete, constrained, noisy problems. | **Implementation** - Recommended high-level API wrapping BoTorch; `ax_client.create_experiment()` with ordered choice parameters for bit-widths. |
| **Optuna** | [Docs](https://optuna.org/) | Hyperparameter optimization framework. TPE sampler, multi-objective, pruning. Already integrated with MASE. | **Implementation** - Already part of MASE; useful for initial search experiments before migrating to Ax/BoTorch for serving-real objectives. |
| **Transfer Learning of Surrogate Models via Domain Affine Transformation** (2025) | [ar5iv](https://arxiv.org/html/2501.14012) | Demonstrates transfer learning techniques for surrogates across different search benchmarks. Reduces data requirements to 50-100 transfer samples. | **Ideas** - Transfer learning approach for reusing surrogates across model sizes in our progressive scaling strategy. |
| **Weights & Biases (W&B)** | [Site](https://wandb.ai/) | Experiment tracking platform. Structured logging, hyperparameter sweeps, Pareto visualization. | **Implementation** - Experiment tracking for all quantization search runs, config logging, and artifact management. |
| **MLflow** | [Site](https://mlflow.org/) | Open-source experiment tracking. Model registry, artifact storage, metric comparison. | **Implementation** - Alternative to W&B for experiment tracking; better for self-hosted setups. |

---

## 13. Hardware Architecture & GPU Documentation

| Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **NVIDIA Hopper Architecture In-Depth** | [Blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) | Detailed technical overview of H100: FP8 Tensor Cores via Transformer Engine, 3.35 TB/s HBM3 bandwidth, 3958 FP8 TOPS (sparse). | **Design & Constraints** - Hardware capabilities that define the quantization format space; FP8 availability on Hopper only. |
| **NVIDIA A100 vs H100 vs H200 Comparison** | [Blog](https://www.e2enetworks.com/blog/nvidia-a100-vs-h100-vs-h200-gpu-comparison) | Side-by-side comparison of datacenter GPU specs: memory bandwidth, compute throughput, quantization support. | **Design** - Hardware constraint tables for the configuration schema; memory budget calculations. |
| **Best GPUs for AI (2026): Quantization Support** | [Blog](https://www.bestgpusforai.com/blog/best-gpus-for-ai) | Overview of quantization kernel support across NVIDIA architectures from Turing to Blackwell. | **Reference** - Quick reference for which formats are accelerated on which hardware. |
| **Mastering LLM Techniques: Inference Optimization (NVIDIA)** | [Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) | Comprehensive guide covering KV caching, batching, quantization, and memory optimization for LLM inference. | **Background** - End-to-end inference optimization context; helps understand how quantization interacts with other serving optimizations. |
| **Optimizing LLMs with Post-Training Quantization (NVIDIA)** | [Blog](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/) | NVIDIA's official PTQ guide: FP8, INT8 SmoothQuant, INT4 AWQ workflows with ModelOpt. | **Implementation** - Step-by-step PTQ workflows for NVIDIA hardware; calibration best practices. |
| **TensorRT-LLM Optimization Stack Guide (Introl)** | [Blog](https://introl.com/blog/tensorrt-llm-optimization-nvidia-inference-stack-guide) | Third-party deep dive into TRT-LLM optimization, including quantization, batching, and kernel selection. | **Implementation** - Practical tips for TRT-LLM deployment and optimization. |

---

## 14. PyTorch Quantization & Low-Level Tools

| Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **PyTorch Quantization Flow Clarification (torchao)** | [Forum](https://dev-discuss.pytorch.org/t/clarification-of-pytorch-quantization-flow-support-in-pytorch-and-torchao/2809) | Official PyTorch team post clarifying migration from `torch.ao.quantization` to `torchao`. `quantize_()` API with `torch.compile` integration. | **Implementation** - Understanding the current PyTorch quantization API; torchao is the future-proof choice. |
| **AutoGPTQ** | [GitHub](https://github.com/AutoGPTQ/AutoGPTQ) / [HF Blog](https://huggingface.co/blog/gptq-integration) | Library for GPTQ quantization with HuggingFace integration. | **Implementation** - Tool for generating GPTQ baselines and comparison points. |
| **bitsandbytes** | [GitHub](https://github.com/bitsandbytes-foundation/bitsandbytes) / [HF Blog](https://huggingface.co/blog/hf-bitsandbytes-integration) | NF4/FP4/8-bit quantization library. Simple API for loading models in reduced precision. | **Implementation** - Quick baseline quantization; NF4 format for QLoRA-style experiments. |
| **Intel Neural Compressor SmoothQuant** | [Docs](https://intel.github.io/neural-compressor/latest/docs/source/smooth_quant.html) | Reference implementation of SmoothQuant in Intel Neural Compressor with configurable a parameter. | **Implementation** - Reference SmoothQuant implementation for cross-validation. |

---

## 15. Emerging Architectures & Optional Extensions

| Paper / Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **Automated Fine-Grained Mixture-of-Experts Quantization** (ACL 2025) | [ACL Anthology](https://aclanthology.org/2025.findings-acl.1386/) | MoE-specific sensitivity analysis and automated precision allocation. Handles expert-level heterogeneity in quantization sensitivity. | **Extension** - Optional scope: if we extend to MoE models, this paper provides the sensitivity analysis framework. |

---

## 16. Additional Informational Resources

| Resource | Link | Description | Useful For |
|:---|:---|:---|:---|
| **LLM Quantization Overview (BentoML Handbook)** | [Blog](https://bentoml.com/llm/getting-started/llm-quantization) | Accessible overview of LLM quantization methods (GPTQ, AWQ, SmoothQuant) with practical deployment guidance. | **Background** - Good introductory resource; practical deployment considerations. |
| **Which Quantization Method? GGUF vs GPTQ vs AWQ (E2E Networks)** | [Blog](https://www.e2enetworks.com/blog/which-quantization-method-is-best-for-you-gguf-gptq-or-awq) | Practical comparison of popular quantization formats with pros/cons and use-case guidance. | **Background** - Quick reference for format selection rationale. |
| **SmoothQuant Explainer (Substack)** | [Blog](https://aakashvarma.substack.com/p/smoothquant) | Detailed walkthrough of SmoothQuant's per-channel activation smoothing mechanism. | **Background** - Accessible explanation of the smoothing transformation math. |
| **Understanding GPTQ Algorithm Mechanics (APXML)** | [Course](https://apxml.com/courses/practical-llm-quantization/chapter-3-advanced-ptq-techniques/gptq-mechanics) | Step-by-step explanation of GPTQ's Optimal Brain Quantization process with visual illustrations. | **Background** - Helpful for understanding GPTQ internals when implementing the candidate builder. |
| **Dynamic Quantization Methods Overview (Emergent Mind)** | [Overview](https://www.emergentmind.com/topics/dynamic-quantization-methods) | Aggregated overview of dynamic/runtime quantization approaches with paper links. | **Background** - Quick landscape scan of dynamic quantization literature. |