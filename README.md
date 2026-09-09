# EdgeCore

### Hardware-Adaptive Runtime & Deployment Compiler for Sovereign Small Language Models

> **Model + Hardware + Workload + Constraints → Search → Benchmark → Verify → Deploy**

EdgeCore is a CPU-first inference optimization and deployment system for Small Language Models (SLMs).

Instead of assuming that one runtime configuration is optimal everywhere, EdgeCore treats deployment as a **hardware- and workload-dependent optimization problem**.

Given a model, target hardware, workload, and deployment constraints, EdgeCore:

1. analyzes the model,
2. profiles the target hardware,
3. constructs candidate execution configurations,
4. benchmarks candidates on the real machine,
5. filters configurations against hard constraints,
6. verifies correctness and model quality,
7. selects a deployable configuration,
8. produces a reproducible deployment package.

The goal is not simply to make inference faster.

The goal is to answer:

> **"For this model, on this machine, under these constraints, what configuration should actually be deployed—and can we prove that it satisfies the requirements?"**

---

## Why EdgeCore?

Small Language Models are increasingly attractive for local, private, and resource-constrained inference.

However, deployment performance is not determined by model size alone.

It depends on the interaction between:

```text
Model
  ×
Hardware
  ×
Workload
  ×
Runtime Configuration
  ×
Deployment Constraints
```

Parameters such as quantization, threading, kernel implementation, memory layout, KV-cache behavior, and workload characteristics can significantly affect latency, throughput, memory consumption, and model quality.

Traditional deployment often looks like:

```text
Engineer
   ↓
Choose runtime
   ↓
Choose quantization
   ↓
Choose thread count
   ↓
Benchmark
   ↓
Change configuration
   ↓
Benchmark again
   ↓
Repeat
```

EdgeCore attempts to turn that process into an explicit optimization pipeline:

```text
Model + Hardware + Workload + Constraints
                    ↓
              EdgeCore Search
                    ↓
          Real Hardware Benchmarking
                    ↓
          Constraint Filtering
                    ↓
              Verification
                    ↓
       Recommended Deployment Config
                    ↓
          Reproducible Package
```

The project is particularly motivated by CPU-first, on-premise, offline, and sovereign AI scenarios where predictable resource usage, reproducibility, and deployment constraints matter.

---

# Architecture

```text
                         ┌───────────────────────┐
                         │      Model Artifact   │
                         │    HF / GGUF / ...    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │     Model Analyzer    │
                         │                       │
                         │ • Architecture       │
                         │ • Layers              │
                         │ • Tensor shapes       │
                         │ • Parameters          │
                         │ • Attention / GQA     │
                         │ • KV dimensions       │
                         │ • Memory estimates    │
                         └───────────┬───────────┘
                                     │
                                     │
                         ┌───────────▼───────────┐
                         │    Hardware Profiler  │
                         │                       │
                         │ • CPU                 │
                         │ • ISA / AVX2          │
                         │ • Cores / threads     │
                         │ • Cache information   │
                         │ • System memory       │
                         └───────────┬───────────┘
                                     │
                                     ▼
                 ┌─────────────────────────────────────┐
                 │       EdgeCore Optimization         │
                 │                                     │
                 │ • Precision                         │
                 │ • Kernel                            │
                 │ • Thread count                      │
                 │ • Batch size                        │
                 │ • KV-cache strategy                  │
                 │ • Memory/runtime configuration       │
                 └──────────────────┬──────────────────┘
                                    │
                                    ▼
                 ┌─────────────────────────────────────┐
                 │          Auto-Tuning / Search        │
                 │                                     │
                 │ Generate → Benchmark → Rank         │
                 │                                     │
                 │ Multi-objective evaluation          │
                 └──────────────────┬──────────────────┘
                                    │
                                    ▼
                 ┌─────────────────────────────────────┐
                 │         C/C++ Inference Runtime     │
                 │                                     │
                 │ • Qwen2 execution                   │
                 │ • FP16 / INT8 paths                 │
                 │ • AVX2                              │
                 │ • SIMD kernels                      │
                 │ • KV cache                          │
                 │ • Arena-style memory management     │
                 └──────────────────┬──────────────────┘
                                    │
                                    ▼
                 ┌─────────────────────────────────────┐
                 │       Verification & Evaluation      │
                 │                                     │
                 │ • Correctness                       │
                 │ • Tokenizer agreement               │
                 │ • Accuracy constraints              │
                 │ • TTFT / TPOT                       │
                 │ • Throughput                        │
                 │ • P50 / P95 / P99                   │
                 │ • Memory                            │
                 │ • Regression checks                 │
                 └──────────────────┬──────────────────┘
                                    │
                                    ▼
                 ┌─────────────────────────────────────┐
                 │        Deployment Packager          │
                 │                                     │
                 │ • Model                             │
                 │ • Runtime                           │
                 │ • Configuration                     │
                 │ • Benchmark results                 │
                 │ • Verification metadata             │
                 │ • Manifest                          │
                 └──────────────────┬──────────────────┘
                                    │
                                    ▼
                         ┌───────────────────────┐
                         │ Offline / On-Premise │
                         │      Deployment      │
                         └───────────────────────┘
```

This architecture follows the project's core idea: combine model analysis, hardware profiling, optimization, automatic search, measurement, verification, and reproducible deployment into one system.

---

# Current Implementation

EdgeCore is currently implemented as a **CPU-first x86/AVX2 prototype**.

### Model support

Current end-to-end validation focuses on:

* **Qwen2.5-0.5B-Instruct**
* Qwen2 architecture
* Hugging Face model artifacts
* Qwen tokenizer and chat-template verification
* FP16 execution
* INT8 execution paths

The initial model choice follows the project's experimental scope: Qwen2.5-0.5B provides a manageable model for rapid experimentation and reproducible evaluation.

### Runtime

The inference runtime is implemented in C/C++ and includes:

* Qwen2 forward pass
* RMSNorm
* RoPE
* Grouped Query Attention
* SwiGLU
* KV cache
* FP16 execution
* INT8 execution paths
* AVX2-oriented CPU execution
* configurable threading
* arena-style memory management

### Toolchain

```text
edgecore/
├── analyze.py
├── bench.py
├── cli.py
├── ecm.py
├── export.py
├── gguf.py
├── package.py
├── reference.py
├── tokenizer.py
└── verify.py

src/
├── gpt2.c
├── runtime_main.c
├── kernels.c
├── tokenizer.c
├── hw_profiler.c
└── edgecore.h

benchmarks/
├── common/
├── pytorch/
├── llamacpp/
├── edgecore/
└── results/

models/
scripts/
deployment/
```

---

# End-to-End Workflow

A typical EdgeCore deployment follows:

```text
1. MODEL INPUT
       ↓
2. MODEL ANALYSIS
       ↓
3. HARDWARE PROFILING
       ↓
4. SEARCH SPACE GENERATION
       ↓
5. CANDIDATE BENCHMARKING
       ↓
6. CONSTRAINT FILTERING
       ↓
7. QUALITY / CORRECTNESS VERIFICATION
       ↓
8. BEST VALID CONFIGURATION
       ↓
9. DEPLOYMENT PACKAGE
```

For example:

```text
Model:
  Qwen2.5-0.5B-Instruct

Hardware:
  Intel Core i3-1115G4
  AVX2
  4 logical CPUs
  8 GB RAM

Workload:
  batch = 1
  context = 2048
  generation = 32 tokens

Constraints:
  RAM < 6 GB
  P95 < 100 ms
  accuracy loss < 1%
```

EdgeCore then evaluates candidate configurations rather than assuming that the fastest raw configuration is automatically deployable.

---

# A Key Design Principle: Fastest ≠ Deployable

One of the most important results from the project so far is that the fastest configuration is not necessarily the correct deployment.

Example:

```text
Configuration     tok/s     P95        Accuracy loss    Decision
----------------------------------------------------------------
FP16 / 2 threads  15.6      83.3 ms    0%               ACCEPT
INT8 / 2 threads  24.5      44.3 ms    1.70%            REJECT
```

The INT8 configuration is substantially faster.

But if the deployment requirement is:

```text
Accuracy loss < 1%
```

then deploying INT8 would be incorrect.

EdgeCore therefore separates:

```text
BEST MEASURED
```

from:

```text
BEST VALID
```

A configuration must satisfy the deployment constraints before it can be packaged as a verified deployment.

This distinction is now enforced by the deployment decision engine.

---

# Verification

EdgeCore does not treat benchmarking as sufficient evidence for deployment.

The verification layer checks:

### Correctness

* runtime output consistency
* numerical agreement
* reference comparisons

### Tokenization

Qwen tokenizer verification uses a reference tokenizer and compares token IDs directly.

Current tokenizer verification covers:

* vocabulary metadata
* plain-text tokenization
* Qwen chat-template formatting
* exact token-ID agreement

Current result:

```text
8 / 8 tokenizer verification cases passed
Token-ID agreement: 100%
```

### Performance

Candidate configurations are evaluated using measured:

* TTFT
* TPOT
* tokens/sec
* P50 latency
* P95 latency
* P99 latency
* peak memory

### Constraint verification

For example:

```text
Requirement:
  P95 < 100 ms
  RAM < 6 GB
  Accuracy loss < 1%

Measured:
  P95 = X ms
  RAM = X GB
  Accuracy loss = X%

STATUS:
  VERIFIED / NO_VALID_CONFIGURATION
```

The deployment packager is gated by this decision.

A failed or `NO_VALID_CONFIGURATION` result cannot silently become a deployment package.

---

# Real Hardware Benchmark

All performance measurements below were obtained on the development machine rather than inferred from theoretical FLOPs.

### Hardware

```text
Acer TravelMate P214-53

CPU:
  Intel Core i3-1115G4

ISA:
  AVX2

Logical CPUs:
  4

RAM:
  8 GB

OS:
  Ubuntu 24.04
```

### Qwen2.5-0.5B-Instruct

Workload:

```text
batch       = 1
context     = 2048
generation  = 32 tokens
```

Representative EdgeCore search:

```text
Precision   Threads    tok/s     P95
------------------------------------------
FP16        1          10.2      106.9 ms
FP16        2          15.6       83.3 ms
FP16        4           9.6      121.9 ms

INT8        1          16.9       63.7 ms
INT8        2          24.5       44.3 ms
INT8        4          14.8       88.3 ms
```

With the project's constraint set:

```text
P95 < 100 ms
RAM < 6 GB
Accuracy loss < 1%
```

EdgeCore selected:

```text
Precision: FP16
Threads:   2
Batch:     1

STATUS: VERIFIED
```

The interesting hardware result is that **more threads were not always faster**.

On this CPU, 2 threads outperformed 4 threads for this workload, demonstrating why runtime configuration cannot safely be hard-coded from core count alone.

---

# Baseline Comparison

EdgeCore is evaluated against real inference baselines rather than only synthetic or weak comparisons.

The current comparison includes:

* PyTorch CPU
* llama.cpp
* EdgeCore

Representative measurements:

```text
Runtime      Threads    tok/s     P95
-------------------------------------------
PyTorch      1           1.2      536.2 ms
PyTorch      2           1.1      589.9 ms
PyTorch      4           0.9     1721.3 ms

llama.cpp    1          11.9       84.7 ms
llama.cpp    2          11.9       86.6 ms
llama.cpp    4          11.7       90.7 ms

EdgeCore     1           5.4      107.8 ms
EdgeCore     2           8.1      101.5 ms
EdgeCore     4           5.7      247.6 ms
```

### What this tells us

EdgeCore is already substantially faster than the PyTorch CPU baseline on this workload.

However:

> **EdgeCore does not currently beat llama.cpp.**

That is an intentional and important finding, not something hidden by the project.

The current performance gap is one of the primary motivations for the next optimization phase.

---

# Project Post-Mortem

## What went wrong

Building a complete inference/deployment pipeline exposed several problems that were not obvious from the initial architecture.

### 1. Chat-template mismatch

The first Qwen2.5-0.5B-Instruct generation tests produced incorrect-looking output.

The underlying issue was not simply the model runtime.

The instruct model expects prompts formatted according to its chat template.

Passing raw user text directly into the model produced a mismatch between the expected prompt representation and the actual token sequence.

### Lesson

For instruction-tuned models:

```text
Model correctness
        ≠
Forward-pass correctness alone
```

The tokenizer, special tokens, chat template, and runtime must agree.

This led to the tokenizer verification system that now compares EdgeCore token IDs against a reference tokenizer.

---

## 2. Initial benchmark measurement was misleading

The first llama.cpp comparison produced suspicious results, including:

* unrealistically low P95 latency
* incorrect memory numbers
* TTFT/TPOT values that were not directly comparable

Investigation showed that parts of the benchmark were measuring different execution boundaries.

For example, llama.cpp TTFT and TPOT were initially derived from separate runs, while total throughput included model loading.

### Lesson

Benchmarking inference systems is itself a systems problem.

Metrics are only meaningful when:

```text
measurement boundary
+
warm/cold state
+
process lifetime
+
token count
+
memory accounting
```

are explicitly defined.

The benchmark harness was subsequently corrected and methodology was documented rather than hiding the original result.

---

## 3. Fastest configuration failed quality requirements

INT8 produced a significant performance improvement.

But the measured accuracy degradation exceeded the deployment requirement:

```text
INT8 accuracy loss ≈ 1.70%

Required:
accuracy loss < 1%
```

Therefore:

```text
FASTER
```

did not mean:

```text
DEPLOYABLE
```

### Lesson

Optimization must be constraint-aware.

A deployment compiler should not optimize a metric while silently violating another requirement.

---

## 4. "More CPU threads" did not guarantee better performance

The experiments showed:

```text
2 threads > 4 threads
```

for the tested workload.

This reinforced the central hypothesis of EdgeCore:

> Hardware characteristics provide useful search boundaries, but empirical measurement is still required to select the final configuration.

---

## 5. A technically valid result can still be operationally invalid

At one stage, EdgeCore produced an apparent "accepted" result around:

```text
P95 ≈ 101.5 ms
```

while the target requirement was:

```text
P95 < 100 ms
```

The decision logic was later tightened so that constraints are evaluated explicitly rather than inferred from ranking.

The correct outcome became:

```text
NO_VALID_CONFIGURATION
```

when no candidate satisfies the hard constraints.

### Lesson

A deployment system must be conservative at the final gate.

It is better to say:

```text
NO_VALID_CONFIGURATION
```

than to produce a package that violates the stated requirements.

---

# What I Learned

The most important lessons from building EdgeCore were not specific to one model or one kernel.

### Systems lesson

Inference performance emerges from interactions between:

```text
model architecture
hardware
memory hierarchy
precision
kernels
threading
workload
runtime configuration
```

### Measurement lesson

A benchmark is only useful when its measurement methodology is explicit and reproducible.

### ML systems lesson

Quantization is not automatically an optimization.

It is an optimization only if:

```text
performance gain
+
acceptable quality
+
resource constraints
```

are all satisfied.

### Compiler/runtime lesson

The interesting problem is not:

> "How do I make this kernel faster?"

It is:

> "How do I systematically discover the best execution strategy for this model on this machine under a real deployment objective?"

That is the problem EdgeCore is designed around.

---

# Current Status

### Completed

* [x] Model analysis pipeline
* [x] Hardware profiling
* [x] Qwen2.5-0.5B end-to-end support
* [x] Qwen2 architecture execution
* [x] FP16 execution
* [x] INT8 execution path
* [x] AVX2 CPU execution
* [x] Configurable threading
* [x] Benchmark harness
* [x] PyTorch comparison
* [x] llama.cpp comparison
* [x] Qwen tokenizer verification
* [x] Chat-template verification
* [x] Automatic configuration decision engine
* [x] Explicit constraint filtering
* [x] `NO_VALID_CONFIGURATION` state
* [x] Deployment package gating
* [x] Verification metadata
* [x] Reproducible deployment artifacts

### In progress

* [ ] Improve EdgeCore runtime performance
* [ ] Profile operator-level bottlenecks
* [ ] Optimize critical kernels
* [ ] Improve auto-tuning/search efficiency
* [ ] Reduce gap to mature CPU inference runtimes
* [ ] Expand Pareto-style configuration selection

---

# Upgrade Roadmap

## Upgrade 1 — End-to-End Qwen Deployment

Completed.

Demonstrated:

```text
Model
 ↓
Analysis
 ↓
Hardware profile
 ↓
Configuration search
 ↓
Benchmark
 ↓
Verification
 ↓
Deployment package
```

---

## Upgrade 2 — Real Runtime Comparison

Completed.

Added controlled comparisons against:

```text
PyTorch CPU
llama.cpp
EdgeCore
```

This established a realistic performance baseline.

---

## Upgrade 3 — Verification & Deployment Gating

Completed.

Added:

```text
Tokenizer verification
        +
Chat-template verification
        +
Constraint engine
        +
Deployment gating
```

The system now distinguishes:

```text
BEST MEASURED
```

from:

```text
BEST VALID
```

and blocks deployment when no valid configuration exists.

---

## Upgrade 4 — Performance Optimization & Auto-Tuning

Next major focus.

The objective is not to optimize blindly.

The process will be:

```text
Profile
  ↓
Identify hotspots
  ↓
Optimize critical path
  ↓
Benchmark
  ↓
Verify correctness
  ↓
Measure regression
  ↓
Keep / Reject optimization
```

The current gap to llama.cpp provides a concrete optimization target.

Priority areas include:

* matrix/vector kernels
* memory movement
* cache behavior
* RMSNorm
* attention path
* KV-cache access
* SwiGLU
* output projection
* threading overhead
* tensor layouts

The intent is to make EdgeCore increasingly competitive while preserving its hardware-adaptive and verification-driven design.

---

# What EdgeCore Is — and Is Not

### EdgeCore is:

* a hardware-aware SLM deployment system
* a runtime configuration search layer
* a benchmarking-driven optimizer
* a verification and deployment pipeline
* a systems research prototype
* a foundation for CPU-first sovereign/offline inference

### EdgeCore is not currently:

* a universal inference engine
* a replacement for llama.cpp
* a production enterprise deployment platform
* a GPU inference framework
* a multi-architecture compiler

That distinction is deliberate.

The current implementation focuses on **x86/AVX2** to keep experimentation measurable and reproducible. The broader architecture can later support additional backends.

---

# Research Direction

EdgeCore is influenced by the idea of hardware-measured automatic optimization used in systems such as Ansor and TVM.

The project's specific direction is to apply this philosophy to the broader deployment problem:

```text
Model
×
Hardware
×
Workload
×
Deployment Constraints
```

rather than optimizing a single operator or assuming a fixed runtime configuration.

The intended output is therefore not simply:

```text
"this configuration is fast"
```

but:

```text
"this configuration is valid for this deployment objective,
was measured on this hardware,
passed the required verification checks,
and can be reproduced from the generated deployment artifact."
```

---

# Future Direction

Potential future extensions include:

* ARM/NEON backend
* AVX512/VNNI optimization
* GPU backend
* NUMA-aware optimization
* learned cost models
* advanced quantization strategies
* speculative decoding
* model/runtime co-design
* fleet-level deployment profiling
* offline/air-gapped deployment management

These are intentionally outside the current MVP scope.

---

# Reproducibility

Every verified deployment is intended to carry the information necessary to understand how the decision was made.

A deployment package contains artifacts such as:

```text
deployment/
├── model/
├── runtime/
├── config/
├── benchmark.json
├── hardware.json
├── verification.json
└── manifest.json
```

This follows the project's objective of producing reproducible deployment artifacts suitable for constrained or offline environments.

---

# Quick Start

```bash
# Build the runtime
make

# Inspect available commands
python3 -m edgecore.cli --help

# Analyze a model
python3 -m edgecore.cli analyze ...

# Export model
python3 -m edgecore.cli export ...

# Run verification
python3 -m edgecore.cli verify ...

# Run deployment workflow
python3 -m edgecore.cli deploy-qwen ...
```

See the individual scripts and benchmark directories for the exact experiment commands used to generate the reported results.

---

# Why This Project Exists

EdgeCore started as a B.Tech major project.

It evolved into a systems question:

> **Can model deployment be treated as an automated, measurable, and verifiable optimization problem instead of a collection of manually chosen runtime parameters?**

The current implementation is an attempt to answer that question with a working system rather than only a design document.

The project deliberately documents both successful results and failed experiments because the failures exposed important engineering constraints around benchmarking, tokenizer correctness, quantization quality, and deployment validation.

---

# Status

```text
PROJECT STATUS

Architecture              ████████████████████  Implemented
Model analysis            ████████████████████  Implemented
Hardware profiling        ████████████████████  Implemented
Qwen2.5-0.5B runtime      ████████████████████  Implemented
Benchmark framework       ████████████████████  Implemented
Tokenizer verification    ████████████████████  Implemented
Decision engine            ████████████████████  Implemented
Deployment gating         ████████████████████  Implemented

Runtime optimization     ███████████░░░░░░░░░  In progress
Auto-tuning maturity      █████████░░░░░░░░░░░  In progress
Multi-architecture        ██░░░░░░░░░░░░░░░░░░  Future
Production hardening      ███░░░░░░░░░░░░░░░░░  Future
```

**Current focus: Upgrade 4 — making the runtime faster without compromising correctness or deployment constraints.**

---

# Author

**Mahesh Jakkala**

B.Tech Major Project

**EdgeCore — Hardware-Adaptive Runtime & Deployment Compiler for Sovereign Small Language Models**

---

## Closing Note

EdgeCore is intentionally not presented as a finished universal inference platform.

The interesting part of the project is the direction:

```text
Measure
  ↓
Understand
  ↓
Optimize
  ↓
Verify
  ↓
Deploy
```

The long-term objective is to make that loop increasingly automatic and hardware-aware.

**Fast is useful.
Fast + valid + reproducible is deployable.**
