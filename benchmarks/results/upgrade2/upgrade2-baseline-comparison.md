============================================================
EDGECORE UPGRADE 2
REAL RUNTIME COMPARISON
============================================================

MODEL
  Qwen2.5-0.5B-Instruct

HARDWARE
  Intel Core i3-1115G4
  AVX2
  8 GB RAM

WORKLOAD
  batch: 1
  context: 2048
  generation: 32 tokens

------------------------------------------------------------
MEASUREMENT METHODOLOGY (as currently measured)
------------------------------------------------------------

PyTorch:
  - TTFT: separate single-token generate() call
  - TPOT: (total - ttft) / (tokens - 1)
  - tok/s: tokens / total wall time
  - RAM: peak process RSS via psutil

llama.cpp:
  - TTFT: separate single-token llama-cli run (warm model)
  - TPOT: (total - separate_ttft) / (tokens - 1)
  - tok/s: tokens / total wall time (includes warm model load)
  - RAM: peak RSS of llama-cli subprocess
  - CAVEAT: TTFT/TPOT use separate runs; tok/s includes load time

EdgeCore:
  - TTFT/TPOT/tok/s: native runtime JSON output
  - RAM: peak RSS from runtime JSON (peak_rss_gb)

------------------------------------------------------------
BASELINE COMPARISON
------------------------------------------------------------

```
Runtime      Threads  tok/s    P95 (ms)   RAM (MB)   Status    
---------------------------------------------------------------
pytorch      1        1.2      536.2      36         —         
pytorch      2        1.1      589.9      26         —         
pytorch      4        0.9      1721.3     27         —         
llamacpp     1        11.9     84.7       1296       —         
llamacpp     2        11.9     86.6       1296       —         
llamacpp     4        11.7     90.7       1296       —         
edgecore     1        5.4      107.8      1361       Reject    
edgecore     2        8.1      101.5      1361       Accept    
edgecore     4        5.7      247.6      1361       Reject    
```

------------------------------------------------------------
RELATIVE PERFORMANCE
------------------------------------------------------------

  EdgeCore / PyTorch (1T): 4.59×
  EdgeCore / llama.cpp (1T): 0.46×
  EdgeCore / PyTorch (2T): 7.10×
  EdgeCore / llama.cpp (2T): 0.68×
  EdgeCore / PyTorch (4T): 6.56×
  EdgeCore / llama.cpp (4T): 0.48×

------------------------------------------------------------
CONSTRAINT
------------------------------------------------------------

  P95 < 100 ms
  Accuracy loss < 1%
  RAM < 6 GB

------------------------------------------------------------
EDGECORE
------------------------------------------------------------

  Recommended:
    FP16 / 2 Threads

  Verification:
    Correctness: PASS
    Accuracy:    PASS
    Performance: FAIL (P95: 101.5 ms)
    Memory:      PASS (RAM: 1361 MB)

  STATUS: FAILED

============================================================
Report generated: 2026-09-09T10:08:44.143496Z
============================================================