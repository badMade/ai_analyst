# Performance Journal

## Optimize `analyze_async` with `AsyncAnthropic`
- **Date**: 2026-02-14
- **Author**: Jules
- **Change**: Replaced `asyncio.to_thread` wrapper around synchronous `analyze` with native `analyze_async` using `AsyncAnthropic`.
- **Reason**: `asyncio.to_thread` uses a thread pool which limits concurrency (typically ~32 threads). `AsyncAnthropic` uses `asyncio` for non-blocking I/O, allowing thousands of concurrent requests.
- **Impact**:
  - Baseline (500 concurrent requests, 100ms latency): ~6.33s (~79 RPS). Bottlenecked by thread pool.
  - Optimized (500 concurrent requests, 100ms latency): ~0.11s (~4560 RPS). Only limited by network/CPU overhead.
  - Speedup: ~58x in synthetic benchmark.
