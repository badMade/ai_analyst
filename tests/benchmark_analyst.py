import asyncio
import time
from unittest.mock import patch
from analyst import StandaloneAnalyst

# Mock response structure
class MockContent:
    text = "Analysis complete."

class MockResponse:
    stop_reason = "end_turn"
    content = [MockContent()]

async def run_benchmark(n_requests=50):
    print(f"Starting benchmark with {n_requests} concurrent requests...")

    # Patch AsyncAnthropic since the new code uses it
    with patch("analyst.AsyncAnthropic") as MockAsyncAnthropic:
        mock_instance = MockAsyncAnthropic.return_value

        # Simulate network latency (non-blocking)
        async def create_side_effect(*args, **kwargs):
            await asyncio.sleep(0.1)
            return MockResponse()

        mock_instance.messages.create.side_effect = create_side_effect

        analyst = StandaloneAnalyst()

        start_time = time.time()

        tasks = []
        for i in range(n_requests):
            tasks.append(analyst.analyze_async(f"Query {i}"))

        await asyncio.gather(*tasks)

        duration = time.time() - start_time
        print(f"Time taken: {duration:.4f} seconds")

        return duration

if __name__ == "__main__":
    asyncio.run(run_benchmark())
