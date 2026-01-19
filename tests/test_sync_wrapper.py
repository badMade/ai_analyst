import asyncio
import time
from unittest.mock import MagicMock, patch
from analyst import StandaloneAnalyst

# Mock response structure
class MockContent:
    text = "Analysis complete."

class MockResponse:
    stop_reason = "end_turn"
    content = [MockContent()]

def test_sync_wrapper():
    print("Testing sync wrapper...")
    with patch("analyst.AsyncAnthropic") as MockAsyncAnthropic:
        mock_instance = MockAsyncAnthropic.return_value

        async def create_side_effect(*args, **kwargs):
            return MockResponse()

        mock_instance.messages.create.side_effect = create_side_effect

        analyst = StandaloneAnalyst()
        result = analyst.analyze("Test Query")
        print(f"Result: {result}")
        assert result == "Analysis complete."

if __name__ == "__main__":
    test_sync_wrapper()
