from unittest.mock import patch
from analyst import StandaloneAnalyst

# Mock response structure
class MockContent:
    text = "Analysis complete."

class MockResponse:
    stop_reason = "end_turn"
    content = [MockContent()]

def test_sync_wrapper():
    with patch("analyst.AsyncAnthropic") as MockAsyncAnthropic:
        mock_instance = MockAsyncAnthropic.return_value

        async def create_side_effect(*args, **kwargs):
            return MockResponse()

        mock_instance.messages.create.side_effect = create_side_effect

        analyst = StandaloneAnalyst()
        result = analyst.analyze("Test Query")
        assert result == "Analysis complete."
