from unittest.mock import patch
import pytest
from analyst import StandaloneAnalyst

# Mock response structure
class MockContent:
    def __init__(self, text):
        self.text = text

class MockResponse:
    def __init__(self, text):
        self.stop_reason = "end_turn"
        self.content = [MockContent(text)]

@pytest.mark.parametrize("query,expected", [
    ("Test Query", "Analysis complete."),
    ("Another query", "Another result."),
    ("Complex analysis request", "Detailed analysis output."),
])
def test_sync_wrapper(query, expected):
    """Test that StandaloneAnalyst.analyze returns expected result with mocked AsyncAnthropic."""
    with patch("analyst.AsyncAnthropic") as MockAsyncAnthropic:
        mock_instance = MockAsyncAnthropic.return_value

        async def create_side_effect(*args, **kwargs):
            return MockResponse(expected)

        mock_instance.messages.create.side_effect = create_side_effect

        analyst = StandaloneAnalyst()
        result = analyst.analyze(query)
        assert result == expected
