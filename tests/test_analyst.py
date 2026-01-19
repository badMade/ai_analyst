"""
Tests for StandaloneAnalyst class.

Tests initialization, the analyze() method, and API interaction.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestStandaloneAnalystInit:
    """Tests for StandaloneAnalyst initialization."""

    def test_init_with_default_model(self, mock_anthropic_client, mock_api_key):
        """StandaloneAnalyst initializes with default model."""
        from analyst import StandaloneAnalyst

        analyst = StandaloneAnalyst()
        assert analyst.model == "claude-sonnet-4-20250514"

    def test_init_with_custom_model(self, mock_anthropic_client, mock_api_key):
        """StandaloneAnalyst accepts custom model."""
        from analyst import StandaloneAnalyst

        analyst = StandaloneAnalyst(model="claude-opus-4-20250514")
        assert analyst.model == "claude-opus-4-20250514"

    def test_init_creates_analysis_context(self, mock_anthropic_client, mock_api_key):
        """StandaloneAnalyst creates fresh AnalysisContext."""
        from analyst import StandaloneAnalyst

        analyst = StandaloneAnalyst()
        assert analyst.context is not None
        assert analyst.context.datasets == {}

    def test_init_sets_max_iterations(self, mock_anthropic_client, mock_api_key):
        """StandaloneAnalyst sets max_iterations."""
        from analyst import StandaloneAnalyst

        analyst = StandaloneAnalyst()
        assert analyst.max_iterations == 15

    def test_init_without_api_key_raises_error(self, mock_anthropic_client, monkeypatch):
        """StandaloneAnalyst raises error without API key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # Clear settings cache to pick up env change
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from analyst import StandaloneAnalyst

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            StandaloneAnalyst()


class TestAnalyzeMethod:
    """Tests for the analyze() method."""

    def test_analyze_simple_query_end_turn(
        self, mock_analyst, mock_end_turn_response
    ):
        """analyze() returns text when model ends turn."""
        mock_analyst.client.messages.create.return_value = mock_end_turn_response(
            "Analysis complete: The data shows positive trends."
        )

        result = mock_analyst.analyze("What are the trends?")

        assert "Analysis complete" in result
        assert "positive trends" in result

    def test_analyze_with_file_path(
        self, mock_analyst, mock_end_turn_response, sample_csv_file
    ):
        """analyze() includes file path in query context."""
        mock_analyst.client.messages.create.return_value = mock_end_turn_response(
            "Analyzed the file."
        )

        mock_analyst.analyze("Describe the data", file_path=str(sample_csv_file))

        # Check that the message includes file path
        call_args = mock_analyst.client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        assert str(sample_csv_file) in messages[0]["content"]

    def test_analyze_executes_tool_and_continues(
        self, mock_analyst, mock_tool_use_response, mock_end_turn_response, sample_csv_file
    ):
        """analyze() executes tools and continues conversation."""
        # First response requests tool use
        tool_response = mock_tool_use_response(
            "load_dataset",
            {"file_path": str(sample_csv_file)},
            "tool_123"
        )

        # Second response ends turn
        end_response = mock_end_turn_response("Data loaded successfully.")

        mock_analyst.client.messages.create.side_effect = [tool_response, end_response]

        result = mock_analyst.analyze("Load the data")

        assert "Data loaded" in result
        assert mock_analyst.client.messages.create.call_count == 2

    def test_analyze_multiple_tool_calls(
        self, mock_analyst, mock_tool_use_response, mock_end_turn_response, sample_csv_file
    ):
        """analyze() handles multiple sequential tool calls."""
        # Load dataset
        load_response = mock_tool_use_response(
            "load_dataset",
            {"file_path": str(sample_csv_file)},
            "tool_1"
        )

        # Preview data
        preview_response = mock_tool_use_response(
            "preview_data",
            {"dataset_name": sample_csv_file.stem},
            "tool_2"
        )

        # End turn
        end_response = mock_end_turn_response("Analysis complete.")

        mock_analyst.client.messages.create.side_effect = [
            load_response, preview_response, end_response
        ]

        result = mock_analyst.analyze("Load and preview data")

        assert mock_analyst.client.messages.create.call_count == 3

    def test_analyze_max_iterations(self, mock_analyst, mock_tool_use_response):
        """analyze() stops after max iterations."""
        # Always request tool use (infinite loop scenario)
        tool_response = mock_tool_use_response(
            "list_datasets",
            {},
            "tool_loop"
        )
        mock_analyst.client.messages.create.return_value = tool_response
        mock_analyst.max_iterations = 3

        result = mock_analyst.analyze("Keep going")

        assert "maximum iterations" in result.lower()
        assert mock_analyst.client.messages.create.call_count == 3

    def test_analyze_empty_end_turn_response(self, mock_analyst):
        """analyze() handles end_turn with no text block."""
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = []  # No content blocks

        mock_analyst.client.messages.create.return_value = response

        result = mock_analyst.analyze("Query")

        assert result == ""

    def test_analyze_unexpected_stop_reason(self, mock_analyst):
        """analyze() handles unexpected stop_reason gracefully."""
        response = MagicMock()
        response.stop_reason = "max_tokens"  # Unexpected
        response.content = []

        mock_analyst.client.messages.create.return_value = response

        # Should not crash, should break loop
        result = mock_analyst.analyze("Query")
        assert isinstance(result, str)

    def test_analyze_uses_system_prompt(self, mock_analyst, mock_end_turn_response):
        """analyze() includes system prompt in API call."""
        mock_analyst.client.messages.create.return_value = mock_end_turn_response("Done")

        mock_analyst.analyze("Query")

        call_args = mock_analyst.client.messages.create.call_args
        assert "system" in call_args.kwargs
        assert "data analyst" in call_args.kwargs["system"].lower()

    def test_analyze_uses_tools(self, mock_analyst, mock_end_turn_response):
        """analyze() includes tools in API call."""
        mock_analyst.client.messages.create.return_value = mock_end_turn_response("Done")

        mock_analyst.analyze("Query")

        call_args = mock_analyst.client.messages.create.call_args
        assert "tools" in call_args.kwargs
        assert len(call_args.kwargs["tools"]) > 0


class TestAnalyzeAsync:
    """Tests for the analyze_async() method."""

    @pytest.mark.asyncio
    async def test_analyze_async_returns_result(
        self, mock_analyst, mock_end_turn_response
    ):
        """analyze_async() returns same result as analyze()."""
        mock_analyst.client.messages.create.return_value = mock_end_turn_response(
            "Async result"
        )

        result = await mock_analyst.analyze_async("Query")

        assert "Async result" in result


class TestCreateAnalyst:
    """Tests for the create_analyst() factory function."""

    def test_create_analyst_returns_instance(self, mock_anthropic_client, mock_api_key):
        """create_analyst() returns StandaloneAnalyst instance."""
        from analyst import create_analyst, StandaloneAnalyst

        analyst = create_analyst()
        assert isinstance(analyst, StandaloneAnalyst)

    def test_create_analyst_with_model(self, mock_anthropic_client, mock_api_key):
        """create_analyst() accepts model parameter."""
        from analyst import create_analyst

        analyst = create_analyst(model="claude-opus-4-20250514")
        assert analyst.model == "claude-opus-4-20250514"


class TestToolIntegration:
    """Integration tests for tool execution within analyze()."""

    def test_tool_result_added_to_messages(
        self, mock_analyst, mock_tool_use_response, mock_end_turn_response
    ):
        """Tool results are properly added to conversation."""
        tool_response = mock_tool_use_response(
            "list_datasets",
            {},
            "tool_123"
        )
        end_response = mock_end_turn_response("Done")

        mock_analyst.client.messages.create.side_effect = [tool_response, end_response]

        mock_analyst.analyze("List datasets")

        # Check second call includes tool result
        second_call = mock_analyst.client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]

        # Should have: user message, assistant message, tool result message
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"

    def test_tool_error_included_in_result(
        self, mock_analyst, mock_tool_use_response, mock_end_turn_response
    ):
        """Tool errors are included in tool result message."""
        # Request non-existent dataset
        tool_response = mock_tool_use_response(
            "preview_data",
            {"dataset_name": "nonexistent"},
            "tool_error"
        )
        end_response = mock_end_turn_response("Error noted")

        mock_analyst.client.messages.create.side_effect = [tool_response, end_response]

        mock_analyst.analyze("Preview nonexistent")

        # Tool result should contain error
        second_call = mock_analyst.client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]
        tool_result = messages[2]["content"][0]["content"]

        result_data = json.loads(tool_result)
        assert "error" in result_data
