"""
Tests for StandaloneAnalyst class.

Tests the core analyst functionality including tool execution and API interaction.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestStandaloneAnalystInit:
    """Tests for StandaloneAnalyst initialization."""

    def test_init_creates_client(self, mock_settings):
        """Should lazily create Anthropic client when first used."""
        with patch("analyst.Anthropic") as mock_client:
            from analyst import StandaloneAnalyst

            analyst = StandaloneAnalyst()
            mock_client.assert_not_called()

            _ = analyst.client
            mock_client.assert_called_once()

    def test_init_uses_provided_model(self, mock_settings):
        """Should use the provided model name."""
        with patch("analyst.Anthropic"), patch("analyst.AsyncAnthropic"):
            from analyst import StandaloneAnalyst

            analyst = StandaloneAnalyst(model="claude-3-opus-20240229")

            assert analyst.model == "claude-3-opus-20240229"

    def test_init_creates_analysis_context(self, mock_settings):
        """Should create an AnalysisContext."""
        with patch("analyst.Anthropic"), patch("analyst.AsyncAnthropic"):
            from analyst import AnalysisContext, StandaloneAnalyst

            analyst = StandaloneAnalyst()

            assert isinstance(analyst.context, AnalysisContext)

    def test_init_sets_max_iterations(self, mock_settings):
        """Should set max iterations limit."""
        with patch("analyst.Anthropic"), patch("analyst.AsyncAnthropic"):
            from analyst import StandaloneAnalyst

            analyst = StandaloneAnalyst()

            assert analyst.max_iterations == 15

    def test_init_raises_without_api_key(self):
        """Should raise error if API key is not set."""
        # Need to patch where get_auth_method is used (in analyst module)
        with patch("analyst.get_auth_method") as mock_get_auth_method:
            mock_get_auth_method.side_effect = ValueError("Missing ANTHROPIC_API_KEY")

            with patch("analyst.Anthropic"):
                from analyst import StandaloneAnalyst

                with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                    StandaloneAnalyst()


class TestExecuteTool:
    """Tests for _execute_tool method."""

    @pytest.fixture
    def analyst_with_data(self, mock_settings, sample_csv_file):
        """Create analyst with loaded dataset."""
        with patch("analyst.Anthropic"), patch("analyst.AsyncAnthropic"):
            from analyst import StandaloneAnalyst

            analyst = StandaloneAnalyst()
            analyst.context.load_dataset(sample_csv_file, "test_data")
            return analyst

    def test_execute_load_dataset(self, mock_settings, sample_csv_file):
        """Should execute load_dataset tool."""
        with patch("analyst.Anthropic"), patch("analyst.AsyncAnthropic"):
            from analyst import StandaloneAnalyst

            analyst = StandaloneAnalyst()

            result = analyst._execute_tool("load_dataset", {
                "file_path": sample_csv_file,
                "name": "my_data"
            })

            result_dict = json.loads(result)
            assert result_dict["name"] == "my_data"
            assert result_dict["rows"] == 100

    def test_execute_list_datasets(self, analyst_with_data):
        """Should list loaded datasets."""
        result = analyst_with_data._execute_tool("list_datasets", {})

        result_dict = json.loads(result)
        assert "test_data" in result_dict["datasets"]
        assert result_dict["count"] == 1

    def test_execute_preview_data(self, analyst_with_data):
        """Should preview dataset rows."""
        result = analyst_with_data._execute_tool("preview_data", {
            "dataset_name": "test_data",
            "n_rows": 5
        })

        result_dict = json.loads(result)
        assert len(result_dict["data"]) == 5
        assert result_dict["total_rows"] == 100

    def test_execute_preview_data_with_columns(self, analyst_with_data):
        """Should preview specific columns."""
        result = analyst_with_data._execute_tool("preview_data", {
            "dataset_name": "test_data",
            "n_rows": 3,
            "columns": ["id", "value"]
        })

        result_dict = json.loads(result)
        assert set(result_dict["columns"]) == {"id", "value"}

    def test_execute_describe_statistics(self, analyst_with_data):
        """Should compute descriptive statistics."""
        result = analyst_with_data._execute_tool("describe_statistics", {
            "dataset_name": "test_data"
        })

        result_dict = json.loads(result)
        assert "statistics" in result_dict
        assert len(result_dict["statistics"]) > 0

    def test_execute_compute_correlation(self, analyst_with_data):
        """Should compute correlation matrix."""
        result = analyst_with_data._execute_tool("compute_correlation", {
            "dataset_name": "test_data",
            "method": "pearson"
        })

        result_dict = json.loads(result)
        assert "correlations" in result_dict
        assert result_dict["method"] == "pearson"

    def test_execute_detect_outliers_iqr(self, analyst_with_data):
        """Should detect outliers using IQR method."""
        result = analyst_with_data._execute_tool("detect_outliers", {
            "dataset_name": "test_data",
            "column": "value",
            "method": "iqr"
        })

        result_dict = json.loads(result)
        assert result_dict["method"] == "iqr"
        assert "outlier_count" in result_dict
        assert "lower_bound" in result_dict
        assert "upper_bound" in result_dict

    def test_execute_detect_outliers_zscore(self, analyst_with_data):
        """Should detect outliers using Z-score method."""
        result = analyst_with_data._execute_tool("detect_outliers", {
            "dataset_name": "test_data",
            "column": "value",
            "method": "zscore",
            "threshold": 2.5
        })

        result_dict = json.loads(result)
        assert result_dict["method"] == "zscore"
        assert result_dict["threshold"] == 2.5

    def test_execute_group_analysis(self, analyst_with_data):
        """Should perform grouped aggregation."""
        result = analyst_with_data._execute_tool("group_analysis", {
            "dataset_name": "test_data",
            "group_by": "category",
            "agg_column": "value"
        })

        result_dict = json.loads(result)
        assert "results" in result_dict
        assert result_dict["group_by"] == "category"
        assert result_dict["n_groups"] > 0

    def test_execute_check_data_quality(self, analyst_with_data):
        """Should check data quality."""
        result = analyst_with_data._execute_tool("check_data_quality", {
            "dataset_name": "test_data"
        })

        result_dict = json.loads(result)
        assert "total_rows" in result_dict
        assert "null_cells" in result_dict
        assert "duplicate_rows" in result_dict
        assert "quality_score" in result_dict

    def test_execute_test_normality(self, analyst_with_data):
        """Should test normality of column."""
        result = analyst_with_data._execute_tool("test_normality", {
            "dataset_name": "test_data",
            "column": "value"
        })

        result_dict = json.loads(result)
        assert result_dict["column"] == "value"
        assert "test" in result_dict
        assert "p_value" in result_dict
        assert "is_normal" in result_dict

    def test_execute_analyze_trend(self, analyst_with_data):
        """Should analyze trend in column."""
        result = analyst_with_data._execute_tool("analyze_trend", {
            "dataset_name": "test_data",
            "column": "value"
        })

        result_dict = json.loads(result)
        assert "trend" in result_dict

    def test_execute_unknown_tool(self, analyst_with_data):
        """Should return error for unknown tool."""
        result = analyst_with_data._execute_tool("unknown_tool", {})

        result_dict = json.loads(result)
        assert "error" in result_dict
        assert "Unknown tool" in result_dict["error"]

    def test_execute_tool_with_error(self, analyst_with_data):
        """Should handle tool execution errors gracefully."""
        result = analyst_with_data._execute_tool("preview_data", {
            "dataset_name": "nonexistent"
        })

        result_dict = json.loads(result)
        assert "error" in result_dict


class TestAnalyze:
    """Tests for analyze method."""

    @pytest.fixture
    def analyst(self, mock_settings):
        """Create analyst with mocked client."""
        with patch("analyst.Anthropic"), patch("analyst.AsyncAnthropic"):
            from analyst import StandaloneAnalyst

            analyst = StandaloneAnalyst()
            analyst.client = MagicMock()
            return analyst

    def test_analyze_returns_text_on_end_turn(self, analyst, mock_api_response_end_turn):
        """Should return text when API ends turn."""
        analyst.client.messages.create.return_value = mock_api_response_end_turn

        result = analyst.analyze("What is the data about?")

        assert result == "Analysis complete. The data shows positive trends."

    def test_analyze_includes_file_path_in_query(self, analyst, mock_api_response_end_turn):
        """Should include file path in the query context."""
        analyst.client.messages.create.return_value = mock_api_response_end_turn

        analyst.analyze("Analyze this", file_path="/data/test.csv")

        call_args = analyst.client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        assert "/data/test.csv" in messages[0]["content"]

    def test_analyze_uses_system_prompt(self, analyst, mock_api_response_end_turn):
        """Should use the system prompt."""
        analyst.client.messages.create.return_value = mock_api_response_end_turn

        analyst.analyze("Test query")

        call_args = analyst.client.messages.create.call_args
        assert "system" in call_args.kwargs
        assert "data analyst" in call_args.kwargs["system"].lower()

    def test_analyze_processes_tool_use(self, analyst, mock_api_response_tool_use, mock_api_response_end_turn, sample_csv_file):
        """Should process tool use requests."""
        # First call returns tool use, second returns end turn
        analyst.client.messages.create.side_effect = [
            mock_api_response_tool_use,
            mock_api_response_end_turn
        ]

        # Mock the tool to not fail
        mock_api_response_tool_use.content[0].input = {
            "file_path": sample_csv_file,
            "name": "test"
        }

        result = analyst.analyze("Load the data")

        assert isinstance(result, str)
        assert analyst.client.messages.create.call_count == 2

    def test_analyze_respects_max_iterations(self, analyst, mock_api_response_tool_use, sample_csv_file):
        """Should stop after max iterations."""
        # Always return tool use to simulate infinite loop
        mock_api_response_tool_use.content[0].input = {
            "file_path": sample_csv_file
        }
        analyst.client.messages.create.return_value = mock_api_response_tool_use
        analyst.max_iterations = 3

        result = analyst.analyze("Keep going")

        assert analyst.client.messages.create.call_count == 3
        assert "maximum iterations" in result.lower()


class TestAnalyzeAsync:
    """Tests for async analyze method."""

    @pytest.fixture
    def analyst(self, mock_settings):
        """Create analyst with mocked client."""
        with patch("analyst.Anthropic"), patch("analyst.AsyncAnthropic"):
            from analyst import StandaloneAnalyst

            analyst = StandaloneAnalyst()
            analyst.async_client = MagicMock()
            analyst.async_client.messages.create = AsyncMock()
            return analyst

    @pytest.mark.asyncio
    async def test_analyze_async_returns_text_on_end_turn(self, analyst, mock_api_response_end_turn):
        """Async method should return text when API ends turn."""
        analyst.async_client.messages.create.return_value = mock_api_response_end_turn

        result = await analyst.analyze_async("Test query")

        assert result == "Analysis complete. The data shows positive trends."
        analyst.async_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_async_with_file_path(self, analyst, mock_api_response_end_turn):
        """Async method should pass file_path."""
        analyst.async_client.messages.create.return_value = mock_api_response_end_turn

        await analyst.analyze_async("Test", file_path="/data/file.csv")

        call_args = analyst.async_client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        assert "/data/file.csv" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_analyze_async_processes_tool_use(self, analyst, mock_api_response_tool_use, mock_api_response_end_turn, sample_csv_file):
        """Should process tool use requests in async mode."""
        analyst.async_client.messages.create.side_effect = [
            mock_api_response_tool_use,
            mock_api_response_end_turn
        ]

        # Mock the tool to not fail
        mock_api_response_tool_use.content[0].input = {
            "file_path": sample_csv_file,
            "name": "test"
        }

        result = await analyst.analyze_async("Load the data")

        assert isinstance(result, str)
        assert analyst.async_client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_analyze_async_concurrent_execution_is_isolated(
        self,
        analyst,
        mock_api_response_tool_use,
        mock_api_response_end_turn,
    ):
        """Concurrent async runs should not share dataset context."""
        first_tool = MagicMock()
        first_tool.type = "tool_use"
        first_tool.name = "list_datasets"
        first_tool.id = "tool-1"
        first_tool.input = {}

        second_tool = MagicMock()
        second_tool.type = "tool_use"
        second_tool.name = "list_datasets"
        second_tool.id = "tool-2"
        second_tool.input = {}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [first_tool]

        second_response = MagicMock()
        second_response.stop_reason = "tool_use"
        second_response.content = [second_tool]

        analyst.async_client.messages.create.side_effect = [
            first_response,
            second_response,
            mock_api_response_end_turn,
            mock_api_response_end_turn,
        ]

        await asyncio.gather(
            analyst.analyze_async("query one"),
            analyst.analyze_async("query two"),
        )

        # Group tool results by originating user prompt to verify per-call isolation.
        calls = analyst.async_client.messages.create.call_args_list
        results_by_query: dict[str, list[dict]] = {"query one": [], "query two": []}

        for call in calls:
            kwargs = call.kwargs
            messages = kwargs.get("messages", [])
            if len(messages) <= 1:
                # Initial call with only the user message; no tool result yet.
                continue

            # The first message in the conversation should be the original user prompt.
            first_message_content = messages[0]["content"][0]["text"]
            if first_message_content not in results_by_query:
                # Ignore any messages that are not part of the two concurrent queries.
                continue

            tool_result_content = messages[-1]["content"][0]["content"]
            tool_result = json.loads(tool_result_content)
            results_by_query[first_message_content].append(tool_result)

        # Each concurrent call should independently see an empty dataset list.
        for query, tool_results in results_by_query.items():
            assert tool_results, f"No tool results captured for {query}"
            for tool_result in tool_results:
                assert tool_result["count"] == 0
                # If the tool returns an explicit list of datasets, it should be empty.
                if "datasets" in tool_result:
                    assert isinstance(tool_result["datasets"], list)
                    assert not tool_result["datasets"]

        # Demonstrate that parsed tool results for each query are independent objects.
        if (
            results_by_query["query one"]
            and results_by_query["query two"]
            and "datasets" in results_by_query["query one"][0]
            and "datasets" in results_by_query["query two"][0]
        ):
            results_by_query["query one"][0]["datasets"].append("dummy-dataset")
            assert results_by_query["query two"][0]["datasets"] == []
class TestCreateAnalyst:
    """Tests for create_analyst factory function."""

    def test_create_analyst_returns_instance(self, mock_settings):
        """Should return StandaloneAnalyst instance."""
        with patch("analyst.Anthropic"), patch("analyst.AsyncAnthropic"):
            from analyst import StandaloneAnalyst, create_analyst

            analyst = create_analyst()

            assert isinstance(analyst, StandaloneAnalyst)

    def test_create_analyst_with_model(self, mock_settings):
        """Should pass model to constructor."""
        with patch("analyst.Anthropic"), patch("analyst.AsyncAnthropic"):
            from analyst import create_analyst

            analyst = create_analyst(model="claude-3-haiku-20240307")

            assert analyst.model == "claude-3-haiku-20240307"


class TestToolDefinitions:
    """Tests for TOOLS constant."""

    def test_tools_is_list(self):
        """TOOLS should be a list."""
        from analyst import TOOLS

        assert isinstance(TOOLS, list)

    def test_tools_have_required_fields(self):
        """Each tool should have name, description, input_schema."""
        from analyst import TOOLS

        for tool in TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_tools_have_valid_schemas(self):
        """Tool input schemas should be valid JSON schemas."""
        from analyst import TOOLS

        for tool in TOOLS:
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_expected_tools_exist(self):
        """Expected tools should be defined."""
        from analyst import TOOLS

        tool_names = {tool["name"] for tool in TOOLS}

        expected = {
            "load_dataset",
            "list_datasets",
            "preview_data",
            "describe_statistics",
            "compute_correlation",
            "detect_outliers",
            "group_analysis",
            "check_data_quality",
            "test_normality",
            "analyze_trend"
        }

        assert expected.issubset(tool_names)
