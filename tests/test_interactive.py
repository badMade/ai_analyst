"""
Tests for interactive REPL module.

Tests command parsing, file loading, and error handling.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestRunInteractiveSetup:
    """Tests for run_interactive setup and initialization."""

    def test_missing_api_key_exits(self, monkeypatch):
        """run_interactive exits if API key not set."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # Clear settings cache
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from interactive import run_interactive

        with pytest.raises(SystemExit) as exc_info:
            with patch("interactive.Prompt.ask", return_value="quit"):
                with patch("interactive.console"):
                    run_interactive()

        assert exc_info.value.code == 1


class TestCommandParsing:
    """Tests for command parsing in interactive mode."""

    def test_quit_command(self, mock_api_key):
        """quit command exits the loop."""
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from interactive import run_interactive

        with patch("interactive.StandaloneAnalyst") as mock_analyst_cls:
            with patch("interactive.console") as mock_console:
                with patch("interactive.Prompt.ask", side_effect=["quit"]):
                    run_interactive()

                # Should print goodbye message
                assert any("Goodbye" in str(call) for call in mock_console.print.call_args_list)

    def test_exit_command(self, mock_api_key):
        """exit command exits the loop."""
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from interactive import run_interactive

        with patch("interactive.StandaloneAnalyst"):
            with patch("interactive.console"):
                with patch("interactive.Prompt.ask", side_effect=["exit"]):
                    run_interactive()

        # Should exit gracefully

    def test_q_command(self, mock_api_key):
        """q command exits the loop."""
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from interactive import run_interactive

        with patch("interactive.StandaloneAnalyst"):
            with patch("interactive.console"):
                with patch("interactive.Prompt.ask", side_effect=["q"]):
                    run_interactive()

        # Should exit gracefully

    def test_clear_command(self, mock_api_key):
        """clear command clears the console."""
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from interactive import run_interactive

        with patch("interactive.StandaloneAnalyst"):
            with patch("interactive.console") as mock_console:
                with patch("interactive.Prompt.ask", side_effect=["clear", "quit"]):
                    run_interactive()

                mock_console.clear.assert_called()

    def test_help_command(self, mock_api_key):
        """help command shows help message."""
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from interactive import run_interactive

        with patch("interactive.StandaloneAnalyst"):
            with patch("interactive.console") as mock_console:
                with patch("interactive.Prompt.ask", side_effect=["help", "quit"]):
                    run_interactive()

                # Should print help info
                assert any("load" in str(call).lower() for call in mock_console.print.call_args_list)

    def test_empty_input_continues(self, mock_api_key):
        """Empty input is ignored."""
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from interactive import run_interactive

        with patch("interactive.StandaloneAnalyst"):
            with patch("interactive.console"):
                with patch("interactive.Prompt.ask", side_effect=["", "   ", "quit"]):
                    run_interactive()

        # Should not crash, should exit normally


class TestLoadCommand:
    """Tests for the load command."""

    def test_load_existing_file(self, mock_api_key, sample_csv_file):
        """load command with existing file sets current_file."""
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from interactive import run_interactive

        with patch("interactive.StandaloneAnalyst"):
            with patch("interactive.console") as mock_console:
                with patch("interactive.Prompt.ask", side_effect=[f"load {sample_csv_file}", "quit"]):
                    run_interactive()

                # Should print loaded message
                assert any("Loaded" in str(call) for call in mock_console.print.call_args_list)

    def test_load_nonexistent_file(self, mock_api_key):
        """load command with non-existent file shows error."""
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from interactive import run_interactive

        with patch("interactive.StandaloneAnalyst"):
            with patch("interactive.console") as mock_console:
                with patch("interactive.Prompt.ask", side_effect=["load /nonexistent/file.csv", "quit"]):
                    run_interactive()

                # Should print error
                assert any("not found" in str(call).lower() for call in mock_console.print.call_args_list)


class TestAnalysisQueries:
    """Tests for analysis queries."""

    def test_query_calls_analyst(self, mock_api_key):
        """Regular input calls analyst.analyze()."""
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from interactive import run_interactive

        mock_analyst = MagicMock()
        mock_analyst.analyze.return_value = "Analysis result"

        with patch("interactive.StandaloneAnalyst", return_value=mock_analyst):
            with patch("interactive.console"):
                with patch("interactive.Prompt.ask", side_effect=["What is the data?", "quit"]):
                    run_interactive()

        mock_analyst.analyze.assert_called_once()

    def test_query_with_file_path(self, mock_api_key, sample_csv_file):
        """Query includes current file path."""
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from interactive import run_interactive

        mock_analyst = MagicMock()
        mock_analyst.analyze.return_value = "Result"

        with patch("interactive.StandaloneAnalyst", return_value=mock_analyst):
            with patch("interactive.console"):
                with patch("interactive.Prompt.ask", side_effect=[
                    f"load {sample_csv_file}",
                    "Analyze this",
                    "quit"
                ]):
                    run_interactive()

        # analyze should be called with file path (positional arg)
        call_args = mock_analyst.analyze.call_args
        # file_path is second positional argument
        assert len(call_args[0]) > 1 and call_args[0][1] is not None


class TestErrorHandling:
    """Tests for error handling in interactive mode."""

    def test_keyboard_interrupt_continues(self, mock_api_key):
        """KeyboardInterrupt shows message but continues."""
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from interactive import run_interactive

        mock_analyst = MagicMock()

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise KeyboardInterrupt()
            return "quit"

        with patch("interactive.StandaloneAnalyst", return_value=mock_analyst):
            with patch("interactive.console") as mock_console:
                with patch("interactive.Prompt.ask", side_effect=side_effect):
                    run_interactive()

                # Should print use quit message
                assert any("quit" in str(call).lower() for call in mock_console.print.call_args_list)

    def test_analysis_error_displayed(self, mock_api_key):
        """Analysis errors are displayed to user."""
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from interactive import run_interactive

        mock_analyst = MagicMock()
        mock_analyst.analyze.side_effect = Exception("Analysis failed")

        with patch("interactive.StandaloneAnalyst", return_value=mock_analyst):
            with patch("interactive.console") as mock_console:
                with patch("interactive.Prompt.ask", side_effect=["query", "quit"]):
                    run_interactive()

                # Should print error
                assert any("Error" in str(call) for call in mock_console.print.call_args_list)


class TestModelParameter:
    """Tests for model parameter handling."""

    def test_custom_model_used(self, mock_api_key):
        """run_interactive passes model to StandaloneAnalyst."""
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from interactive import run_interactive

        with patch("interactive.StandaloneAnalyst") as mock_cls:
            mock_cls.return_value = MagicMock()
            with patch("interactive.console"):
                with patch("interactive.Prompt.ask", return_value="quit"):
                    run_interactive(model="claude-opus-4-20250514")

        mock_cls.assert_called_once_with(model="claude-opus-4-20250514")


class TestInitialFilePath:
    """Tests for initial file path parameter."""

    def test_initial_file_displayed(self, mock_api_key, sample_csv_file):
        """Initial file path is displayed on startup."""
        from ai_analyst.utils.config import get_settings
        get_settings.cache_clear()

        from interactive import run_interactive

        with patch("interactive.StandaloneAnalyst") as mock_cls:
            mock_cls.return_value = MagicMock()
            with patch("interactive.console") as mock_console:
                with patch("interactive.Prompt.ask", return_value="quit"):
                    run_interactive(file_path=str(sample_csv_file))

                # Should print working with message
                assert any("Working with" in str(call) or str(sample_csv_file) in str(call)
                           for call in mock_console.print.call_args_list)
