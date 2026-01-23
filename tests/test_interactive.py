"""
Tests for interactive REPL mode.

Tests the interactive session, command handling, and user interface.

Note: interactive.py is in the project root, not in the ai_analyst package.
"""

import pytest
from unittest.mock import MagicMock, patch
class TestRunInteractive:
    """Tests for run_interactive function."""

    @pytest.fixture
    def mock_console(self):
        """Mock Rich console."""
        with patch("interactive.console") as mock:
            yield mock

    @pytest.fixture
    def mock_prompt(self):
        """Mock Rich prompt."""
        with patch("interactive.Prompt") as mock:
            yield mock

    @pytest.fixture
    def mock_analyst_class(self):
        """Mock StandaloneAnalyst class."""
        with patch("interactive.StandaloneAnalyst") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock, mock_instance

    @pytest.fixture
    def mock_settings_valid(self):
        """Mock valid settings with API key."""
        with patch("interactive.get_settings") as mock:
            mock_settings = MagicMock()
            mock_settings.anthropic_api_key = "valid-key"
            mock.return_value = mock_settings
            yield mock_settings

    @pytest.fixture
    def mock_setup_logging(self):
        """Mock setup_logging function."""
        with patch("interactive.setup_logging"):
            yield

    def test_exits_without_api_key(self, mock_console, mock_setup_logging):
        """Should exit if API key is not set."""
        with patch("interactive.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = ""

            with pytest.raises(SystemExit) as exc_info:
                from interactive import run_interactive
                run_interactive()

            assert exc_info.value.code == 1

    def test_displays_welcome_panel(
        self,
        mock_console,
        mock_prompt,
        mock_analyst_class,
        mock_settings_valid,
        mock_setup_logging
    ):
        """Should display welcome panel on start."""
        mock_prompt.ask.side_effect = ["quit"]

        from interactive import run_interactive
        run_interactive()

        # Check that Panel was used in console.print
        print_calls = mock_console.print.call_args_list
        assert len(print_calls) > 0

    def test_quit_command_exits(
        self,
        mock_console,
        mock_prompt,
        mock_analyst_class,
        mock_settings_valid,
        mock_setup_logging
    ):
        """Should exit on 'quit' command."""
        mock_prompt.ask.side_effect = ["quit"]

        from interactive import run_interactive
        run_interactive()

        # Should have printed goodbye
        goodbye_printed = any(
            "goodbye" in str(call).lower()
            for call in mock_console.print.call_args_list
        )
        assert goodbye_printed

    def test_exit_command_exits(
        self,
        mock_console,
        mock_prompt,
        mock_analyst_class,
        mock_settings_valid,
        mock_setup_logging
    ):
        """Should exit on 'exit' command."""
        mock_prompt.ask.side_effect = ["exit"]

        from interactive import run_interactive
        run_interactive()

        goodbye_printed = any(
            "goodbye" in str(call).lower()
            for call in mock_console.print.call_args_list
        )
        assert goodbye_printed

    def test_q_command_exits(
        self,
        mock_console,
        mock_prompt,
        mock_analyst_class,
        mock_settings_valid,
        mock_setup_logging
    ):
        """Should exit on 'q' command."""
        mock_prompt.ask.side_effect = ["q"]

        from interactive import run_interactive
        run_interactive()

        goodbye_printed = any(
            "goodbye" in str(call).lower()
            for call in mock_console.print.call_args_list
        )
        assert goodbye_printed

    def test_clear_command_clears_console(
        self,
        mock_console,
        mock_prompt,
        mock_analyst_class,
        mock_settings_valid,
        mock_setup_logging
    ):
        """Should clear console on 'clear' command."""
        mock_prompt.ask.side_effect = ["clear", "quit"]

        from interactive import run_interactive
        run_interactive()

        mock_console.clear.assert_called()

    def test_help_command_shows_help(
        self,
        mock_console,
        mock_prompt,
        mock_analyst_class,
        mock_settings_valid,
        mock_setup_logging
    ):
        """Should show help on 'help' command."""
        mock_prompt.ask.side_effect = ["help", "quit"]

        from interactive import run_interactive
        run_interactive()

        help_printed = any(
            "commands" in str(call).lower()
            for call in mock_console.print.call_args_list
        )
        assert help_printed

    def test_load_command_loads_file(
        self,
        mock_console,
        mock_prompt,
        mock_analyst_class,
        mock_settings_valid,
        mock_setup_logging,
        tmp_path
    ):
        """Should load file on 'load' command."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("a,b\n1,2")

        mock_prompt.ask.side_effect = [f"load {test_file}", "quit"]

        from interactive import run_interactive
        run_interactive()

        loaded_printed = any(
            "loaded" in str(call).lower()
            for call in mock_console.print.call_args_list
        )
        assert loaded_printed

    def test_load_command_handles_missing_file(
        self,
        mock_console,
        mock_prompt,
        mock_analyst_class,
        mock_settings_valid,
        mock_setup_logging
    ):
        """Should show error for missing file."""
        mock_prompt.ask.side_effect = ["load /nonexistent/file.csv", "quit"]

        from interactive import run_interactive
        run_interactive()

        error_printed = any(
            "not found" in str(call).lower()
            for call in mock_console.print.call_args_list
        )
        assert error_printed

    def test_empty_input_continues(
        self,
        mock_console,
        mock_prompt,
        mock_analyst_class,
        mock_settings_valid,
        mock_setup_logging
    ):
        """Should continue on empty input."""
        mock_prompt.ask.side_effect = ["", "quit"]

        from interactive import run_interactive
        run_interactive()

        # Should have asked twice
        assert mock_prompt.ask.call_count == 2

    def test_query_calls_analyst(
        self,
        mock_console,
        mock_prompt,
        mock_analyst_class,
        mock_settings_valid,
        mock_setup_logging
    ):
        """Should call analyst.analyze for queries."""
        mock_class, mock_instance = mock_analyst_class
        mock_instance.analyze.return_value = "Analysis result"

        mock_prompt.ask.side_effect = ["What is the average?", "quit"]

        from interactive import run_interactive
        run_interactive()

        mock_instance.analyze.assert_called_once()

    def test_shows_analyzing_message(
        self,
        mock_console,
        mock_prompt,
        mock_analyst_class,
        mock_settings_valid,
        mock_setup_logging
    ):
        """Should show 'Analyzing...' message."""
        mock_class, mock_instance = mock_analyst_class
        mock_instance.analyze.return_value = "Result"

        mock_prompt.ask.side_effect = ["Analyze this", "quit"]

        from interactive import run_interactive
        run_interactive()

        analyzing_printed = any(
            "analyzing" in str(call).lower()
            for call in mock_console.print.call_args_list
        )
        assert analyzing_printed

    def test_handles_keyboard_interrupt(
        self,
        mock_console,
        mock_prompt,
        mock_analyst_class,
        mock_settings_valid,
        mock_setup_logging
    ):
        """Should handle KeyboardInterrupt gracefully."""
        mock_prompt.ask.side_effect = [KeyboardInterrupt(), "quit"]

        from interactive import run_interactive
        run_interactive()

        # Should show message about using quit
        quit_message = any(
            "quit" in str(call).lower()
            for call in mock_console.print.call_args_list
        )
        assert quit_message

    def test_handles_exception(
        self,
        mock_console,
        mock_prompt,
        mock_analyst_class,
        mock_settings_valid,
        mock_setup_logging
    ):
        """Should handle exceptions gracefully."""
        mock_class, mock_instance = mock_analyst_class
        mock_instance.analyze.side_effect = Exception("Test error")

        mock_prompt.ask.side_effect = ["cause error", "quit"]

        from interactive import run_interactive
        run_interactive()

        error_printed = any(
            "error" in str(call).lower()
            for call in mock_console.print.call_args_list
        )
        assert error_printed

    def test_uses_provided_file_path(
        self,
        mock_console,
        mock_prompt,
        mock_analyst_class,
        mock_settings_valid,
        mock_setup_logging,
        tmp_path
    ):
        """Should use provided file_path."""
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2")

        mock_class, mock_instance = mock_analyst_class
        mock_instance.analyze.return_value = "Result"

        mock_prompt.ask.side_effect = ["analyze", "quit"]

        from interactive import run_interactive
        run_interactive(file_path=str(test_file))

        # Check that the file path was passed to analyze
        call_args = mock_instance.analyze.call_args
        assert str(test_file) in str(call_args)

    def test_uses_provided_model(
        self,
        mock_console,
        mock_prompt,
        mock_analyst_class,
        mock_settings_valid,
        mock_setup_logging
    ):
        """Should use provided model."""
        mock_class, mock_instance = mock_analyst_class

        mock_prompt.ask.side_effect = ["quit"]

        from interactive import run_interactive
        run_interactive(model="claude-3-opus-20240229")

        mock_class.assert_called_once_with(model="claude-3-opus-20240229")
