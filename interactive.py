"""
Interactive REPL for AI Analyst

Provides a continuous analysis session with conversation history.
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from analyst import StandaloneAnalyst
from ai_analyst.utils.config import get_settings, setup_logging

console = Console()


def run_interactive(file_path: str | None = None, model: str = "claude-sonnet-4-20250514"):
    """Run interactive analysis REPL."""
    setup_logging()
    settings = get_settings()
    
    if not settings.anthropic_api_key:
        console.print("[red]Error:[/red] ANTHROPIC_API_KEY not set")
        sys.exit(1)
    
    console.print(Panel(
        "[bold cyan]AI Analyst Interactive Mode[/bold cyan]\n\n"
        "Commands:\n"
        "  [green]load <file>[/green]  - Load a dataset\n"
        "  [green]quit/exit[/green]    - Exit session\n"
        "  [green]clear[/green]        - Clear screen\n"
        "  [green]help[/green]         - Show this help\n\n"
        "Or just type your analysis question.",
        title="Welcome",
        border_style="blue"
    ))
    
    analyst = StandaloneAnalyst(model=model)
    current_file = file_path
    
    if current_file:
        console.print(f"\n[dim]Working with:[/dim] {current_file}")
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("\n[dim]Goodbye![/dim]")
                break
            
            if user_input.lower() == "clear":
                console.clear()
                continue
            
            if user_input.lower() == "help":
                console.print(
                    "Commands: load <file>, quit, clear, help\n"
                    "Or ask any analysis question."
                )
                continue
            
            if user_input.lower().startswith("load "):
                new_file = user_input[5:].strip()
                if Path(new_file).exists():
                    current_file = new_file
                    console.print(f"[green]Loaded:[/green] {current_file}")
                else:
                    console.print(f"[red]File not found:[/red] {new_file}")
                continue
            
            # Run analysis
            console.print("[dim]Analyzing...[/dim]")
            
            response = analyst.analyze(user_input, current_file)
            
            console.print("\n[bold green]Analyst[/bold green]")
            console.print(Markdown(response))
        
        except KeyboardInterrupt:
            console.print("\n[dim]Use 'quit' to exit[/dim]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


if __name__ == "__main__":
    import sys
    file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_interactive(file_arg)
