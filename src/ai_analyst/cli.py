#!/usr/bin/env python3
"""
CLI entry point for AI Analyst.

Provides command-line interface for data analysis using Claude.
"""

import os
import sys
from pathlib import Path

import click
from rich.console import Console

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="ai-analyst")
def main():
    """AI Analyst - Intelligent data analysis powered by Claude."""
    pass


@main.command()
@click.argument("file_path", type=click.Path(exists=True), required=False)
@click.option("--model", "-m", default="claude-sonnet-4-20250514", help="Claude model to use")
def interactive(file_path: str | None, model: str):
    """Start interactive analysis session."""
    from ai_analyst.utils.config import setup_logging, get_auth_method, AuthMethod

    setup_logging()

    # Show authentication method
    try:
        auth_method, _ = get_auth_method()
        if auth_method == AuthMethod.PRO_SUBSCRIPTION:
            console.print("[green]Using Claude Pro subscription[/green]")
        else:
            console.print("[yellow]Using API key authentication[/yellow]")
    except ValueError as e:
        console.print(f"[red]Authentication Error:[/red]\n{e}")
        sys.exit(1)

    # Import here to avoid circular imports
    from interactive import run_interactive

    run_interactive(file_path, model)


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--query", "-q", default="Provide a comprehensive analysis of this dataset", help="Analysis query")
@click.option("--model", "-m", default="claude-sonnet-4-20250514", help="Claude model to use")
def analyze(file_path: str, query: str, model: str):
    """Analyze a data file."""
    from ai_analyst.utils.config import setup_logging, get_auth_method, AuthMethod

    setup_logging()

    # Show authentication method
    try:
        auth_method, _ = get_auth_method()
        if auth_method == AuthMethod.PRO_SUBSCRIPTION:
            console.print("[green]Using Claude Pro subscription[/green]")
        else:
            console.print("[yellow]Using API key authentication[/yellow]")
    except ValueError as e:
        console.print(f"[red]Authentication Error:[/red]\n{e}")
        sys.exit(1)

    # Import here to avoid circular imports
    from analyst import StandaloneAnalyst

    analyst = StandaloneAnalyst(model=model)
    result = analyst.analyze(query, file_path)

    from rich.markdown import Markdown
    console.print(Markdown(result))


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
def inspect(file_path: str):
    """Inspect a data file structure."""
    import pandas as pd
    from rich.table import Table

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        console.print(f"[red]Unsupported format:[/red] {suffix}")
        sys.exit(1)

    console.print(f"\n[bold]File:[/bold] {file_path}")
    console.print(f"[bold]Shape:[/bold] {df.shape[0]} rows x {df.shape[1]} columns\n")

    table = Table(title="Column Information")
    table.add_column("Column", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Non-Null", style="yellow")
    table.add_column("Null %", style="red")

    for col in df.columns:
        non_null = df[col].notna().sum()
        null_pct = (df[col].isna().sum() / len(df)) * 100
        table.add_row(col, str(df[col].dtype), str(non_null), f"{null_pct:.1f}%")

    console.print(table)


@main.command()
def auth():
    """Check authentication status and available methods."""
    from ai_analyst.utils.config import (
        get_auth_method,
        check_pro_subscription_available,
        get_settings,
        AuthMethod,
    )

    console.print("\n[bold]Authentication Status[/bold]\n")

    # Check Pro subscription
    pro_available = check_pro_subscription_available()
    if pro_available:
        console.print("[green]\u2713[/green] Claude Pro subscription: Available")
    else:
        console.print("[red]\u2717[/red] Claude Pro subscription: Not configured")
        console.print("    Run [cyan]claude login[/cyan] to authenticate")

    # Check API key
    settings = get_settings()
    api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        console.print(f"[green]\u2713[/green] API key: Configured ({masked_key})")
    else:
        console.print("[red]\u2717[/red] API key: Not set")
        console.print("    Set [cyan]ANTHROPIC_API_KEY[/cyan] environment variable")

    # Show which method will be used
    console.print("\n[bold]Active Authentication:[/bold]")
    try:
        auth_method, _ = get_auth_method()
        if auth_method == AuthMethod.PRO_SUBSCRIPTION:
            console.print("[green]Using Claude Pro subscription[/green] (primary)")
        else:
            console.print("[yellow]Using API key[/yellow] (fallback)")
    except ValueError:
        console.print("[red]No authentication method available[/red]")

    console.print("\n[dim]Tip: Pro subscription is checked first by default.[/dim]")
    console.print("[dim]Set AUTH_PREFERENCE=api to prioritize API key instead.[/dim]\n")


if __name__ == "__main__":
    main()
