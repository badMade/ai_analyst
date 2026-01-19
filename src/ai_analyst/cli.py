import click
from ai_analyst.interactive import run_interactive
from ai_analyst.analyst import StandaloneAnalyst

@click.group()
def main():
    """AI Analyst CLI"""
    pass

@main.command()
@click.argument('file_path', required=False)
def interactive(file_path):
    """Run interactive mode"""
    run_interactive(file_path)

@main.command()
@click.argument('file_path')
@click.option('--query', '-q', required=True, help='Analysis query')
def analyze(file_path, query):
    """Run analysis on a file"""
    analyst = StandaloneAnalyst()
    print(analyst.analyze(query, file_path))

if __name__ == '__main__':
    main()
