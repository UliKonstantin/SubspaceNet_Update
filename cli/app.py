"""SubspaceNet CLI v2 application."""
import sys
from pathlib import Path

import click

from cli.help_text import CLI_GROUP_HELP


def _register_commands():
    """Lazy-import commands so `--help` does not load Simulation/torch."""
    from cli.commands.run import run_command
    from cli.commands.save import save_command
    from cli.commands.show import show_command

    cli.add_command(run_command)
    cli.add_command(show_command)
    cli.add_command(save_command)


@click.group()
@click.pass_context
def cli(ctx):
    """SubspaceNet CLI v2."""
    ctx.ensure_object(dict)
    dcd_music_path = Path(__file__).resolve().parent.parent / "DCD_MUSIC"
    if dcd_music_path.exists():
        sys.path.append(str(dcd_music_path.parent))


cli.help = CLI_GROUP_HELP

_register_commands()
