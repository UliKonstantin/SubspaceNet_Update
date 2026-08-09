"""SubspaceNet CLI v2 application."""
import sys
from pathlib import Path

import click

from cli.help_text import CLI_GROUP_HELP, RUN_SHORT_HELP

_COMMAND_SHORT_HELP = {
    "run": RUN_SHORT_HELP,
    "show": "Show the configuration without running an experiment.",
    "save": "Save a configuration to a new file.",
}


class _LazyCLI(click.Group):
    """Load heavy command modules only when a subcommand is invoked."""

    def list_commands(self, ctx):
        return list(_COMMAND_SHORT_HELP)

    def format_commands(self, ctx, formatter):
        commands = self.list_commands(ctx)
        if not commands:
            return
        max_len = max(len(name) for name in commands)
        formatter.write_paragraph()
        rows = [(name.ljust(max_len), _COMMAND_SHORT_HELP[name]) for name in commands]
        if rows:
            with formatter.indentation():
                formatter.write_dl(rows)

    def get_command(self, ctx, cmd_name):
        if cmd_name == "run":
            from cli.commands.run import run_command

            return run_command
        if cmd_name == "show":
            from cli.commands.show import show_command

            return show_command
        if cmd_name == "save":
            from cli.commands.save import save_command

            return save_command
        return None


@click.group(cls=_LazyCLI)
@click.pass_context
def cli(ctx):
    """SubspaceNet CLI v2."""
    ctx.ensure_object(dict)
    dcd_music_path = Path(__file__).resolve().parent.parent / "DCD_MUSIC"
    if dcd_music_path.exists():
        sys.path.append(str(dcd_music_path.parent))


cli.help = CLI_GROUP_HELP
