"""Show command implementation."""
import sys
import json
import click
import logging
from ..options import config_option, override_option
from config_handler import setup_configuration

logger = logging.getLogger('SubspaceNet')

@click.command('show')
@config_option
@override_option
def show_command(config: str, override: list[str]):
    """Show the configuration without running an experiment."""
    try:
        config_obj, _, _ = setup_configuration(config, None, override)
        print(json.dumps(config_obj.dict(), indent=2))
    except Exception as e:
        logger.error(f"Error showing configuration: {e}", exc_info=True)
        sys.exit(1) 