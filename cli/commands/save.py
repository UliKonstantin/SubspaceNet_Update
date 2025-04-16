"""Save command implementation."""
import sys
import click
import logging
from ..options import config_option, override_option
from config.loader import save_config
from config_handler import setup_configuration

logger = logging.getLogger('SubspaceNet')

@click.command('save')
@config_option
@click.option('--output', '-o', required=True,
              help='Output path for the configuration file')
@override_option
def save_command(config: str, output: str, override: list[str]):
    """Save a configuration to a new file."""
    try:
        config_obj, _, _ = setup_configuration(config, None, override)
        save_config(config_obj, output)
        logger.info(f"Configuration saved to {output}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}", exc_info=True)
        sys.exit(1) 