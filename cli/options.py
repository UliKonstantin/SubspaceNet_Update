"""Common CLI options for SubspaceNet commands."""
from click import option

config_option = option(
    '--config', '-c', 
    default='configs/default_config.yaml',
    help='Path to the configuration file'
)

output_option = option(
    '--output', '-o',
    default=None,
    help='Output directory for experiment results'
)

override_option = option(
    '--override', '-O',
    multiple=True,
    help='Override configuration parameter (format: key=value)'
) 