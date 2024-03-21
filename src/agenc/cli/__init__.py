import argparse
import logging
import logging.config
from pathlib import Path

from .runtime_configuration import RuntimeConfiguration

DEFAULT_RUNTIME_CONFIGURATION_PATH = Path("runtime.yaml")


def initialize(
    log_level: str | int | None = None,
    runtime_configuration: Path | None = None,
    *,
    parse_arguments: bool = True,
) -> None:
    """Initialize an experiment for commandline usage.

    This function configures commandline arguments and initializes the logging
    framework. Three sources of configuration are checked in the following
    priority:
        1. Commandline arguments
        2. Configuration file
        3. Function arguments
    That is, commandline arguments take precedence over the configuration
    file, and the configuration file takes precedence over the function
    arguments.

    Example:
        >>> import agenc
        >>> agenc.initialize(log_level="DEBUG")

    Example:
        > python my_experiment.py --log-level DEBUG

    Args:
        log_level: The logging level to use.
        runtime_configuration: The path to the runtime configuration file.
        parse_arguments: Whether to parse commandline arguments.
    """
    if parse_arguments:
        arguments = _parse_arguments()
        if arguments.log_level is not None:
            log_level = arguments.log_level
        if arguments.runtime_configuration is not None:
            runtime_configuration = arguments.runtime_configuration
        elif Path(DEFAULT_RUNTIME_CONFIGURATION_PATH).exists():
            runtime_configuration = DEFAULT_RUNTIME_CONFIGURATION_PATH

    if runtime_configuration is not None:
        configuration = RuntimeConfiguration.load_from_file(
            runtime_configuration,
        )
        if "logging" in configuration:
            logging.config.dictConfig(configuration["logging"])
            if log_level is not None:
                logging.getLogger().setLevel(log_level)
            return

    logging.basicConfig(
        level=log_level or logging.INFO,
        format="%(asctime)s [%(name)s][%(levelname)s] %(message)s",
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        type=str,
        help="The logging level to use.",
    )
    parser.add_argument(
        "--runtime-configuration",
        type=Path,
        help="The path to the runtime configuration file.",
    )
    return parser.parse_args()
