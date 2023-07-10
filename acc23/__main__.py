# pylint: disable=import-outside-toplevel
"""Entry point"""


import os
import sys
from pathlib import Path

import click
from loguru import logger as logging


def _setup_logging(logging_level: str) -> None:
    """
    Sets logging format and level. The format is

        %(asctime)s [%(levelname)-8s] %(message)s

    e.g.

        2022-02-01 10:41:43,797 [INFO    ] Hello world
        2022-02-01 10:42:12,488 [CRITICAL] We're out of beans!

    Args:
        logging_level (str): Either 'critical', 'debug', 'error', 'info', or
            'warning', case insensitive. If invalid, defaults to 'info'.
    """
    logging.remove()
    logging.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            + "[<level>{level: <8}</level>] "
            + "<level>{message}</level>"
        ),
        level=logging_level.upper(),
        enqueue=True,
        colorize=True,
    )


@click.group()
@click.option(
    "--logging-level",
    default=os.getenv("LOGGING_LEVEL", "info"),
    help=(
        "Logging level, among 'critical', 'debug', 'error', 'info', and "
        "'warning', case insensitive."
    ),
    type=click.Choice(
        ["critical", "debug", "error", "info", "warning"],
        case_sensitive=False,
    ),
)
def main(logging_level: str):
    """Entrypoint."""
    _setup_logging(logging_level)


@main.command()
@click.argument(
    "csv_file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@click.argument(
    "output_file",
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        path_type=Path,
    ),
)
def preprocess(csv_file: Path, output_file: Path, *_, **__) -> None:
    """Preprocess and impute a CSV (train or test) file"""
    from acc23.preprocessing import load_csv

    logging.info("Preprocessing file '{}'", csv_file)
    df = load_csv(csv_file)
    logging.info("Saving to '{}'", output_file)
    df.to_csv(output_file, index=False)


@main.command()
@click.argument(
    "csv_file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@click.argument(
    "ipynb_file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "-c",
    "--challenge-id",
    default=1439,  # Allergen Chip Challenge
    help="Trustii challenge id",
    show_default=True,
    type=int,
)
@click.option(
    "-t",
    "--token",
    default="",
    show_default=True,
    type=str,
)
def submit(
    csv_file: Path, ipynb_file: Path, challenge_id: int, token: str, *_, **__
):
    """Submits a run to trustii"""

    from acc23.postprocessing import submit_to_trustii

    if not token:
        logging.error("No token provided")
        sys.exit(1)
    submit_to_trustii(csv_file, ipynb_file, challenge_id, token)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
