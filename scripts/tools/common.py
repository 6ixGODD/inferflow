"""Common Utilities for CLI Scripts Provides logging, formatting, and
helper functions for command-line tools."""

from __future__ import annotations

import pathlib as pl
import sys
import typing as t


def print_separator(char: str = "=", length: int = 76) -> None:
    """Print a separator line.

    Args:
        char: Character to use for the separator
        length: Length of the separator line
    """
    print(ansi.ANSI.format(char * length, ansi.ANSI.STYLE.DIM))


def print_header(title: str) -> None:
    """Print a formatted header.

    Args:
        title: Header title text
    """
    print()
    print_separator()
    print(ansi.ANSI.format(title, ansi.ANSI.STYLE.BOLD, ansi.ANSI.FG.CYAN))
    print_separator()
    print()


def log_info(message: str) -> None:
    """Print an info message.

    Args:
        message: Info message to display
    """
    icon = ansi.ANSI.format("[I]", ansi.FG.BLUE)
    print(f"{icon} {message}")


def log_success(message: str) -> None:
    """Print a success message.

    Args:
        message: Success message to display
    """
    icon = ansi.ANSI.format("[✓]", ansi.ANSI.FG.GREEN)
    print(f"{icon} {message}")


def log_warn(message: str) -> None:
    """Print a warning message.

    Args:
        message: Warning message to display
    """
    icon = ansi.ANSI.format("[W]", ansi.ANSI.FG.YELLOW)
    print(f"{icon} {message}")


def log_error(message: str) -> None:
    """Print an error message.

    Args:
        message: Error message to display
    """
    icon = ansi.ANSI.format("[E]", ansi.ANSI.FG.RED)
    print(f"{icon} {message}", file=sys.stderr)


def log_step(message: str) -> None:
    """Print a step message.

    Args:
        message: Step message to display
    """
    icon = ansi.ANSI.format("[*]", ansi.ANSI.FG.MAGENTA)
    print(f"{icon} {message}")


def die(message: str, exit_code: int = 1) -> t.NoReturn:
    """Print an error message and exit.

    Args:
        message: Error message to display
        exit_code: Exit code to use (default: 1)
    """
    print()
    log_error(f"Error: {message}")
    print()
    sys.exit(exit_code)


def format_path(path: str | pl.Path) -> str:
    """Format a file path with color.

    Args:
        path: Path to format

    Returns:
        Formatted path string
    """
    return ansi.ANSI.format(str(path), ansi.ANSI.FG.CYAN)


def format_command(command: str) -> str:
    """Format a command with color.

    Args:
        command: Command to format

    Returns:
        Formatted command string
    """
    return ansi.ANSI.format(command, ansi.ANSI.FG.BRIGHT_CYAN)


def format_key(key: str) -> str:
    """Format a key/identifier with color.

    Args:
        key: Key to format

    Returns:
        Formatted key string
    """
    return ansi.ANSI.format(key, ansi.ANSI.FG.YELLOW)


def format_code(text: str) -> str:
    """Format inline code."""
    return ansi.ANSI.format(text, ansi.ANSI.FG.BRIGHT_MAGENTA)


def format_value(value: str) -> str:
    """Format a value with color.

    Args:
        value: Value to format

    Returns:
        Formatted value string
    """
    return ansi.ANSI.format(value, ansi.ANSI.FG.GREEN)


def format_dim(text: str) -> str:
    """Format text as dimmed.

    Args:
        text: Text to format

    Returns:
        Formatted text string
    """
    return ansi.ANSI.format(text, ansi.ANSI.STYLE.DIM)


def format_bold(text: str) -> str:
    """Format text as bold.

    Args:
        text: Text to format

    Returns:
        Formatted text string
    """
    return ansi.ANSI.format(text, ansi.ANSI.STYLE.BOLD)


def format_status_success(text: str = "SUCCESS ✓") -> str:
    """Format success status text.

    Args:
        text: Status text to format

    Returns:
        Formatted status string
    """
    return ansi.ANSI.format(text, ansi.ANSI.FG.GREEN, ansi.ANSI.STYLE.BOLD)


def format_status_failed(text: str = "FAILED ✗") -> str:
    """Format failed status text.

    Args:
        text: Status text to format

    Returns:
        Formatted status string
    """
    return ansi.ANSI.format(text, ansi.ANSI.FG.RED, ansi.ANSI.STYLE.BOLD)


def ensure_dir(directory: str | pl.Path) -> pl.Path:
    """Ensure a directory exists, create if it doesn't.

    Args:
        directory: Directory path

    Returns:
        Path object for the directory
    """
    path = pl.Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_file_exists(filepath: pl.Path, force: bool = False) -> bool:
    """Check if file exists and handle accordingly.

    Args:
        filepath: Path to check
        force: If True, skip confirmation prompt

    Returns:
        True if should proceed with overwrite, False otherwise
    """
    if not filepath.exists():
        return False

    if force:
        log_warn(f"Overwriting existing file: {filepath}")
        return True

    log_warn(f"File already exists: {filepath}")
    prompt = ansi.ANSI.format("[?]", ansi.ANSI.FG.YELLOW)
    response = input(f"{prompt} Overwrite? [y/N]: ").strip().lower()

    if response in ("y", "yes"):
        log_info("Overwriting file...")
        return True
    log_info("Skipping file generation")
    return False


def dir_exists(directory: str | pl.Path) -> bool:
    """Check if a directory exists.

    Args:
        directory: Directory path to check

    Returns:
        True if directory exists, False otherwise
    """
    return pl.Path(directory).is_dir()


def file_exists(filepath: str | pl.Path) -> bool:
    """Check if a file exists.

    Args:
        filepath: File path to check

    Returns:
        True if file exists, False otherwise
    """
    return pl.Path(filepath).is_file()


def confirm(message: str, default: bool = False) -> bool:
    """Ask user for confirmation.

    Args:
        message: Confirmation message
        default: Default response if user just presses Enter

    Returns:
        True if user confirmed, False otherwise
    """
    prompt_suffix = " [Y/n]: " if default else " [y/N]: "
    prompt = ansi.ANSI.format("?", ansi.ANSI.FG.YELLOW)
    response = input(f"{prompt} {message}{prompt_suffix}").strip().lower()

    if not response:
        return default

    return response in ("y", "yes")


def prompt_input(message: str, default: str = "") -> str:
    """Prompt user for input.

    Args:
        message: Prompt message
        default: Default value if user just presses Enter

    Returns:
        User's input or default value
    """
    prompt = ansi.ANSI.format("?", ansi.ANSI.FG.CYAN)

    if default:
        default_text = format_dim(f"[{default}]")
        full_message = f"{prompt} {message} {default_text}: "
    else:
        full_message = f"{prompt} {message}: "

    response = input(full_message).strip()
    return response if response else default


def init_ansi_formatter() -> None:
    """Initialize ANSI formatter based on environment."""
    if not ansi.ANSI.supports_color():
        ansi.ANSI.enable(False)


def setup_quiet_mode() -> None:
    """Redirect stdout to devnull for quiet mode."""
    import os

    sys.stdout = open(os.devnull, "w")  # noqa: PTH123, SIM115
