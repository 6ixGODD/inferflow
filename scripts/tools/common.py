"""Common Utilities for CLI Scripts.

Provides logging, formatting, and helper functions for command-line
tools
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys
import textwrap
import typing as t

import halo

from scripts.tools.ansi import ANSI

spinner = halo.Halo(spinner="dots")


def success(message: str, /, prefix: str = "✓") -> None:
    """Print a success message."""
    print(f"{ANSI.success(prefix)} {message}")


def error(message: str, /, prefix: str = "✗") -> None:
    """Print an error message."""
    print(f"{ANSI.error(prefix)} {message}", file=sys.stderr)


def warning(message: str, /, prefix: str = "⚠") -> None:
    """Print a warning message."""
    print(f"{ANSI.warning(prefix)} {message}")


def info(message: str, /, prefix: str = "ℹ") -> None:
    """Print an info message."""
    print(f"{ANSI.info(prefix)} {message}")


def debug(message: str, /, prefix: str = "→") -> None:
    """Print a debug message (dimmed)."""
    print(ANSI.format(f"{prefix} {message}", ANSI.STYLE.DIM))


def step(message: str, /, step: int | None = None) -> None:
    """Print a step message in a process."""
    if step is not None:
        prefix = ANSI.format(f"[{step}]", ANSI.FG.CYAN, ANSI.STYLE.BOLD)
        print(f"\n{prefix} {message}")
    else:
        print(f"\n▸ {message}")


def path(
    path: str | pathlib.Path | os.PathLike[str],
    /,
    label: str | None = None,
    exists: bool | None = None,
) -> None:
    """Print a formatted file path."""
    path = pathlib.Path(path)

    parts = []
    if label:
        parts.append(ANSI.format(f"{label}:", ANSI.STYLE.BOLD))

    if exists is not None:
        indicator = ANSI.success("✓") if exists else ANSI.error("✗")
        parts.append(indicator)

    parts.append(ANSI.format(str(path), ANSI.FG.CYAN))
    print(" ".join(parts))


def command(cmd: str, /) -> None:
    """Print a command being executed."""
    print(ANSI.format(f"  $ {cmd}", ANSI.FG.GRAY, ANSI.STYLE.DIM))


def header(text: str, /) -> None:
    """Print a section header."""
    print()
    print(ANSI.format(text, ANSI.STYLE.BOLD))
    print(ANSI.format("─" * len(text), ANSI.FG.GRAY))


def separator(char: str = "─", length: int = 60) -> None:
    """Print a separator line."""
    print(ANSI.format(char * length, ANSI.FG.GRAY))


def key_value(
    data: dict[str, t.Any],
    /,
    indent: int = 0,
) -> None:
    """Print key-value pairs in a clean format."""
    if not data:
        return

    max_key_len = max(len(str(k)) for k in data)
    indent_str = "  " * indent

    for key, value in data.items():
        key_str = ANSI.format(str(key).ljust(max_key_len), ANSI.STYLE.BOLD)
        value_str = str(value)
        print(f"{indent_str}{key_str}  {value_str}")


def list_items(
    items: list[str],
    /,
    bullet: str = "•",
    indent: int = 0,
) -> None:
    """Print a bulleted list of items."""
    indent_str = "  " * indent
    for item in items:
        print(f"{indent_str}{ANSI.format(bullet, ANSI.FG.CYAN)} {item}")


@contextlib.contextmanager
def loading(
    text: str = "Loading",
    /,
    success_text: str | None = None,
    error_text: str | None = None,
) -> t.Generator[t.Any, None, None]:
    """Context manager for showing a loading spinner."""
    spinner.text = text
    spinner.start()

    try:
        yield spinner
        if success_text:
            spinner.succeed(success_text)
        else:
            spinner.succeed()
    except Exception as e:
        if error_text:
            spinner.fail(error_text)
        else:
            spinner.fail(f"{text} failed: {e}")
        raise
    finally:
        spinner.stop()


@contextlib.contextmanager
def section(title: str, /) -> t.Generator[None, None, None]:
    """Context manager for a named section."""
    print()
    print(ANSI.format(f"┌─ {title}", ANSI.FG.CYAN, ANSI.STYLE.BOLD))

    try:
        yield
    finally:
        print(ANSI.format("└─" + "─" * (len(title) + 2), ANSI.FG.CYAN))


def banner(text: str, /, subtitle: str | None = None, version: str | None = None) -> None:
    """Print an application banner."""
    width = max(len(text), len(subtitle) if subtitle else 0) + 4

    print()
    print(ANSI.format("┌" + "─" * (width - 2) + "┐", ANSI.FG.CYAN, ANSI.STYLE.BOLD))
    print(ANSI.format(f"│ {text.center(width - 4)} │", ANSI.FG.CYAN, ANSI.STYLE.BOLD))

    if subtitle:
        print(ANSI.format(f"│ {subtitle.center(width - 4)} │", ANSI.FG.CYAN))

    if version:
        version_text = f"v{version}"
        print(ANSI.format(f"│ {version_text.center(width - 4)} │", ANSI.FG.GRAY))

    print(ANSI.format("└" + "─" * (width - 2) + "┘", ANSI.FG.CYAN, ANSI.STYLE.BOLD))
    print()


def confirm(prompt: str, /, default: bool = False) -> bool:
    """Ask for user confirmation with a yes/no prompt."""
    suffix = ANSI.format("[Y/n]" if default else "[y/N]", ANSI.STYLE.DIM)
    response = input(f"{prompt} {suffix} ").strip().lower()

    if not response:
        return default

    if response in ("y", "yes"):
        return True
    if response in ("n", "no"):
        return False

    error("Please answer 'y' or 'n'")
    return confirm(prompt, default)


def prompt_input(message: str, /, default: str = "") -> str:
    """Prompt user for input."""
    if default:
        default_text = ANSI.format(f"[{default}]", ANSI.STYLE.DIM)
        full_message = f"{message} {default_text}:  "
    else:
        full_message = f"{message}: "

    response = input(full_message).strip()
    return response if response else default


def exception_detail(exc: Exception, /, show_traceback: bool = False) -> None:
    """Display detailed exception information."""
    print()
    print(ANSI.error("┌" + "─" * 68 + "┐"))
    print(ANSI.error(f"│ ERROR: {type(exc).__name__}".ljust(70) + "│"))
    print(ANSI.error("├" + "─" * 68 + "┤"))

    # Wrap error message
    msg = str(exc)
    for line in textwrap.wrap(msg, width=66):
        print(ANSI.error(f"│ {line}".ljust(70) + "│"))

    print(ANSI.error("└" + "─" * 68 + "┘"))

    if show_traceback:
        print()
        print(ANSI.format("Traceback:", ANSI.STYLE.DIM))
        print(ANSI.format("─" * 70, ANSI.FG.GRAY))
        import traceback

        traceback.print_exception(type(exc), exc, exc.__traceback__)


def error_summary(
    title: str,
    /,
    details: dict[str, str] | None = None,
    suggestions: list[str] | None = None,
) -> None:
    """Display an error summary with optional details and suggestions."""
    print()
    print(ANSI.error(f"✗ {title}"))
    print()

    if details:
        print(ANSI.format("Details:", ANSI.STYLE.BOLD))
        key_value(details, indent=1)
        print()

    if suggestions:
        print(ANSI.format("Suggestions:", ANSI.STYLE.BOLD))
        list_items(suggestions, bullet="→", indent=1)
        print()


def fatal_error(message: str, /, exit_code: int = 1) -> t.NoReturn:
    """Display a fatal error and exit."""
    print()
    print(ANSI.error("╔" + "═" * 68 + "╗"))
    print(ANSI.error("║" + " FATAL ERROR ".center(68) + "║"))
    print(ANSI.error("╠" + "═" * 68 + "╣"))

    for line in textwrap.wrap(message, width=66):
        print(ANSI.error("║ " + line.ljust(67) + "║"))

    print(ANSI.error("╚" + "═" * 68 + "╝"))
    print()
    sys.exit(exit_code)


def show_error(exc: Exception, /, verbose: bool = False) -> None:
    """Display error information."""
    error(f"{type(exc).__name__}: {exc}")

    if verbose:
        print()
        print(ANSI.format("Traceback:", ANSI.STYLE.DIM))
        import traceback

        traceback.print_exception(type(exc), exc, exc.__traceback__)


def table_dict(
    data: dict[str, t.Any],
    /,
) -> None:
    """Print a simple table from a dictionary."""
    if not data:
        return

    max_key_len = max(len(str(k)) for k in data)

    for key, value in data.items():
        key_str = ANSI.format(str(key).ljust(max_key_len), ANSI.STYLE.BOLD)
        print(f"  {key_str}  {value}")


# Utility functions for file operations
def ensure_dir(directory: str | pathlib.Path, /) -> pathlib.Path:
    """Ensure a directory exists, create if it doesn't."""
    path = pathlib.Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_file_exists(filepath: pathlib.Path, /, force: bool = False) -> bool:
    """Check if file exists and handle accordingly.

    Returns:
        True if should proceed with overwrite, False otherwise
    """
    if not filepath.exists():
        return False

    if force:
        warning(f"Overwriting existing file: {filepath}")
        return True

    warning(f"File already exists: {filepath}")
    if confirm("Overwrite?", default=False):
        info("Overwriting file...")
        return True

    info("Skipping file generation")
    return False


def dir_exists(directory: str | pathlib.Path, /) -> bool:
    """Check if a directory exists."""
    return pathlib.Path(directory).is_dir()


def file_exists(filepath: str | pathlib.Path, /) -> bool:
    """Check if a file exists."""
    return pathlib.Path(filepath).is_file()


# Formatting helper functions
def format_path(path: str | pathlib.Path, /) -> str:
    """Format a file path with color."""
    return ANSI.format(str(path), ANSI.FG.CYAN)


def format_command(command: str, /) -> str:
    """Format a command with color."""
    return ANSI.format(command, ANSI.FG.BRIGHT_CYAN)


def format_key(key: str, /) -> str:
    """Format a key/identifier with color."""
    return ANSI.format(key, ANSI.FG.YELLOW)


def format_code(text: str, /) -> str:
    """Format inline code."""
    return ANSI.format(text, ANSI.FG.BRIGHT_MAGENTA)


def format_value(value: str, /) -> str:
    """Format a value with color."""
    return ANSI.format(value, ANSI.FG.GREEN)


def format_dim(text: str, /) -> str:
    """Format text as dimmed."""
    return ANSI.format(text, ANSI.STYLE.DIM)


def format_bold(text: str, /) -> str:
    """Format text as bold."""
    return ANSI.format(text, ANSI.STYLE.BOLD)


def init_ansi_formatter() -> None:
    """Initialize ANSI formatter based on environment."""
    if not ANSI.supports_color():
        ANSI.enable(False)


def setup_quiet_mode() -> None:
    """Redirect stdout to devnull for quiet mode."""
    sys.stdout = open(os.devnull, "w")  # noqa: PTH123, SIM115
