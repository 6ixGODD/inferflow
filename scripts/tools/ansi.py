from __future__ import annotations

import enum
import os
import sys


class ANSI:
    """Enhanced formatter for ANSI color and style formatting in console
    output.

    Provides organized color constants, helper methods, and context
    managers for applying consistent styling to terminal output.
    Automatically detects color support in the terminal environment.
    """

    class FG(enum.StrEnum):
        """Foreground colors."""

        BLACK = "\033[30m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"
        GRAY = "\033[90m"
        BRIGHT_RED = "\033[91m"
        BRIGHT_GREEN = "\033[92m"
        BRIGHT_YELLOW = "\033[93m"
        BRIGHT_BLUE = "\033[94m"
        BRIGHT_MAGENTA = "\033[95m"
        BRIGHT_CYAN = "\033[96m"
        BRIGHT_WHITE = "\033[97m"

    class BG(enum.StrEnum):
        """Background colors."""

        BLACK = "\033[40m"
        RED = "\033[41m"
        GREEN = "\033[42m"
        YELLOW = "\033[43m"
        BLUE = "\033[44m"
        MAGENTA = "\033[45m"
        CYAN = "\033[46m"
        WHITE = "\033[47m"
        GRAY = "\033[100m"
        BRIGHT_RED = "\033[101m"
        BRIGHT_GREEN = "\033[102m"
        BRIGHT_YELLOW = "\033[103m"
        BRIGHT_BLUE = "\033[104m"
        BRIGHT_MAGENTA = "\033[105m"
        BRIGHT_CYAN = "\033[106m"
        BRIGHT_WHITE = "\033[107m"

    class STYLE(enum.StrEnum):
        """Text styles."""

        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        ITALIC = "\033[3m"
        UNDERLINE = "\033[4m"
        BLINK = "\033[5m"
        REVERSE = "\033[7m"
        HIDDEN = "\033[8m"
        STRIKETHROUGH = "\033[9m"

    # For backward compatibility
    RESET = STYLE.RESET
    BOLD = STYLE.BOLD
    UNDERLINE = STYLE.UNDERLINE
    REVERSED = STYLE.REVERSE
    RED = FG.BRIGHT_RED
    GREEN = FG.BRIGHT_GREEN
    YELLOW = FG.BRIGHT_YELLOW
    BLUE = FG.BRIGHT_BLUE
    MAGENTA = FG.BRIGHT_MAGENTA
    CYAN = FG.BRIGHT_CYAN
    WHITE = FG.BRIGHT_WHITE

    # Control whether ANSI colors are enabled
    _enabled = True

    @classmethod
    def supports_color(cls) -> bool:
        """Determine if the current terminal supports colors.

        Returns:
            bool: True if the terminal supports colors, False otherwise.
        """
        # Check for NO_COLOR environment variable (https://no-color.org/)
        if os.environ.get("NO_COLOR", ""):
            return False

        # Check for explicit color control
        if os.environ.get("FORCE_COLOR", ""):
            return True

        # Check if stdout is a TTY
        return hasattr(sys.stdout, "isatty") or sys.stdout.isatty()

    @classmethod
    def enable(cls, enabled: bool = True) -> None:
        """Enable or disable ANSI formatting.

        Args:
            enabled: True to enable colors, False to disable.
        """
        cls._enabled = enabled and cls.supports_color()

    @classmethod
    def format(cls, text: str, /, *styles: STYLE | FG | BG | None) -> str:
        """Format text with the specified ANSI styles.

        Intelligently reapplies styles after any reset sequences in the text.
        If colors are disabled, returns the original text without formatting.

        Args:
            text: The text to format.
            *styles: One or more ANSI style codes to apply.

        Returns:
            The formatted text with ANSI styles applied.
        """
        if not cls._enabled or not styles:
            return text
        if any(s is None for s in styles):
            styles = tuple(s for s in styles if s is not None)

        style_str = "".join(styles)

        # Handle text that already contains reset codes
        if cls.STYLE.RESET in text:
            text = text.replace(cls.STYLE.RESET, f"{cls.STYLE.RESET}{style_str}")

        return f"{style_str}{text}{cls.STYLE.RESET}"

    @classmethod
    def success(cls, text: str, /) -> str:
        """Format text as a success message (green, bold)."""
        return cls.format(text, cls.FG.BRIGHT_GREEN, cls.STYLE.BOLD)

    @classmethod
    def error(cls, text: str, /) -> str:
        """Format text as an error message (red, bold)."""
        return cls.format(text, cls.FG.BRIGHT_RED, cls.STYLE.BOLD)

    @classmethod
    def warning(cls, text: str, /) -> str:
        """Format text as a warning message (yellow, bold)."""
        return cls.format(text, cls.FG.BRIGHT_YELLOW, cls.STYLE.BOLD)

    @classmethod
    def info(cls, text: str, /) -> str:
        """Format text as an info message (cyan)."""
        return cls.format(text, cls.FG.BRIGHT_CYAN)

    @classmethod
    def highlight(cls, text: str, /) -> str:
        """Format text as highlighted (magenta, bold)."""
        return cls.format(text, cls.FG.BRIGHT_MAGENTA, cls.STYLE.BOLD)

    @classmethod
    def rgb(cls, text: str, /, r: int, g: int, b: int, background: bool = False) -> str:
        """Format text with a specific RGB color.

        Args:
            text: The text to format
            r: R
            g: G
            b: B
            background: If True, set as background color instead of foreground

        Returns:
            Formatted text with the specified RGB color
        """
        if not cls._enabled:
            return text

        code = 48 if background else 38
        color_seq = f"\033[{code};2;{r};{g};{b}m"
        return f"{color_seq}{text}{cls.STYLE.RESET}"
