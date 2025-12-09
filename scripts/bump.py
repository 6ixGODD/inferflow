#!/usr/bin/env python3
"""Version Bump Script.

This script bumps version across multiple files and creates git tags.
Supports PEP 440 version format and provides dry-run mode for safety.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

from scripts.tools import common

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
VERSION_FILE = PROJECT_ROOT / "VERSION"
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"
PROJECT_NAME = "inferflow"
INIT_FILE = PROJECT_ROOT / PROJECT_NAME / "__init__.py"

# PEP 440 version regex
# Supports:  X.Y.Z, X.Y.ZaN, X.Y.ZbN, X.Y.ZrcN, X.Y.Z.postN, X.Y.Z.devN
VERSION_PATTERN = re.compile(r"^[0-9]+(\.[0-9]+){2}((a|b|rc)[0-9]+|\.post[0-9]+|\.dev[0-9]+)?$")


# ============================================================================
# Version Operations
# ============================================================================


def validate_version(version: str) -> bool:
    """Validate version format against PEP 440.

    Args:
        version:  Version string to validate

    Returns:
        True if valid, False otherwise
    """
    if not VERSION_PATTERN.match(version):
        common.error(
            f"Invalid version format: {version}\n"
            f"Expected PEP 440 format (e.g., 0.1.0, 0.1.0a1, 0.1.0.post1, 0.1.0.dev1)"
        )
        return False

    common.success(f"Version format validated (PEP 440): {version}")
    return True


def get_current_version() -> str:
    """Get current version from VERSION file.

    Returns:
        Current version string or "unknown" if not found
    """
    if VERSION_FILE.exists():
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    return "unknown"


def update_version_file(version: str, dry_run: bool = False) -> bool:
    """Update VERSION file.

    Args:
        version:  New version string
        dry_run: If True, only show what would be done

    Returns:
        True if successful
    """
    common.step("Updating VERSION file...")

    if dry_run:
        common.info(f"[DRY-RUN] Would write '{version}' to {VERSION_FILE.name}")
        return True

    try:
        VERSION_FILE.write_text(f"{version}\n", encoding="utf-8")
        common.success(f"✓ Updated:  {VERSION_FILE.name}")
        return True
    except Exception as e:
        common.error(f"Failed to update {VERSION_FILE.name}: {e}")
        return False


def update_pyproject_toml(version: str, dry_run: bool = False) -> bool:
    """Update version in pyproject.toml.

    Args:
        version: New version string
        dry_run:  If True, only show what would be done

    Returns:
        True if successful
    """
    common.step("Updating pyproject.toml...")

    if not PYPROJECT_FILE.exists():
        common.warning(f"File not found: {PYPROJECT_FILE.name}")
        return False

    if dry_run:
        common.info(f"[DRY-RUN] Would update version in {PYPROJECT_FILE.name}")
        return True

    try:
        content = PYPROJECT_FILE.read_text(encoding="utf-8")
        original_content = content

        # Update [tool.poetry] section
        poetry_pattern = r'(\[tool\.poetry\].*?version\s*=\s*")[^"]+(")'
        if re.search(poetry_pattern, content, re.DOTALL):
            content = re.sub(
                poetry_pattern,
                rf"\g<1>{version}\g<2>",
                content,
                flags=re.DOTALL,
            )
            common.success(f"✓ Updated: {PYPROJECT_FILE.name} ([tool.poetry] section)")

        # Update [project] section if exists
        project_pattern = r'(\[project\].*?version\s*=\s*")[^"]+(")'
        if re.search(project_pattern, content, re.DOTALL):
            content = re.sub(
                project_pattern,
                rf"\g<1>{version}\g<2>",
                content,
                flags=re.DOTALL,
            )
            common.success(f"✓ Updated:  {PYPROJECT_FILE.name} ([project] section)")

        if content == original_content:
            common.warning(
                f"No version fields found in {PYPROJECT_FILE.name} (checked [tool.poetry] and [project] sections)"
            )
            return False

        PYPROJECT_FILE.write_text(content, encoding="utf-8")
        return True

    except Exception as e:
        common.error(f"Failed to update {PYPROJECT_FILE.name}: {e}")
        return False


def update_init_py(version: str, dry_run: bool = False) -> bool:
    """Update __version__ in __init__.py.

    Args:
        version: New version string
        dry_run:  If True, only show what would be done

    Returns:
        True if successful
    """
    common.step(f"Updating {PROJECT_NAME}/__init__.py...")

    if not INIT_FILE.exists():
        common.warning(f"File not found: {INIT_FILE.relative_to(PROJECT_ROOT)}")
        return False

    if dry_run:
        common.info(f"[DRY-RUN] Would update __version__ in {INIT_FILE.name}")
        return True

    try:
        content = INIT_FILE.read_text(encoding="utf-8")

        # Update __version__ variable
        pattern = r'__version__\s*=\s*["\'][^"\']+["\']'
        if not re.search(pattern, content):
            common.warning(f"__version__ not found in {INIT_FILE.name}")
            return False

        updated_content = re.sub(
            pattern,
            f'__version__ = "{version}"',
            content,
        )

        INIT_FILE.write_text(updated_content, encoding="utf-8")
        common.success(f"✓ Updated: {INIT_FILE.relative_to(PROJECT_ROOT)}")
        return True

    except Exception as e:
        common.error(f"Failed to update {INIT_FILE.name}: {e}")
        return False


def verify_updates(version: str) -> bool:
    """Verify all files were updated correctly.

    Args:
        version: Expected version string

    Returns:
        True if all files match expected version
    """
    common.step("Verifying updates...")

    errors = 0

    # Check VERSION file
    if VERSION_FILE.exists():
        current = VERSION_FILE.read_text(encoding="utf-8").strip()
        if current == version:
            common.success(f"✓ VERSION file: {current}")
        else:
            common.error(f"✗ VERSION file: expected {version}, found {current}")
            errors += 1

    # Check pyproject.toml
    if PYPROJECT_FILE.exists():
        content = PYPROJECT_FILE.read_text(encoding="utf-8")
        if f'version = "{version}"' in content:
            common.success(f"✓ pyproject.toml: {version}")
        else:
            common.error("✗ pyproject.toml: version not found or incorrect")
            errors += 1

    # Check __init__.py
    if INIT_FILE.exists():
        content = INIT_FILE.read_text(encoding="utf-8")
        if f'__version__ = "{version}"' in content:
            common.success(f"✓ __init__.py: {version}")
        else:
            common.error("✗ __init__.py: version not found or incorrect")
            errors += 1

    if errors > 0:
        print()
        common.error(f"Verification failed with {errors} error(s)")
        return False

    print()
    common.success("All files verified successfully")
    return True


# ============================================================================
# Git Operations
# ============================================================================


def check_git_available() -> bool:
    """Check if git is available and we're in a git repository.

    Returns:
        True if git is available and in a repo
    """
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            check=True,
            capture_output=True,
            cwd=PROJECT_ROOT,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_current_branch() -> str | None:
    """Get current git branch name.

    Returns:
        Branch name or None if not available
    """
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def create_git_commit(version: str, dry_run: bool = False) -> bool:
    """Create git commit for version bump.

    Args:
        version: New version string
        dry_run:  If True, only show what would be done

    Returns:
        True if successful
    """
    if not check_git_available():
        common.warning("Git not available, skipping commit")
        return False

    common.step("Creating git commit...")

    if dry_run:
        common.info(f"[DRY-RUN] Would commit with message: 'chore: bump version to {version}'")
        return True

    try:
        # Add files
        files_to_add = [VERSION_FILE, PYPROJECT_FILE, INIT_FILE]
        for file in files_to_add:
            if file.exists():
                subprocess.run(
                    ["git", "add", str(file)],
                    check=True,
                    cwd=PROJECT_ROOT,
                )

        # Check if there are changes
        result = subprocess.run(
            ["git", "diff", "--staged", "--quiet"],
            cwd=PROJECT_ROOT,
        )

        if result.returncode == 0:
            common.warning("No changes to commit")
            return True

        # Create commit
        subprocess.run(
            ["git", "commit", "-m", f"chore: bump version to {version}"],
            check=True,
            cwd=PROJECT_ROOT,
        )

        common.success("✓ Git commit created")
        return True

    except subprocess.CalledProcessError as e:
        common.error(f"Failed to create commit: {e}")
        return False


def check_tag_exists(tag: str) -> bool:
    """Check if a git tag exists.

    Args:
        tag: Tag name

    Returns:
        True if tag exists
    """
    try:
        subprocess.run(
            ["git", "rev-parse", tag],
            check=True,
            capture_output=True,
            cwd=PROJECT_ROOT,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def create_git_tag(version: str, dry_run: bool = False, force: bool = False) -> bool:
    """Create git tag for version.

    Args:
        version:  New version string
        dry_run: If True, only show what would be done
        force: Force tag creation even if it exists

    Returns:
        True if successful
    """
    if not check_git_available():
        common.warning("Git not available, skipping tag")
        return False

    common.step("Creating git tag...")

    tag_name = f"v{version}"

    # Check if tag exists
    if check_tag_exists(tag_name):
        common.warning(f"Tag {tag_name} already exists")
        if not dry_run and not force:
            if common.confirm("Do you want to delete and recreate the tag?", default=False):
                try:
                    subprocess.run(
                        ["git", "tag", "-d", tag_name],
                        check=True,
                        cwd=PROJECT_ROOT,
                    )
                    common.info(f"Deleted existing tag: {tag_name}")
                except subprocess.CalledProcessError:
                    common.error(f"Failed to delete tag: {tag_name}")
                    return False
            else:
                return True

    if dry_run:
        common.info(f"[DRY-RUN] Would create tag:  {tag_name}")
        return True

    try:
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", f"Release version {version}"],
            check=True,
            cwd=PROJECT_ROOT,
        )
        common.success(f"✓ Git tag created: {tag_name}")
        print()
        common.info(f"To push the tag, run: git push origin {tag_name}")
        return True

    except subprocess.CalledProcessError as e:
        common.error(f"Failed to create tag: {e}")
        return False


def push_to_remote(version: str, dry_run: bool = False) -> bool:
    """Push commit and tag to remote.

    Args:
        version:  New version string
        dry_run: If True, only show what would be done

    Returns:
        True if successful
    """
    if not check_git_available():
        return False

    common.step("Pushing to remote...")

    tag_name = f"v{version}"
    branch = get_current_branch()

    if not branch:
        common.error("Could not determine current branch")
        return False

    if dry_run:
        common.info(f"[DRY-RUN] Would push:  git push origin {branch}")
        common.info(f"[DRY-RUN] Would push:  git push origin {tag_name}")
        return True

    # Ask for confirmation
    if not common.confirm("Do you want to push commit and tag to remote?", default=False):
        common.info("Skipping push. Run manually:")
        print(f"  git push origin {branch}")
        print(f"  git push origin {tag_name}")
        return True

    try:
        # Push commit
        subprocess.run(
            ["git", "push", "origin", branch],
            check=True,
            cwd=PROJECT_ROOT,
        )
        common.success("✓ Pushed commit to remote")

        # Push tag
        subprocess.run(
            ["git", "push", "origin", tag_name],
            check=True,
            cwd=PROJECT_ROOT,
        )
        common.success(f"✓ Pushed tag to remote: {tag_name}")
        return True

    except subprocess.CalledProcessError as e:
        common.error(f"Failed to push: {e}")
        return False


# ============================================================================
# Summary
# ============================================================================


def show_summary(old_version: str, new_version: str, no_git: bool = False, no_push: bool = False) -> None:
    """Show summary of version bump.

    Args:
        old_version: Old version string
        new_version: New version string
        no_git: Whether git operations were skipped
        no_push:  Whether push was skipped
    """
    common.separator()
    common.info("Version Bump Summary:")
    print()
    print(f"  Old Version: {common.format_bold(old_version)}")
    print(f"  New Version:  {common.format_bold(new_version)}")
    print()
    print("  Updated files:")
    print(f"    - {VERSION_FILE.name}")
    print(f"    - {PYPROJECT_FILE.name}")
    print(f"    - {INIT_FILE.relative_to(PROJECT_ROOT)}")
    print()

    if not no_git:
        print("  Git operations:")
        print("    - Commit created")
        print(f"    - Tag created: v{new_version}")
        print()

        if no_push:
            common.info("Next steps:")
            branch = get_current_branch()
            if branch:
                print(f"    git push origin {branch}")
            print(f"    git push origin v{new_version}")

    common.separator()


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Bump version across multiple files and git tag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 0.2.0                # Bump to version 0.2.0
  %(prog)s 0.2.0 --dry-run      # Show what would be changed
  %(prog)s 0.2.0 --no-git       # Skip git operations
  %(prog)s 0.2.0 --no-push      # Don't push to remote
  %(prog)s 0.2.0a1              # Bump to alpha version
  %(prog)s 0.2.0.post1          # Bump to post-release version
        """,
    )

    parser.add_argument(
        "version",
        help="New version number (e.g., 0.2.0, 0.2.0a1, 0.2.0.post1)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )

    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Skip git operations (commit and tag)",
    )

    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push to remote (implies manual push)",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force operation without confirmation prompts",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Initialize ANSI formatter
    common.init_ansi_formatter()

    common.header("Version Bump")

    # Validate version format
    if not validate_version(args.version):
        sys.exit(1)

    print()

    # Get current version
    current_version = get_current_version()
    common.info(f"Current version: {common.format_bold(current_version)}")
    common.info(f"New version: {common.format_bold(args.version)}")
    print()

    # Confirm if not dry-run and not forced
    if not args.dry_run and not args.force:
        if not common.confirm("Do you want to proceed with version bump?", default=True):
            common.info("Version bump cancelled")
            sys.exit(0)
        print()

    # Update files
    success = True
    success &= update_version_file(args.version, dry_run=args.dry_run)
    success &= update_pyproject_toml(args.version, dry_run=args.dry_run)
    success &= update_init_py(args.version, dry_run=args.dry_run)
    print()

    if not success:
        common.error("Failed to update all files")
        sys.exit(1)

    # Verify updates (skip for dry-run)
    if not args.dry_run and not verify_updates(args.version):
        sys.exit(1)

    # Git operations
    if not args.dry_run and not args.no_git:
        print()
        create_git_commit(args.version, dry_run=args.dry_run)
        create_git_tag(args.version, dry_run=args.dry_run, force=args.force)

        if not args.no_push:
            push_to_remote(args.version, dry_run=args.dry_run)

    elif args.dry_run and not args.no_git:
        print()
        create_git_commit(args.version, dry_run=True)
        create_git_tag(args.version, dry_run=True)
        if not args.no_push:
            push_to_remote(args.version, dry_run=True)

    # Show summary
    print()
    if args.dry_run:
        common.info("Dry-run complete. No files were modified.")
        print()
    else:
        show_summary(
            current_version,
            args.version,
            no_git=args.no_git,
            no_push=args.no_push,
        )
        common.success("Version bump complete!")

    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        common.warning("Operation cancelled by user")
        print()
        sys.exit(130)
    except Exception as e:
        print()
        common.error(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        print()
        sys.exit(1)
