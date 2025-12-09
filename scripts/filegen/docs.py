#!/usr/bin/env python3
"""Generate MkDocs API reference documentation from source code.

This script automatically generates markdown documentation for Python
modules and updates the mkdocs.yml navigation structure.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys
import traceback
import typing as t

import yaml

from scripts.tools import common

# Add project root to path
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Configuration
# ============================================================================

MODULES_TO_DOCUMENT = ["runtime", "pipeline", "batch", "workflow", "asyncio", "types"]

SOURCE_ROOT = PROJECT_ROOT / "inferflow"
DOCS_ROOT = PROJECT_ROOT / "docs"
API_DOCS_DIR = DOCS_ROOT / "reference"
MKDOCS_CONFIG = PROJECT_ROOT / "mkdocs.yml"


# ============================================================================
# Module Information Extraction
# ============================================================================


class ModuleInfo:
    """Information about a Python module."""

    def __init__(
        self,
        name: str,
        path: pathlib.Path,
        is_package: bool,
        docstring: str | None = None,
        title: str | None = None,
    ):
        self.name = name
        self.path = path
        self.is_package = is_package
        self.docstring = docstring or ""
        self._custom_title = title
        self.submodules: list[ModuleInfo] = []

    @property
    def import_path(self) -> str:
        """Get the full import path."""
        rel_path = self.path.relative_to(SOURCE_ROOT)
        parts = rel_path.parts if self.is_package else rel_path.with_suffix("").parts
        return f"inferflow.{'.'.join(parts)}"

    @property
    def doc_path(self) -> pathlib.Path:
        """Get the documentation file path."""
        rel_path = self.path.relative_to(SOURCE_ROOT)

        if self.is_package:
            # Package -> directory with index.md
            return API_DOCS_DIR / rel_path / "index.md"
        # Module -> md file in parent directory
        parent = rel_path.parent
        name = rel_path.stem
        return API_DOCS_DIR / parent / f"{name}.md"

    @property
    def nav_title(self) -> str:
        """Get the navigation title.

        Priority:
        1. Custom __doctitle__ from module
        2. Capitalized name with underscores replaced by spaces
        """
        if self._custom_title:
            return self._custom_title

        # Fallback:  capitalize and replace underscores
        return self.name.replace("_", " ").title()

    def __repr__(self) -> str:
        type_str = "package" if self.is_package else "module"
        return f"ModuleInfo({self.name!r}, {type_str})"


def extract_module_title(file_path: pathlib.Path) -> str | None:
    """Extract __doctitle__ variable from a Python file.

    Args:
        file_path: Path to Python file

    Returns:
        Title string if found, None otherwise
    """
    try:
        with file_path.open(encoding="utf-8") as f:
            tree = ast.parse(f.read())

        # Look for __doctitle__ assignment at module level
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Check if it's assigning to __doctitle__
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "__doctitle__"
                        and isinstance(node.value, ast.Constant)
                    ):
                        return str(node.value.value)

        return None
    except Exception:
        return None


def extract_docstring(file_path: pathlib.Path) -> str | None:
    """Extract docstring from a Python file."""
    try:
        with file_path.open(encoding="utf-8") as f:
            tree = ast.parse(f.read())

        return ast.get_docstring(tree)
    except Exception:
        return None


def is_package(path: pathlib.Path) -> bool:
    """Check if a path is a Python package."""
    return path.is_dir() and (path / "__init__.py").exists()


def discover_module(module_spec: str) -> ModuleInfo | None:
    """Discover a module and its submodules.

    Args:
        module_spec: Module specification (e.g., "entity" or "filters.__init__"
            or "exceptions")

    Returns:
        ModuleInfo object or None if module not found
    """
    # Handle special case:  filters.__init__
    if module_spec.endswith(".__init__"):
        base_name = module_spec.replace(".__init__", "")
        module_path = SOURCE_ROOT / base_name.replace(".", "/")

        if not is_package(module_path):
            common.warning(f"Module {module_spec} is not a package")
            return None

        init_file = module_path / "__init__.py"
        docstring = extract_docstring(init_file)
        title = extract_module_title(init_file)

        # Create module info for the __init__.py only
        return ModuleInfo(
            name=base_name.split(".")[-1],
            path=module_path,
            is_package=True,
            docstring=docstring,
            title=title,
        )

    # Try to find the module - it could be a package directory or a .py file
    module_path_dir = SOURCE_ROOT / module_spec.replace(".", "/")
    module_path_file = SOURCE_ROOT / f"{module_spec.replace('.', '/')}.py"

    # Check if it's a package (directory with __init__.py)
    if is_package(module_path_dir):
        init_file = module_path_dir / "__init__.py"
        docstring = extract_docstring(init_file)
        title = extract_module_title(init_file)

        info = ModuleInfo(
            name=module_spec.split(".")[-1],
            path=module_path_dir,
            is_package=True,
            docstring=docstring,
            title=title,
        )

        # Recursively discover submodules
        for item in sorted(module_path_dir.iterdir()):
            if item.name.startswith("_") and item.name != "__init__.py":
                continue

            if item.is_file() and item.suffix == ".py" and item.name != "__init__.py":
                # It's a module file
                submodule_docstring = extract_docstring(item)
                submodule_title = extract_module_title(item)
                submodule = ModuleInfo(
                    name=item.stem,
                    path=item,
                    is_package=False,
                    docstring=submodule_docstring,
                    title=submodule_title,
                )
                info.submodules.append(submodule)

            elif is_package(item):
                # It's a subpackage
                submodule_spec = f"{module_spec}.{item.name}"
                submodule = discover_module(submodule_spec)
                if submodule:
                    info.submodules.append(submodule)

        return info

    # Check if it's a single .py file module
    if module_path_file.exists() and module_path_file.is_file():
        docstring = extract_docstring(module_path_file)
        title = extract_module_title(module_path_file)
        return ModuleInfo(
            name=module_spec.split(".")[-1],
            path=module_path_file,
            is_package=False,
            docstring=docstring,
            title=title,
        )

    # Module not found
    # Provide more helpful error message
    common.warning(
        f"Module {module_spec} not found. Tried:\n  - Package:  {module_path_dir}\n  - File: {module_path_file}"
    )
    return None


# ============================================================================
# Documentation Generation
# ============================================================================


def generate_module_doc(module: ModuleInfo) -> str:
    """Generate markdown documentation for a module.

    Args:
        module: ModuleInfo object

    Returns:
        Markdown content
    """
    lines = []

    # Title
    title = f"{module.name}"
    lines.append(f"# {title}")
    lines.append("")

    # Docstring
    if module.docstring:
        lines.append(module.docstring)
        lines.append("")

    # mkdocstrings reference
    lines.append(f"::: {module.import_path}")
    lines.append("    options:")
    lines.append("      show_root_heading: true")
    lines.append("      show_source: true")
    lines.append("      heading_level: 2")
    lines.append("      members_order: source")
    lines.append("      show_bases: true")
    lines.append("      show_docstring_attributes: true")
    lines.append("      show_docstring_functions: true")
    lines.append("      show_docstring_classes: true")
    lines.append("      show_docstring_modules: true")
    lines.append("      show_docstring_description: true")
    lines.append("      show_docstring_examples: true")
    lines.append("      show_docstring_other_parameters: true")
    lines.append("      show_docstring_parameters: true")
    lines.append("      show_docstring_raises: true")
    lines.append("      show_docstring_receives: true")
    lines.append("      show_docstring_returns: true")
    lines.append("      show_docstring_warns: true")
    lines.append("      show_docstring_yields: true")
    lines.append("      show_signature_annotations: true")
    lines.append("      separate_signature: true")
    lines.append("")

    # If it's a package with submodules, add links
    if module.is_package and module.submodules:
        lines.append("## Submodules")
        lines.append("")

        for submodule in module.submodules:
            # Calculate relative link
            link = f"{submodule.name}/index.md" if submodule.is_package else f"{submodule.name}.md"
            lines.append(f"- [{submodule.nav_title}]({link})")

        lines.append("")

    return "\n".join(lines)


def write_module_docs(module: ModuleInfo, force: bool = False) -> bool:
    """Write documentation for a module and its submodules.

    Args:
        module: ModuleInfo object
        force:  Overwrite existing files

    Returns:
        True if successful
    """
    doc_path = module.doc_path

    # Check if file exists
    if doc_path.exists() and not force and not common.check_file_exists(doc_path, force):
        return False

    try:
        # Create parent directory
        doc_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate content
        content = generate_module_doc(module)

        # Write file
        doc_path.write_text(content, encoding="utf-8")

        common.success(f"Generated:  {common.format_path(doc_path.relative_to(PROJECT_ROOT))}")

        # Recursively write submodule docs
        for submodule in module.submodules:
            write_module_docs(submodule, force)

        return True

    except Exception as e:
        common.error(f"Failed to write {doc_path}: {e}")
        return False


# ============================================================================
# MkDocs Configuration Update
# ============================================================================


class MkDocsConfigLoader(yaml.SafeLoader):
    """Custom YAML loader that handles MkDocs-specific tags."""


class MkDocsConfigDumper(yaml.SafeDumper):
    """Custom YAML dumper that preserves MkDocs-specific tags."""


# Tag marker class to preserve tags during round-trip
class TaggedValue:
    """Wrapper for values with custom YAML tags."""

    def __init__(self, tag: str, value: t.Any):
        self.tag = tag
        self.value = value

    def __repr__(self) -> str:
        return f"TaggedValue({self.tag!r}, {self.value!r})"


def env_constructor(loader: yaml.Loader, node: yaml.Node) -> TaggedValue:
    """Constructor for !ENV tag."""
    value = loader.construct_scalar(node)  # type: ignore
    return TaggedValue("!ENV", value)


def python_name_constructor(loader: yaml.Loader, node: yaml.Node) -> TaggedValue:
    """Constructor for !!python/name: tag."""
    value = loader.construct_scalar(node)  # type: ignore
    return TaggedValue("!!python/name:", value)


def python_object_constructor(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> TaggedValue:
    """Constructor for other !!python/* tags."""
    # tag_suffix contains the part after 'python/', e.g., 'object/apply' or 'name:'
    short_tag = f"!!python/{tag_suffix}"

    # Try to construct the value
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node)
    elif isinstance(node, yaml.MappingNode):
        value = loader.construct_mapping(node)
    else:
        value = None

    return TaggedValue(short_tag, value)


# Register constructors
MkDocsConfigLoader.add_constructor("!ENV", env_constructor)
# Note: We need to handle python/name:  specially since it's used frequently
MkDocsConfigLoader.add_constructor("tag:yaml.org,2002:python/name:", python_name_constructor)
# Handle all other python/* tags with multi_constructor
MkDocsConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", python_object_constructor)


def tagged_value_representer(dumper: MkDocsConfigDumper, data: TaggedValue) -> yaml.Node:
    """Representer for TaggedValue objects."""
    tag = data.tag
    value = data.value

    # Convert short-form tags back to full tags for certain types
    if tag == "!!python/name:":
        full_tag = "tag:yaml.org,2002:python/name:"
        return dumper.represent_scalar(full_tag, value)  # type: ignore
    if tag.startswith("!!python/"):
        # Convert !!python/xx to tag:yaml.org,2002:python/xx
        python_part = tag[len("!!python/") :]
        full_tag = f"tag:yaml.org,2002:python/{python_part}"

        if isinstance(value, dict):
            return dumper.represent_mapping(full_tag, value)  # type: ignore
        if isinstance(value, list):
            return dumper.represent_sequence(full_tag, value)  # type: ignore
        return dumper.represent_scalar(full_tag, value)  # type: ignore
    if tag == "!ENV":
        return dumper.represent_scalar("!ENV", value)  # type: ignore
    # Fallback
    return dumper.represent_scalar(tag, str(value))  # type: ignore


# Register representer
MkDocsConfigDumper.add_representer(TaggedValue, tagged_value_representer)


def preserve_tagged_values(data: t.Any) -> t.Any:
    """Recursively process data structure, keeping TaggedValue objects."""
    if isinstance(data, TaggedValue):
        # Keep TaggedValue objects as-is
        return data
    if isinstance(data, dict):
        return {k: preserve_tagged_values(v) for k, v in data.items()}
    if isinstance(data, list):
        return [preserve_tagged_values(item) for item in data]
    return data


def build_nav_structure(modules: list[ModuleInfo]) -> list[dict[str, t.Any]]:
    """Build navigation structure for API reference.

    Args:
        modules: List of ModuleInfo objects

    Returns:
        Navigation structure as nested dictionaries
    """

    def build_module_nav(module: ModuleInfo) -> dict[str, t.Any] | str:
        """Recursively build navigation for a single module.

        Args:
            module: ModuleInfo object

        Returns:
            Navigation item (dict for packages with submodules, str for leaf nodes)
        """
        if module.is_package and module.submodules:
            # Package with submodules - create nested structure
            subnav = []

            # Add package index as "Overview"
            rel_path = module.doc_path.relative_to(DOCS_ROOT)
            subnav.append({"Overview": str(rel_path.as_posix())})

            # Recursively add all submodules
            for submodule in module.submodules:
                sub_nav_item = build_module_nav(submodule)

                if isinstance(sub_nav_item, dict):
                    # Submodule returned a dict (nested structure)
                    subnav.append(sub_nav_item)
                else:
                    # Submodule returned a string (leaf node)
                    subnav.append({submodule.nav_title: sub_nav_item})

            # Return as titled section
            return {module.nav_title: subnav}

        # Leaf node (module file or empty package) - return just the path
        rel_path = module.doc_path.relative_to(DOCS_ROOT)
        return str(rel_path.as_posix())

    # Build navigation for all top-level modules
    nav_items = []

    for module in modules:
        nav_item = build_module_nav(module)

        if isinstance(nav_item, dict):
            # Module returned a dict (has structure)
            nav_items.append(nav_item)
        else:
            # Module is a single file - wrap with title
            nav_items.append({module.nav_title: nav_item})

    return nav_items


def update_mkdocs_config(modules: list[ModuleInfo], force: bool = False) -> bool:
    """Update mkdocs.yml with API reference navigation.

    Args:
        modules: List of ModuleInfo objects
        force: Force update without prompting

    Returns:
        True if successful
    """
    if not MKDOCS_CONFIG.exists():
        common.error(f"MkDocs config not found:  {MKDOCS_CONFIG}")
        return False

    try:
        # Load existing config with custom loader
        config = yaml.load(MKDOCS_CONFIG.open("r"), Loader=MkDocsConfigLoader) or {}  # type: ignore

        # Preserve all tagged values
        config = preserve_tagged_values(config)

        # Ensure nav exists
        if "nav" not in config:
            config["nav"] = []

        nav = config["nav"]

        # Ensure Home exists
        has_home = any(isinstance(item, dict) and "Home" in item for item in nav)

        if not has_home:
            nav.insert(0, {"Home": "index.md"})

        # Build API reference structure
        api_nav = build_nav_structure(modules)

        # Find and replace/add API Reference
        api_ref_index = None
        for i, item in enumerate(nav):
            if isinstance(item, dict) and "API Reference" in item:
                api_ref_index = i
                break

        if api_ref_index is not None:
            # Replace existing API Reference
            if not force:
                common.confirm("API Reference already exists in nav. Replace?", default=False)

            nav[api_ref_index] = {"API Reference": api_nav}
            common.info("Updated existing API Reference in nav")
        else:
            # Add new API Reference
            nav.append({"API Reference": api_nav})
            common.info("Added API Reference to nav")

        # Write updated config with custom dumper
        with MKDOCS_CONFIG.open("w", encoding="utf-8") as f:
            yaml.dump(
                config,
                f,
                Dumper=MkDocsConfigDumper,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=120,
                indent=2,
            )

        common.success(f"Updated: {common.format_path(MKDOCS_CONFIG.relative_to(PROJECT_ROOT))}")
        return True

    except Exception as e:
        common.error(f"Failed to update mkdocs.yml: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MkDocs API reference documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Generate docs for all modules
  %(prog)s --force            # Regenerate all, overwrite existing
  %(prog)s --no-nav-update    # Generate docs without updating mkdocs.yml
  %(prog)s --module entity    # Generate docs for specific module only
  %(prog)s --list-modules     # List all modules that will be documented
        """,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files without prompting",
    )

    parser.add_argument(
        "--no-nav-update",
        action="store_true",
        help="Skip updating mkdocs.yml navigation",
    )

    parser.add_argument(
        "--module",
        "-m",
        type=str,
        action="append",
        metavar="MODULE",
        help="Generate docs for specific module(s) only (can be used multiple times)",
    )

    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="List all modules that will be documented and exit",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    return parser.parse_args()


def list_modules() -> None:
    """List all modules that will be documented."""
    print()
    common.info("Modules to be documented:")
    print()

    for module_spec in MODULES_TO_DOCUMENT:
        is_init_only = module_spec.endswith(".__init__")
        marker = common.format_key(" (init only)" if is_init_only else "")
        print(f"  • inferflow.{module_spec}{marker}")

    print()
    print(f"Total:  {common.format_bold(str(len(MODULES_TO_DOCUMENT)))} modules")
    print()
    print(f"Source root: {common.format_path(SOURCE_ROOT)}")
    print(f"Docs output: {common.format_path(API_DOCS_DIR)}")
    print()


def main() -> None:
    args = parse_args()

    # Handle quiet mode
    if args.quiet:
        common.setup_quiet_mode()

    # Initialize ANSI formatter
    common.init_ansi_formatter()

    # List modules and exit
    if args.list_modules:
        list_modules()
        sys.exit(0)

    common.header("MkDocs API Documentation Generator")

    # Determine which modules to document
    modules_to_process = args.module if args.module else MODULES_TO_DOCUMENT

    common.info(f"Modules to document: {len(modules_to_process)}")
    print()

    # Discover modules
    common.step("Discovering modules...")
    discovered_modules: list[ModuleInfo] = []

    for module_spec in modules_to_process:
        try:
            module_info = discover_module(module_spec)
            if module_info:
                discovered_modules.append(module_info)
                submodule_count = len(module_info.submodules)
                sub_info = f" ({submodule_count} submodules)" if submodule_count > 0 else ""
                common.success(f"Discovered:  {module_spec}{sub_info}")
            else:
                common.warning(f"Could not discover:  {module_spec}")
        except Exception as e:
            common.error(f"Error discovering {module_spec}: {e}")

    if not discovered_modules:
        common.fatal_error("No modules discovered")

    print()
    common.success(f"Discovered {len(discovered_modules)} top-level modules")
    print()

    # Generate documentation
    common.step("Generating documentation files...")
    print()

    success_count = 0
    for module in discovered_modules:
        if write_module_docs(module, force=args.force):
            success_count += 1

    print()
    common.success(f"Generated documentation for {success_count}/{len(discovered_modules)} modules")
    print()

    # Update mkdocs.yml
    if not args.no_nav_update:
        common.step("Updating mkdocs.yml...")
        if update_mkdocs_config(discovered_modules, force=args.force):
            print()
        else:
            common.warning("Failed to update mkdocs.yml")
            print()

    # Summary
    common.info("Next steps:")
    print()
    print("  1. Review the generated documentation:")
    print(f"     {common.format_path(API_DOCS_DIR.relative_to(PROJECT_ROOT))}")
    print()
    print("  2. Preview the documentation:")
    print(f"     {common.format_key('mkdocs serve')}")
    print()
    print("  3. Build the documentation:")
    print(f"     {common.format_key('mkdocs build')}")
    print()

    sys.exit(0 if success_count == len(discovered_modules) else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        common.warning("Operation cancelled by user")
        print()
        sys.exit(130)
    except Exception as exc:
        print()
        common.error(f"Unexpected error: {exc}")
        traceback.print_exc()
        print()
        sys.exit(1)
