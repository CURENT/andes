#!/usr/bin/env python3
"""
Generate requirements.txt from pyproject.toml (single source of truth).

This script is provided for users who need requirements.txt for legacy tools.
The authoritative source of dependencies is pyproject.toml.

Usage:
    python scripts/generate_requirements.py
    python scripts/generate_requirements.py --extra dev
    python scripts/generate_requirements.py --extra all
"""

import argparse
import sys
from pathlib import Path

try:
    import tomli as tomllib
except ImportError:
    try:
        import tomllib
    except ImportError:
        print("Error: tomllib/tomli not available", file=sys.stderr)
        print("Install with: pip install tomli", file=sys.stderr)
        sys.exit(1)


def load_pyproject():
    """Load and parse pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(pyproject_path, "rb") as f:
        return tomllib.load(f)


def format_dependencies(deps):
    """Format dependencies for requirements.txt."""
    return sorted(deps)


def main():
    parser = argparse.ArgumentParser(
        description="Generate requirements.txt from pyproject.toml"
    )
    parser.add_argument(
        "--extra",
        choices=["dev", "doc", "interop", "web", "all"],
        help="Include optional dependencies",
    )
    parser.add_argument(
        "--output",
        default="requirements.txt",
        help="Output file (default: requirements.txt)",
    )
    args = parser.parse_args()

    # Load pyproject.toml
    data = load_pyproject()
    project = data.get("project", {})

    # Get core dependencies
    dependencies = project.get("dependencies", [])

    # Add optional dependencies if requested
    if args.extra:
        optional_deps = project.get("optional-dependencies", {})
        if args.extra in optional_deps:
            dependencies.extend(optional_deps[args.extra])
        else:
            print(f"Warning: Optional dependency group '{args.extra}' not found", file=sys.stderr)

    # Write to file
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        f.write("# Generated from pyproject.toml - DO NOT EDIT MANUALLY\n")
        f.write("# To update: python scripts/generate_requirements.py\n")
        f.write("# Single source of truth: pyproject.toml\n\n")

        for dep in format_dependencies(dependencies):
            f.write(f"{dep}\n")

    print(f"✓ Generated {output_path} with {len(dependencies)} dependencies")
    if args.extra:
        print(f"  Including optional group: {args.extra}")


if __name__ == "__main__":
    main()
