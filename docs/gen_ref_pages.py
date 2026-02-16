"""Generate API reference pages automatically from source code.

This script is run by the mkdocs-gen-files plugin during documentation build.
It discovers all Python modules in the forward_model package and creates
corresponding documentation pages with mkdocstrings directives.
"""

from pathlib import Path

import mkdocs_gen_files

# Package to document
PACKAGE_NAME = "forward_model"
PACKAGE_PATH = Path("forward_model")

# Create navigation file
nav = mkdocs_gen_files.Nav()

# Iterate through all Python files in the package
for path in sorted(PACKAGE_PATH.rglob("*.py")):
    # Get module path relative to package root
    module_path = path.relative_to(PACKAGE_PATH.parent).with_suffix("")

    # Convert path to module name (e.g., forward_model/io/loaders.py -> forward_model.io.loaders)
    doc_path = path.relative_to(PACKAGE_PATH.parent).with_suffix(".md")
    full_doc_path = Path("api-reference", doc_path)

    # Get parts for navigation
    parts = tuple(module_path.parts)

    # Skip __main__ and __pycache__
    if parts[-1] == "__main__" or "__pycache__" in parts:
        continue

    # For __init__.py files, use the parent directory name
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    # Skip if no parts (shouldn't happen, but be safe)
    if not parts:
        continue

    # Add to navigation
    nav[parts] = doc_path.as_posix()

    # Create the documentation file
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Convert parts to module identifier
        module_identifier = ".".join(parts)

        # Write page header
        if parts[-1] == parts[0]:  # Top-level package
            fd.write(f"# {parts[-1].replace('_', ' ').title()} Package\n\n")
        else:
            fd.write(f"# {parts[-1].replace('_', ' ').title()}\n\n")

        # Write mkdocstrings directive to generate API docs
        fd.write(f"::: {module_identifier}\n")

    # Set edit path for this file
    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Write the navigation file
with mkdocs_gen_files.open("api-reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
