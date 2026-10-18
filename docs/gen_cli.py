"""Generate the CLI reference page from the argparse parsers at docs build time."""

import argparse
import os

import mkdocs_gen_files

os.environ["COLUMNS"] = "100"  # stable help wrapping, independent of the terminal

from lunar_toolkit.cli import _build_parser  # noqa: E402

parser = _build_parser()
sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
summaries = {a.dest: a.help or "" for a in sub._choices_actions}

lines = [
    "# CLI reference",
    "",
    "Generated from `lunar <command> --help` when the docs are built.",
    "",
    "| Command | Purpose |",
    "|---|---|",
    *(f"| [`{name}`](#lunar-{name}) | {summaries.get(name, '')} |" for name in sub.choices),
    "",
    "## `lunar`",
    "",
    "```text",
    parser.format_help().rstrip(),
    "```",
]
for name, cmd in sub.choices.items():
    lines += ["", f"## `lunar {name}`", "", "```text", cmd.format_help().rstrip(), "```"]

with mkdocs_gen_files.open("cli.md", "w") as f:
    f.write("\n".join(lines) + "\n")
