from __future__ import annotations

import argparse
from pathlib import Path

from agent.config import ObsidianVaultPathError, get_settings, validate_obsidian_vault_path


ENV_PATH = Path(".env")


def main() -> int:
    parser = argparse.ArgumentParser(description="Atlas setup utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "configure-obsidian",
        help="Interactively save OBSIDIAN_VAULT_PATH to .env",
    )
    subparsers.add_parser(
        "test-obsidian",
        help="Validate the configured Obsidian vault path before indexing",
    )

    args = parser.parse_args()
    if args.command == "configure-obsidian":
        return configure_obsidian()
    if args.command == "test-obsidian":
        return test_obsidian()

    parser.error(f"Unknown command: {args.command}")
    return 2


def configure_obsidian() -> int:
    print("Paste the full path to your Obsidian vault folder.")
    print("Tip: on macOS, right-click the vault folder in Finder, hold Option, and choose 'Copy ... as Pathname'.")
    raw_path = input("Obsidian vault path: ").strip().strip('"').strip("'")

    try:
        vault_path = validate_obsidian_vault_path(Path(raw_path).expanduser() if raw_path else None)
    except ObsidianVaultPathError as exc:
        print(f"Error: {exc}")
        return 1

    _set_env_value(ENV_PATH, "OBSIDIAN_VAULT_PATH", str(vault_path))
    print(f"Saved OBSIDIAN_VAULT_PATH to {ENV_PATH}")
    return 0


def test_obsidian() -> int:
    settings = get_settings()
    try:
        vault_path = validate_obsidian_vault_path(settings.obsidian_vault_path)
    except ObsidianVaultPathError as exc:
        print(f"Error: {exc}")
        return 1

    markdown_count = sum(1 for path in vault_path.rglob("*.md") if not _is_hidden_path(path, vault_path))
    print(f"Obsidian vault path is valid: {vault_path}")
    print(f"Markdown files found: {markdown_count}")
    return 0


def _set_env_value(path: Path, key: str, value: str) -> None:
    line = f'{key}="{_escape_env_value(value)}"'
    if not path.exists():
        path.write_text(f"{line}\n", encoding="utf-8")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    updated = False
    next_lines: list[str] = []

    for existing_line in lines:
        stripped = existing_line.strip()
        if stripped.startswith(f"{key}="):
            next_lines.append(line)
            updated = True
        else:
            next_lines.append(existing_line)

    if not updated:
        next_lines.append(line)

    path.write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def _escape_env_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _is_hidden_path(path: Path, vault_path: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(vault_path).parts)


if __name__ == "__main__":
    raise SystemExit(main())
