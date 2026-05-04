#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd -P)"

# symlink files in this directory into $HOME/.claude where
# Claude Code will actually see and use them
mkdir -p "$HOME/.claude" > /dev/null
ln -sv "$SCRIPT_DIR/CLAUDE.md" "$HOME/.claude/CLAUDE.md"
ln -sv "$SCRIPT_DIR/settings.json" "$HOME/.claude/settings.json"

# Claude tries to update settings.json too much (e.g. user changes the model or
# effort within a session -> claude tries to make that the new default for future
# sessions by changing settings.json), so make it read-only to prevent that.
# Actually I've decided to allow it for now so the next line is commented out.
#chmod -w "$SCRIPT_DIR" "$SCRIPT_DIR/settings.json"