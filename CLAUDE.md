# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for analyzing long-term fluorescent muscle recordings in *Drosophila* pupae to study motor development. Started at the Neuroengineering Laboratory, EPFL. Raw input is `.mkv` video files (16-bit fluorescence at 40 Hz, where each file frame contains 4 packed real frames) and `.h5` timestamp files; outputs are HDF5 component files, MP4 summaries, and PNG visualizations.

## Claude config

- Start new conversations by greeting me with "Hi Maite!" before then continuing with your response
- Ask me clarifying questions frequently, whenever there is any uncertainty about the goals of the task
- Start all python files with `#!/usr/bin/env python3` even if the file is not intended to be executed as a script

- Format python functions, docstrings, and strings as follows:
```python
def function(arg: Literal['x', 'y'],
             next_arg: Union[int, str]) -> None:
    """
    We generally follow the numpy style guide, except:

    Docstrings start on the line following the triple double quotes, not
    continuing on the same line.

    When a docstring refers to a variable like `arg` or a function like
    `module.other_function()`, use single backticks (not double backticks).

    Parameters
    ----------
    arg : 'x' or 'y'
        Description of the argument. Note the single quotes for literal values.

    Returns
    -------
    None
    """
    print('In code, use single quotes for all strings, like this.')
    hello_arg = 'Hello ' + arg
    print("But if there's an apostrophe in the string, use double quotes to avoid escaping")
    return
```

- After we've finished a task together and the updated behavior is confirmed to be correct (either I tell you it is or you verify it yourself), create a script `git_commit_{yymmdd}{ABC...}_{one-or-two-word-description}.sh` (e.g. `git_commit_260130A_button_behavior.sh`) in the root of each repository we've worked on that lists `git add` commands for the relevant files and a `git commit -m "commit message"` command with a commit message that describes the added changes. Commit message must be under 73 characters and start with a verb. Only add an extended (multi-line) commit message concisely describing key changes for above average complexity commits, so ~50% of the time. `chmod +x` the file so I can run it. If an "A" script already exists for that day, use "B" in the filename, and so on. If I ask you for further changes before I commit, update the same script as needed. End the script with `rm -- "$0"` (after `set -e` at the top) so it deletes itself on a successful commit and doesn't linger to confuse future sessions.