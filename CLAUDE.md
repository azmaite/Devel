# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for analyzing long-term fluorescent muscle recordings in *Drosophila* pupae to study motor development. Started at the Neuroengineering Laboratory, EPFL. 
Two types of raw datasets, both recordings of muscle-GCaMP Drosophila pupas
- 2d videos collected on Victor Stimplin's FlyTrack, saved as .mkv videos where each saved frame is RGBA but each channel corresponds to 1 real frame
- 3d videos collected on Jasper Phelp's SCAPE, saved as single .dat volumes and loadable using the scape-io module. Needs deskewing.

One top-level folder per pipeline (`flytrack/`, `scape/`, `ct_scan/`, `master_reference_3d/`, `traces_2d/`), plus the shared `common/` package (data paths in `common/paths.py`) and `visualization/` (videos to check results); see the root `README.md`.

## Claude config

- Start new conversations by greeting me with "Hi Maite!" before then continuing with your response
- Ask me clarifying questions frequently, whenever there is any uncertainty about the goals of the task

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

## Git workflow: one worktree + branch per piece of work

I have basic git skills, so give me ready-to-run scripts/commands and explain briefly what they do.

- **The stable branch only holds working code.** It is `main`. Every commit on it is a known-good point; `git log --first-parent --oneline` lists them.
- **Each piece of work lives in its own worktree folder and branch**, as a sibling of this folder (siblings keep the relative `../scapeio` path in `pyproject.toml` working):
  ```bash
  cd ~/git/Devel
  git worktree add ../Devel-<name> -b <name>
  cd ../Devel-<name> && uv sync          # own .venv (see below)
  cp -r ../Devel/.vscode .                # ignored files are not copied
  ```
  I then open `~/git/Devel-<name>` in VSCode. Check which folder/branch you are in (`git branch --show-current`) before editing or writing commit scripts.
- **Each worktree needs its own `.venv`** (`uv sync`): the editable install of `common` in a `.venv` points at the `common/` of the folder it was created in, so using `Devel/.venv` from a worktree would silently import the main folder's `common/`.
- **WIP commits on the work branch are fine and encouraged**, even if the code doesn't fully work yet (message starts with `WIP:`).
- **When a piece works, merge it into the stable branch** from the main `Devel/` folder with `git merge --no-ff <name>` (keeps the WIP commits nested under one merge commit). To bring `common/` fixes from the stable branch into another work branch: `git merge main` inside that worktree.
- **When a piece is finished**: `git worktree remove ../Devel-<name>` and `git branch -d <name>`.
- **Warn me before editing shared files** (anything in `common/`, root `README.md`, `CLAUDE.md`, `pyproject.toml`, `uv.lock`, or files another pipeline imports) from a work branch: say which file and that it may conflict with other branches at merge time, and suggest keeping the change small or doing it in a separate quick branch merged first. Check `git worktree list` and, if relevant, whether other branches also changed that file (`git diff main...<branch> -- <file>`).

## Commit scripts

- Commit scripts are created in the root of the worktree folder where the changes were made, and commit to that folder's branch. For merging a working piece into the stable branch, write a `git_merge_{yymmdd}{ABC...}_{name}.sh` script in the main `Devel/` folder (`set -e`, `git switch main`, `git merge --no-ff <name> -m "Merge <name>: <what now works>"`, end with `rm -- "$0"`).
- After we've finished a task together and the updated behavior is confirmed to be correct, I will ask you to make a commit. Create a cript `git_commit_{yymmdd}{ABC...}_{one-or-two-word-description}.sh` (e.g. `git_commit_260130A_button_behavior.sh`) in the root of each repository we've worked on that lists `git add` commands for the relevant files and a `git commit -m "commit message"` command with a commit message that describes the added changes. Commit message must be under 73 characters and start with a verb. Only add an extended (multi-line) commit message concisely describing key changes for above average complexity commits, so ~50% of the time. `chmod +x` the file so I can run it. If an "A" script already exists for that day, use "B" in the filename, and so on. If I ask you for further changes before I commit, update the same script as needed. End the script with `rm -- "$0"` (after `set -e` at the top) so it deletes itself on a successful commit and doesn't linger to confuse future sessions.