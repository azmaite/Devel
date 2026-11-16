"""
Helpers for tkinter dialogs shared by the pipelines.

sort_dialog_descending: make a tkinter file dialog list folders and files in
    descending order, so date-named (`YYYYMMDD_...`) folders show newest first.
ask_directory: open a folder-selection dialog listing the newest folders first.
"""

import os
from typing import Any


def sort_dialog_descending(tk_root: Any) -> None:
    """
    Make Tk file dialogs list folders and files in descending (Z-A) order.

    Our recording folders start with a date (``YYYYMMDD_...``), so this shows
    the newest ones first. Tk's built-in Linux dialog (used by
    `tkinter.filedialog.askdirectory()` and `askopenfilename()`) has no sort
    option, so this wraps its internal listing proc
    ``::tk::dialog::file::GlobFiltered`` to reverse the list it returns. Call it
    once on the Tk root, before opening the dialog; calling it again does nothing.

    Parameters
    ----------
    tk_root : tkinter.Tk
        The Tk root window the dialog will be opened from.

    Returns
    -------
    None
    """
    tk_root.tk.eval('''
        if {[info commands ::tk::dialog::file::GlobFilteredAscending] eq ""} {
            auto_load ::tk::dialog::file::
            rename ::tk::dialog::file::GlobFiltered ::tk::dialog::file::GlobFilteredAscending
            proc ::tk::dialog::file::GlobFiltered {args} {
                lreverse [uplevel 1 [list ::tk::dialog::file::GlobFilteredAscending {*}$args]]
            }
        }
    ''')


def ask_directory(initialdir: str | os.PathLike,
                  title: str) -> str:
    """
    Open a folder-selection dialog starting in `initialdir`, with the newest
    (``YYYYMMDD_...``) folders first (see `sort_dialog_descending()`).

    Some native (e.g. GTK/KDE portal) directory choosers ignore `initialdir` and
    start in the current working directory instead, so the working directory is
    changed to `initialdir` while the dialog is open.

    Parameters
    ----------
    initialdir : str or os.PathLike
        Folder the dialog starts in.
    title : str
        Title of the dialog.

    Returns
    -------
    str
        The selected folder, or '' if cancelled.
    """
    import tkinter as tk
    from tkinter import filedialog

    original_cwd = os.getcwd()
    if os.path.isdir(initialdir):
        os.chdir(initialdir)
    try:
        root = tk.Tk()
        root.withdraw()
        sort_dialog_descending(root)
        folder = filedialog.askdirectory(initialdir=str(initialdir), title=title)
        root.destroy()
    finally:
        os.chdir(original_cwd)
    return folder or ''
