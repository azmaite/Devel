"""
Helpers for tkinter dialogs shared by the pipelines.

sort_dialog_descending: make a tkinter file dialog list folders and files in
    descending order, so date-named (`YYYYMMDD_...`) folders show newest first.
"""

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
