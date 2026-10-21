"""Console diagnostics must not turn a successful operation into a failure."""
import builtins
import sys


def safe_print(*values, **kwargs):
    try:
        builtins.print(*values, **kwargs)
    except UnicodeEncodeError:
        stream = kwargs.get("file") or sys.stdout
        encoding = getattr(stream, "encoding", None) or "ascii"
        escaped = [str(value).encode(encoding, errors="backslashreplace").decode(encoding)
                   for value in values]
        builtins.print(*escaped, **kwargs)
