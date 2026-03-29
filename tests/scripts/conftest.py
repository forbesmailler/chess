collect_ignore_glob = []

try:
    import chess  # noqa: F401
except ModuleNotFoundError:
    collect_ignore_glob.append("test_build_opening_book.py")
