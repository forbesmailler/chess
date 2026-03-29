collect_ignore_glob = []

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    collect_ignore_glob.append("test_export_nnue.py")
    collect_ignore_glob.append("test_train_nnue.py")
