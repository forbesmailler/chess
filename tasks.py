import sys

from invoke import task

CPP_FILES = "engine/*.cpp engine/*.h"


@task
def format(c):
    """Format all code (Python + C++)."""
    format_py(c)
    format_cpp(c)


@task
def format_py(c):
    """Format Python code with ruff."""
    c.run("ruff format .")
    c.run("ruff check --fix --unsafe-fixes .")


@task
def format_cpp(c):
    """Format C++ code with clang-format."""
    c.run(f"clang-format -i {CPP_FILES}")


@task
def test(c):
    """Run all tests (Python + C++)."""
    test_py(c)
    test_cpp(c)


@task
def test_py(c):
    """Run pytest."""
    c.run("pytest")


@task
def test_cpp(c):
    """Run C++ engine tests."""
    if sys.platform == "win32":
        c.run("engine\\build\\Release\\lichess_bot.exe")
    else:
        c.run("engine/build/lichess_bot")


@task
def build_cpp(c):
    """Build C++ engine."""
    with c.cd("engine/build"):
        c.run("cmake --build . --config Release")
