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
    c.run("pytest", warn=True)
    with c.cd("engine/build"):
        c.run("ctest -C Release --output-on-failure")


@task
def test_py(c):
    """Run pytest."""
    c.run("pytest")


@task
def test_cpp(c):
    """Run C++ unit tests with ctest."""
    with c.cd("engine/build"):
        c.run("ctest -C Release --output-on-failure")


@task
def build_cpp(c):
    """Build C++ engine."""
    with c.cd("engine/build"):
        c.run("cmake --build . --config Release")
