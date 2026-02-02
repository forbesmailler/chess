from invoke import task


@task
def format(c):
    """Format Python code with ruff."""
    c.run("ruff format .")
    c.run("ruff check --fix .")


@task
def test(c):
    """Run pytest."""
    c.run("pytest")
