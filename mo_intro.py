import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium", app_title="Intro to Marimo notebooks")


@app.cell
def _(mo):
    mo.md(r"""# Marimo notebooks""")
    return


@app.cell
def _(mo):
    mo.md(r"""## tab completion and type 'aware'""")
    return


app._unparsable_cell(
    r"""
    a = \"foo\"

    def foo(a: str):
        a.
        retrun a

    """,
    name="_"
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
