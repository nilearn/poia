# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "pandas==2.2.3",
#     "plotly==6.0.1",
# ]
# ///

import marimo

__generated_with = "0.12.8"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""Aims to list all functions of nilearn that are not in its user facing public API, but""")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import nilearn
    import importlib
    import inspect
    return importlib, inspect, mo, nilearn, pd


@app.cell
def _(nilearn):
    nilearn.__version__
    return


@app.cell
def _(importlib, inspect, mo, nilearn):
    public_api = ["nilearn"]
    for subpackage in nilearn.__all__:
        public_api.append(subpackage)
        if subpackage.startswith("_"):
            continue
        mod = importlib.import_module(f"nilearn.{subpackage}")
        public_api.extend(mod.__all__)
        for x in mod.__all__:
            if inspect.ismodule(mod.__dict__[x]):
                submod = importlib.import_module(f"nilearn.{subpackage}.{x}")
                if hasattr(submod, '__all__'):
                    public_api.extend(submod.__all__)
    mo.md("List all modules, classes, functions that are part of nilearn API.")
    return mod, public_api, submod, subpackage, x


@app.cell
def _(mo, pd):
    df = pd.read_csv(mo.notebook_location() / 'public'/  'nilearn' / "functions_used.csv")
    return (df,)


@app.cell
def _(df, public_api):
    mask = ~df["object"].isin(public_api)
    return (mask,)


@app.cell
def _(df, mask):
    df[mask]
    return


@app.cell
def _():
    from poia import plot_usage
    return (plot_usage,)


@app.cell
def _(plot_usage):
    _, defs = plot_usage.run()
    return (defs,)


@app.cell
def _(df, mask):
    df_counts = df[mask]['object'].value_counts().reset_index()
    df_counts.columns = ['object', 'count']
    df_counts
    return (df_counts,)


@app.cell
def _(df, mask):
    df_weighted = df[mask].groupby('object', as_index=False)['n'].sum().sort_values('n', ascending=False)
    df_weighted.columns = ['object', 'weighted_count']
    df_weighted
    return (df_weighted,)


@app.cell
def _(defs, df, mask):
    fig = defs["plot_usage"](df[mask], color="extracted_version")
    fig.show()
    return (fig,)


if __name__ == "__main__":
    app.run()
