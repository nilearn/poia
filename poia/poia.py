# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "beautifulsoup4==4.13.3",
#     "ipython==9.1.0",
#     "marimo",
#     "matplotlib==3.10.1",
#     "matplotlib-venn==1.1.2",
#     "nbconvert==7.16.6",
#     "nbformat==5.10.4",
#     "packaging==24.2",
#     "pandas==2.2.3",
#     "plotly==6.0.1",
#     "requests==2.32.3",
#     "rich==14.0.0",
#     "toml==0.10.2",
# ]
# ///
"""Package Of Interest Auditor."""

import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium", app_title="POIA")


@app.cell(hide_code=True)
def _(config, mo):
    mo.md(
        f"""
        # POIA: Package Of Interest Audit

        Audit usage of a package of interest (POI)
        (in this case '{config["PACKAGE_OF_INTEREST"]}') on public repos on github.

        Will list repo that:

        - are listed as dependents on the GitHub ∈sightsinsights tab of a repo

        - if that is not possible it will ping the GitHub API to find repos that:

            - contain {config["PACKAGE_OF_INTEREST"]} in one the common files
              used to declare dependencies
              (pyproject.toml, setup.cfg, requirements.txt...)

            - import {config["PACKAGE_OF_INTEREST"]} in a python module or a ipython notebook.

        Then it will clone those repos, collect and plot information about them.
        """
    )
    return


@app.cell(hide_code=True)
def _(config, mo):
    mo.md(
        f"""
        ## Search repos for {config["PACKAGE_OF_INTEREST"]} on github

        If you now the github package where {config["PACKAGE_OF_INTEREST"]} is stored,
        we'll try to scrap the github UI ∈sightsinsights tab of that repo
        to get a list of dependentsdependents for {config["PACKAGE_OF_INTEREST"]}.
        Results are saved in a dependents.jsondependents.json file.

        If the repo of {config["PACKAGE_OF_INTEREST"]} is unknown
        or no dependents.jsondependents.json is found or mentioned in the configuration,
        then POIA will start pinging the GitHub API to search
        where this {config["PACKAGE_OF_INTEREST"]} may be mentioned.
        """
    )
    return


@app.cell
def _(
    QUERIES,
    config,
    get_dependents,
    json,
    load_cache,
    logger,
    search_repositories,
    update_cache,
):
    dependents = None

    dependents_file = (
        config["OUTPUT"]["DIR"] / config["PACKAGE_OF_INTEREST"] / config["OUTPUT"]["DEPENDENTS"]
    )

    repos = []

    if dependents_file.exists() or config["PACKAGE_OF_INTEREST_REPO"]:
        if dependents_file.exists():
            logger.info(f"Loading dependents file: {dependents_file}")
            with dependents_file.open("r") as f:
                dependents = json.load(f)
        elif config["PACKAGE_OF_INTEREST_REPO"]:
            logger.info(f"Dependents file not found: {dependents_file}")
            logger.info("Scrapping github for dependents...")
            dependents = get_dependents("nilearn/nilearn")
            update_cache(dependents_file, dependents)

        repos = [f"https://github.com/{x}" for x in dependents]

        repo_url_cache_file = (
            config["OUTPUT"]["DIR"]
            / config["PACKAGE_OF_INTEREST"]
            / config["OUTPUT"]["REPOSITORIES"]
        )
        if repo_url_cache_file.exists():
            logger.info("Adding repos grabbed from github API...")
            repos.extend(load_cache(repo_url_cache_file))
        repos = list(set(repos))

    else:
        logger.info(f"'dependents' file not found: {dependents_file}")
        logger.info(f"No known repo for: {config['PACKAGE_OF_INTEREST']}")
        logger.info("Pipping github API to list repos...")
        repos = search_repositories(
            queries=QUERIES,
            config=config,
            cache_file=config["OUTPUT"]["REPOSITORIES"],
        )
    return dependents, dependents_file, f, repo_url_cache_file, repos


@app.cell
def _(Path, config, mo, repos, shutil):
    cloned_repos = set()
    for r in set(repos):
        url_path = Path(r)

        repo_name = url_path.name
        user_name = url_path.parents[0].name

        if (config["CACHE"]["DIR"] / user_name / repo_name).exists():
            if (config["CACHE"]["DIR"] / user_name / repo_name / ".git").exists():
                cloned_repos.add(r)
            else:
                shutil.rmtree(r)

        if (config["CACHE"]["DIR"] / user_name).exists() and not list(
            (config["CACHE"]["DIR"] / user_name).iterdir()
        ):
            shutil.rmtree(config["CACHE"]["DIR"] / user_name)

    mo.md(
        f"""
        {len(set(repos))} repositories found.

        {len(cloned_repos)} repositories already cloned.
        """
    )
    return cloned_repos, r, repo_name, url_path, user_name


@app.cell
def _(repos):
    sorted(repos)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Cloning repos""")
    return


@app.cell(disabled=True)
def clone_repositories(clone_repo, config, load_cache, logger, repos):
    from concurrent.futures import ThreadPoolExecutor

    ignore_list = load_cache(
        config["OUTPUT"]["DIR"] / f"{config['PACKAGE_OF_INTEREST']} / {config['OUTPUT']['IGNORE']}"
    )
    with ThreadPoolExecutor(max_workers=config["N_JOBS"]) as executor:
        executor.map(clone_repo, [x for x in repos if x not in ignore_list])
    logger.info("Cloning done.")
    return ThreadPoolExecutor, executor, ignore_list


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Extracting data from repos

        Extract useful data from each repository and save them as a JSON to disk.
        """
    )
    return


@app.cell
def _(config, extract_data):
    extract_data(config)
    return


@app.cell
def _(
    config,
    get_extracted_version,
    get_lock_file,
    literal_eval,
    load_cache,
    logger,
    mo,
    pd,
):
    content_cache_file = (
        config["OUTPUT"]["DIR"] / config["PACKAGE_OF_INTEREST"] / config["OUTPUT"]["CONTENT"]
    )

    data_cache_file = mo.notebook_location() / config["OUTPUT"]["DATA"]

    try:
        data_poi = pd.read_csv(
            data_cache_file,
            na_values="n/a",
            converters={
                "versions": literal_eval,
            },
            parse_dates=["last_commit"],
        )

        logger.info(f"File {data_cache_file} loaded.")

    except Exception as e:
        logger.error(repr(e))
        logger.error(f"Could not load {data_cache_file}")
        logger.error(f"Trying to create dataframe from {content_cache_file}")

        data_projects = load_cache(content_cache_file)

        data_poi = pd.DataFrame(data_projects)

        data_poi["last_commit"] = pd.to_datetime(data_poi["last_commit"])

        data_poi["extracted_version"] = data_poi["versions"].apply(get_extracted_version)
        data_poi["lockfile"] = data_poi["versions"].apply(get_lock_file)

        data_poi["has_version"] = data_poi["extracted_version"].astype("bool", errors="ignore")
        data_poi["has_imports"] = data_poi["import_counts"].astype("bool", errors="ignore")
        data_poi["use_imports"] = data_poi["function_counts"].astype("bool", errors="ignore")
        data_poi["include"] = (
            data_poi["has_version"] | data_poi["has_imports"] | data_poi["use_imports"]
        )

        data_poi[["user", "repo"]] = data_poi["name"].str.split("/", expand=True)

        data_poi["content"] = "mixed"
        pure_notebook = (data_poi["n_notebook_parsed"] > 0) & (
            data_poi["n_python_file_parsed"] == 0
        )
        pure_python = (data_poi["n_notebook_parsed"] == 0) & (data_poi["n_python_file_parsed"] > 0)
        data_poi.loc[pure_notebook, "content"] = "notebook"
        data_poi.loc[pure_python, "content"] = "python"

        data_poi["p_notebook_parsed"] = data_poi["n_notebook_parsed"] / (data_poi["n_notebook"])
        data_poi["p_python_file_parsed"] = (
            data_poi["n_python_file_parsed"] / (data_poi["n_python_file"])
        )

        data_poi["duplicated_repo"] = data_poi["repo"].duplicated()

        data_poi.fillna("n/a").to_csv(data_cache_file, index=False)

    mo.md(
        f"""
        Load data from JSON or CSV.

        If loading from JSON, do some data cleaning (see below) before saving to CSV.

        - for now drop the repos that apparently have several pinned versions

        - mark as repos to include those
          that either declare {config["PACKAGE_OF_INTEREST"]}
          as dependency, import or use it.

        - flag repos that seem to be duplicated:
          githud code search should not return forks,
          but it is possible that the provenance info was lost
          in some cases.

        - check which repos use {config["PACKAGE_OF_INTEREST"]}
          only python files, jupyter notebook or both.
        """
    )
    return (
        content_cache_file,
        data_cache_file,
        data_poi,
        data_projects,
        pure_notebook,
        pure_python,
    )


@app.cell(hide_code=True)
def _(data_poi, mo):
    mo.md(
        f"""
        Found {len(data_poi)}
        repositories. {data_poi["duplicated_repo"].sum()}
        of them seem to be duplicated.
        """
    )
    return


@app.cell
def _(data_poi):
    data_poi[data_poi["duplicated_repo"]][["user", "repo"]]
    return


@app.cell
def _(data_poi):
    data_poi
    return


@app.cell
def _(data_poi, mo):
    transformed_df = mo.ui.dataframe(data_poi)
    transformed_df
    return (transformed_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Plotting results""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Content""")
    return


@app.cell(hide_code=True)
def _(config, data_poi, mo, px):
    fig_content = px.histogram(
        data_poi[data_poi["include"]],
        x="content",
        title=f"Content of {data_poi['include'].sum()} repositories",
    )

    fig_content.show()
    mo.md(f"""
    Content of repositories distinguishing those
    using {config["PACKAGE_OF_INTEREST"]}
    in notebooks, python files or both.
    """)
    return (fig_content,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### File content""")
    return


@app.cell(hide_code=True)
def _(config, data_poi, mo, plt):
    from matplotlib_venn import venn3

    as_dependency = data_poi["name"][data_poi["has_version"]].to_list()
    actually_import = data_poi["name"][data_poi["has_imports"]].to_list()
    actually_mentions = data_poi["name"][
        (data_poi["n_notebook"] > 0) | (data_poi["n_python_file"] > 0)
    ].to_list()
    venn3(
        subsets=(
            set(as_dependency),
            set(actually_mentions),
            set(actually_import),
        ),
        set_labels=(
            f"{config['PACKAGE_OF_INTEREST']} as dependency",
            f"mentions {config['PACKAGE_OF_INTEREST']}",
            f"imports {config['PACKAGE_OF_INTEREST']}",
        ),
    )
    plt.show()
    mo.md(
        f"""
        Let's try to see how many repositories have
        the {config["PACKAGE_OF_INTEREST"]} as dependency but do not import it.

        Some of those numbers are explained by files
        where {config["PACKAGE_OF_INTEREST"]} is mentioned
        but could not be properly parsed.
        """
    )
    return actually_import, actually_mentions, as_dependency, venn3


@app.cell(hide_code=True)
def _(
    data_poi,
    mo,
    px,
    radio_include,
    radio_include_element,
    radio_x,
    radio_x_element,
    radio_y,
    radio_y_element,
):
    def interactive_plot(df, include_mask=None, x=radio_x.value, y=radio_y.value):
        if include_mask is not None:
            df = df[include_mask]
        plot = mo.ui.plotly(
            px.scatter(
                df,
                x=x,
                y=y,
                width=900,
                height=600,
                hover_name="name",
                hover_data="name",
                marginal_x="histogram",
                marginal_y="histogram",
                color="content",
            )
        )
        return plot

    plot = interactive_plot(data_poi, include_mask=radio_include.value)

    mo.vstack(
        [
            mo.md("File content"),
            mo.hstack(
                [radio_include_element, radio_x_element, radio_y_element, mo.vstack([])],
                align="center",
            ),
            plot,
        ]
    )
    return interactive_plot, plot


@app.cell
def _(plot):
    plot.value
    return


@app.cell
def _(data_poi):
    never_imported = data_poi[(data_poi["has_version"]) & (~data_poi["use_imports"])]
    never_imported
    return (never_imported,)


@app.cell
def _(data_poi, print):
    for row in data_poi.iterrows():
        if row[1]["versions"]:
            print(row[1]["versions"])
    return (row,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Lockfile used""")
    return


@app.cell(hide_code=True)
def _(config, data_poi, mo, px, radio_include, radio_include_element):
    def plot_lockfiles(df, include_mask=None):
        if include_mask is not None:
            df = df[include_mask]
        df = df[~(df["extracted_version"].eq("several_versions_detected"))]
        df = df.drop_duplicates(subset=["name"])

        fig = px.histogram(
            df,
            x="lockfile",
            title=(
                f"lockfile used to declare {config['PACKAGE_OF_INTEREST']} "
                f"as dependency - {len(df)} repository included"
            ),
        )
        return fig

    fig_lockfile = plot_lockfiles(data_poi, include_mask=radio_include.value)
    fig_lockfile.show()
    mo.vstack(
        [
            mo.md("Lockfile used"),
            mo.hstack(
                [radio_include_element, mo.vstack([])],
                align="center",
            ),
        ]
    )
    return fig_lockfile, plot_lockfiles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Last commit date""")
    return


@app.cell(hide_code=True)
def _(
    config,
    data_poi,
    explanation_version_0,
    mo,
    plot_repos,
    radio_color,
    radio_color_element,
    radio_include,
    radio_include_element,
    radio_timebin,
    radio_timebin_element,
):
    fig_repo = plot_repos(
        data_poi,
        color=radio_color.value,
        bin_size=radio_timebin.value,
        include_mask=radio_include.value,
    )
    fig_repo.show()

    mo.vstack(
        [
            mo.md(f"""
    Let's see how recently those repos were last updated
    split by what {radio_color.value} uses {config["PACKAGE_OF_INTEREST"]}.
    {explanation_version_0}"""),
            mo.hstack(
                [radio_color_element, radio_timebin_element, radio_include_element, mo.vstack([])],
                align="center",
            ),
        ]
    )
    return (fig_repo,)


@app.cell(hide_code=True)
def version_used(data_poi, mo, plot_versions):
    fig_version = plot_versions(data_poi)
    fig_version.show()

    mo.md("""
    ### Version used
    """)
    return (fig_version,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Subpackage use""")
    return


@app.cell
def _(data_poi, extract_object_count):
    import_df = extract_object_count(data_poi[data_poi["has_imports"]], col="import_counts")
    import_df
    return (import_df,)


@app.cell
def _(import_df):
    import_df_counts = import_df["object"].value_counts().reset_index()
    import_df_counts.columns = ["object", "count"]
    import_df_counts
    return (import_df_counts,)


@app.cell
def _(import_df):
    import_df_weighted = (
        import_df.groupby("object", as_index=False)["n"].sum().sort_values("n", ascending=False)
    )
    import_df_weighted.columns = ["object", "weighted_count"]
    import_df_weighted
    return (import_df_weighted,)


@app.cell(hide_code=True)
def _(
    config,
    explanation_version_0,
    import_df,
    mo,
    plot_usage,
    radio_color,
    radio_color_element,
    switch,
    switch_weigthed_element,
):
    subpackage_fig = plot_usage(import_df, color=radio_color.value, weighted=switch.value)
    subpackage_fig.show()

    mo.vstack(
        [
            mo.md(f"""
    Analysis of which subpackage of {config["PACKAGE_OF_INTEREST"]}
    are used split by {radio_color.value}.
    {explanation_version_0}
    """),
            mo.hstack(
                [radio_color_element, switch_weigthed_element, mo.vstack([])],
                align="center",
            ),
        ],
    )
    return (subpackage_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Function / Class use""")
    return


@app.cell
def _(config, data_poi, extract_object_count):
    function_df = extract_object_count(data_poi[data_poi["use_imports"]], col="function_counts")
    function_df.to_csv(
        config["OUTPUT"]["DIR"] / config["PACKAGE_OF_INTEREST"] / "functions_used.csv", index=False
    )
    function_df
    return (function_df,)


@app.cell
def _(function_df):
    function_df_counts = function_df["object"].value_counts().reset_index()
    function_df_counts.columns = ["object", "count"]
    function_df_counts
    return (function_df_counts,)


@app.cell
def _(function_df):
    function_df_weighted = (
        function_df.groupby("object", as_index=False)["n"].sum().sort_values("n", ascending=False)
    )
    function_df_weighted.columns = ["object", "weighted_count"]
    function_df_weighted
    return (function_df_weighted,)


@app.cell(hide_code=True)
def _(
    config,
    function_df,
    mo,
    plot_usage,
    radio_color,
    radio_color_element,
    switch,
    switch_weigthed_element,
):
    function_fig = plot_usage(function_df, color=radio_color.value, weighted=switch.value)
    function_fig.show()
    mo.vstack(
        [
            mo.md(f"""
    Analysis of which class / functions of {config["PACKAGE_OF_INTEREST"]}
    are used split by {radio_color.value}.
    """),
            mo.hstack(
                [radio_color_element, switch_weigthed_element, mo.vstack([])],
                align="center",
            ),
        ]
    )
    return (function_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Helper functions""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Logging""")
    return


@app.cell
def _():
    import logging

    from rich.logging import RichHandler

    def poia_logger(log_level: str = "INFO") -> logging.Logger:
        FORMAT = "%(message)s"

        logging.basicConfig(
            level=log_level,
            format=FORMAT,
            datefmt="[%X]",
            handlers=[RichHandler()],
        )

        return logging.getLogger("cohort_creator")

    return RichHandler, logging, poia_logger


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Plotting""")
    return


@app.cell(hide_code=True)
def plot_usage(Version, mcolors, plt, px):
    def plot_usage(df, color=None, weighted=False):
        """Plot how frequently subpackage, classes, functions are used."""
        df = df[~(df["extracted_version"].eq("several_versions_detected"))]

        col = "object"
        y = None

        if weighted:
            group_by = ["name", col, color] if color is not None else ["name", col]
            df = df.groupby(group_by, as_index=False)["n"].sum()
            df = df.sort_values("n", ascending=False)
            y = "n"

        color_map = None
        if color:
            df = df.dropna(subset=[color])

            # Sort version labels naturally
            if color == "extracted_version":
                ordered_versions = sorted(df[color].unique(), key=Version)

                # Get Jet colors for each version using matplotlib
                cmap = plt.get_cmap("jet", len(ordered_versions))
                color_map = [mcolors.to_hex(cmap(i)) for i in range(len(ordered_versions))]

        # Aggregate and sort modules by total count
        order = df.groupby(col)["n"].count().sort_values(ascending=False).index.tolist()
        if weighted:
            order = df.groupby(col)["n"].sum().sort_values(ascending=False).index.tolist()
        category_orders = {col: order}
        if color:
            category_orders[color] = (
                ordered_versions if color == "extracted_version" else sorted(df[color].unique())
            )

        fig = px.histogram(
            df,
            x=col,
            y=y,
            title=f"Analysis of {len(df['name'].unique())} repositories",
            category_orders=category_orders,
            color_discrete_sequence=color_map,
            color=color,
        )

        fig.update_layout(xaxis_title=col, yaxis_title="Usage Count")

        return fig

    return (plot_usage,)


@app.cell(hide_code=True)
def plot_repos(Version, mcolors, plt, px):
    def plot_repos(df, color=None, bin_size="M3", include_mask=None):
        df = df[~(df["extracted_version"].eq("several_versions_detected"))]

        df.drop_duplicates(subset=["name"])

        if include_mask is not None:
            df = df[include_mask]

        category_orders = None
        color_map = None
        if color:
            df = df.dropna(subset=[color])

            # Sort version labels naturally
            category_orders = {color: sorted(df[color].unique())}
            if color == "extracted_version":
                ordered_versions = sorted(df[color].unique(), key=Version)
                category_orders = {color: ordered_versions}

                # Get Jet colors for each version using matplotlib
                cmap = plt.get_cmap("jet", len(ordered_versions))
                color_map = [mcolors.to_hex(cmap(i)) for i in range(len(ordered_versions))]

        start_date = df["last_commit"].min()
        end_date = df["last_commit"].max()

        fig = px.histogram(
            df,
            x="last_commit",
            color=color,
            category_orders=category_orders,
            color_discrete_sequence=color_map,
            title=f"Last commit date - analysis of {len(df)} repositories",
        )

        fig.update_layout(xaxis_title="Last Commit Date", yaxis_title="Usage Count")

        fig.update_xaxes(tickformat="%Y-%m")

        # Update the x-axis bin size to 3 months
        fig.update_traces(xbins={"start": start_date, "end": end_date, "size": bin_size})

        return fig

    return (plot_repos,)


@app.cell(hide_code=True)
def _(Version, px):
    def plot_versions(df):
        df = df[~(df["extracted_version"].eq("several_versions_detected"))]
        df = df[df["include"]]
        df = df.drop_duplicates(subset=["name"])

        # Drop NaNs in extracted_version
        df = df.dropna(subset=["extracted_version"])

        # Sort versions naturally
        ordered_versions = sorted(df["extracted_version"].unique(), key=Version)

        fig = px.histogram(
            df,
            x="extracted_version",
            category_orders={"extracted_version": ordered_versions},
            title=f"Analysis of {len(df)} repositories",
            color="content",
        )
        fig.update_layout(xaxis_title="Version", yaxis_title="Repository Count")
        return fig

    return (plot_versions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### AST parsing""")
    return


@app.cell(hide_code=True)
def _():
    dummy_file = """
    import nilearn
    import nilearn.plotting
    from nilearn import plotting
    from nilearn.plotting import plot_img
    from nilearn.maskers.nifti_masker import NiftiMasker

    plot_img(x)
    plotting.plot_img(x)
    plotting.view_img(x)

    NiftiMasker()
    """
    return (dummy_file,)


@app.cell(hide_code=True)
def _(ast, warnings):
    def count_imports(content, config):
        """Count imports of modules of the package of interest in a python file."""
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=SyntaxWarning)
            try:
                tree = ast.parse(content)
            except Exception as e:
                return e

        import_counts = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(config["PACKAGE_OF_INTEREST"]):
                        submodules = alias.name.split(".")[1:]
                        if len(submodules) > 0:
                            import_counts[submodules[0]] = import_counts.get(submodules[0], 0) + 1
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith(config["PACKAGE_OF_INTEREST"])
            ):
                submodules = node.module.split(".")[1:]
                if len(submodules) > 0:
                    import_counts[submodules[0]] = import_counts.get(submodules[0], 0) + 1
                else:
                    submodules = [submod.name for submod in node.names]
                    for submodule in submodules:
                        import_counts[submodule] = import_counts.get(submodule, 0) + 1

        return import_counts

    return (count_imports,)


@app.cell(hide_code=True)
def test_count_imports(config, count_imports, dummy_file):
    expected_count_imports = {"plotting": 3, "maskers": 1}
    actual_count_imports = count_imports(dummy_file, config)
    assert actual_count_imports == expected_count_imports, (
        f"{actual_count_imports=} {expected_count_imports=}"
    )
    return actual_count_imports, expected_count_imports


@app.cell(hide_code=True)
def _(ast, count_imports, warnings):
    def count_functions(content, config, imports=None):
        """Count usage of classes / functions from the package of interest in a Python file."""
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=SyntaxWarning)
            try:
                tree = ast.parse(content)
            except Exception as e:
                return e

        package = config["PACKAGE_OF_INTEREST"]
        function_counts = {}
        alias_map = {}  # Maps alias to full import path

        # First pass: build a map of imported names and their sources
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(package):
                        alias_map[alias.asname or alias.name] = alias.name
            elif (
                isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(package)
            ):
                for alias in node.names:
                    full_name = f"{node.module}.{alias.name}"
                    alias_map[alias.asname or alias.name] = full_name

        # Second pass: find all class usages
        class ClassVisitor(ast.NodeVisitor):
            def visit_Attribute(self, node):
                value = node.value
                if isinstance(value, ast.Name):
                    full_ref = alias_map.get(value.id)
                    if full_ref and full_ref.startswith(package):
                        class_name = node.attr
                        function_counts[class_name] = function_counts.get(class_name, 0) + 1
                self.generic_visit(node)

            def visit_Name(self, node):
                full_ref = alias_map.get(node.id)
                if full_ref and full_ref.startswith(package):
                    # Handle direct usage of imported class
                    class_name = full_ref.split(".")[-1]
                    function_counts[class_name] = function_counts.get(class_name, 0) + 1
                self.generic_visit(node)

        ClassVisitor().visit(tree)

        if imports is None:
            imports = count_imports(content, config)
        function_counts = {k: v for k, v in function_counts.items() if k not in imports}

        return function_counts

    return (count_functions,)


@app.cell(hide_code=True)
def test_count_functions(config, count_functions, dummy_file):
    expected_count_functions = {"plot_img": 2, "view_img": 1, "NiftiMasker": 1}
    actual_count_functions = count_functions(dummy_file, config)
    assert actual_count_functions == expected_count_functions, (
        f"{actual_count_functions=} {expected_count_functions=}"
    )
    return actual_count_functions, expected_count_functions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Github repo search""")
    return


@app.cell(hide_code=True)
def _(logger, requests, time):
    from urllib.parse import quote

    def call_api(query: str, page: int, config):
        """Wrap github API call and response handling."""
        url = config["GITHUB_API"]["SEARCH"].format(query=quote(query), page=page)
        response = requests.get(url, headers=config["GITHUB_API"]["HEADERS"])

        if response.status_code == 403:
            logger.warning(
                "GitHub API rate limit exceeded. "
                f"Waiting {config['GITHUB_API']['RATE_LIMIT_SLEEP_TIME']} "
                "seconds before retrying..."
            )
            time.sleep(config["GITHUB_API"]["RATE_LIMIT_SLEEP_TIME"])

        elif response.status_code == 422:
            logger.error(f"GitHub API query error (422). Check query format: {query}")

        elif response.status_code != 200:
            logger.error(f"Error {response.status_code}: {response.json()}")

        time.sleep(config["GITHUB_API"]["SLEEP_TIME"])

        return response

    return call_api, quote


@app.cell(hide_code=True)
def _(Path, call_api, load_cache, logger, update_cache):
    def search_repositories(queries: list[str], config, cache_file: Path | None = None):
        """Search GitHub for some queries.

        Save responses and list of repos.
        """
        repo_url_cache_file = config["OUTPUT"]["DIR"] / config["PACKAGE_OF_INTEREST"] / cache_file

        if cache_file and not config["CACHE"]["REFRESH"]:
            if repo_url_cache_file.exists():
                logger.info("🔄 Loading data from cache...")
                return load_cache(repo_url_cache_file)
            else:
                logger.info("No cache file found.")

        logger.info(f"🔍 Searching repos using '{config['PACKAGE_OF_INTEREST']}'...")

        repo_urls = set()

        for i_query, query in enumerate(queries):
            if config["DEBUG"] and i_query > 1:
                break

            logger.info(query)

            page = 1

            while True:
                response = call_api(query, page, config)

                if response.status_code == 403:
                    continue
                elif response.status_code != 200:
                    break

                results = response.json().get("items", [])
                if not results:
                    break

                for item in results:
                    repo_url = item["repository"]["html_url"]
                    repo_urls.add(repo_url)

                logger.info(f"Fetched {len(results)} results from page {page}")

                update_cache(repo_url_cache_file, repo_urls)

                page += 1
                if config["DEBUG"] and page > 1:
                    break

        logger.info("Done.")

        return list(repo_urls)

    return (search_repositories,)


@app.cell(hide_code=True)
def _(collections, logger, requests):
    from bs4 import BeautifulSoup

    def get_dependents(repo_of_interest):
        """Scrap github insights-dependency graph to find dependents."""
        dependents = set()

        for type in ["PACKAGE", "REPOSITORY"]:
            logger.info(f"Finding dependent {type.lower()} for {repo_of_interest}...")

            url = f"https://github.com/{repo_of_interest}/network/dependents?dependent_type={type}"

            nextExists = True
            while nextExists:
                logger.debug(url)

                r = requests.get(url)

                soup = BeautifulSoup(r.content, "html.parser")

                for t in soup.find_all("div", {"class": "Box-row"}):
                    user = t.find("a", {"data-hovercard-type": "user"})
                    repo = t.find("a", {"data-hovercard-type": "repository"})
                    if not user:
                        user = t.find("a", {"data-hovercard-type": "organization"})

                    if not user:
                        img = t.find("img", {"alt": "@ghost"})
                        if img:
                            logger.debug("ghost account")
                            continue
                        else:
                            logger.warning("unknown div")
                            logger.warning(f"{t}")

                    dependents.add(f"{user.text}/{repo.text}")

                nextExists = False
                if not soup.find("div", {"class": "paginate-container"}):
                    nextExists = True
                    continue
                for u in soup.find("div", {"class": "paginate-container"}).find_all("a"):
                    if u.text == "Next":
                        nextExists = True
                        url = u["href"]

        dependents = sorted(dependents)

        repo_names = {x.split("/")[1] for x in dependents}
        duplicates = [item for item, count in collections.Counter(repo_names).items() if count > 1]
        if duplicates:
            logger.info("Contain repos with same names: probably forks?")
            logger.info(duplicates)

        return dependents

    return BeautifulSoup, get_dependents


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Cloning""")
    return


@app.cell(hide_code=True)
def _(Path, config, logger, os, subprocess):
    def clone_repo(url):
        url_path = Path(url)

        repo_name = url_path.name
        user_name = url_path.parents[0].name

        (config["CACHE"]["DIR"] / user_name).mkdir(exist_ok=True)

        if (config["CACHE"]["DIR"] / user_name / repo_name).exists():
            logger.debug(f"Repo {url.replace('https://github.com/', '')} already cloned. Skipping.")
            return

        logger.info(f"Cloning {url.replace('https://github.com/', '')}.")

        os.chdir(config["CACHE"]["DIR"] / user_name)

        try:
            subprocess.run(["git", "clone", "--quiet", "--depth", "1", url], check=True)
            logger.info(f"Cloned: {url}")
        except subprocess.CalledProcessError:
            logger.error(f"Failed to clone: {url}")

    return (clone_repo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### IO""")
    return


@app.cell(hide_code=True)
def _update_cache(Path, json, load_cache, logger):
    def update_cache(cache_file: Path | None, data: list[str]):
        """Update data to a JSON cache file."""
        if cache_file is None:
            return
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        cache = load_cache(cache_file)
        cache.extend(data)
        if not any(isinstance(x, dict) for x in cache):
            try:
                cache = list(set(cache))
            except TypeError:
                logger.error("TypeError: unhashable type: 'dict'")
        with cache_file.open("w") as f:
            json.dump(cache, f, indent=2)

    return (update_cache,)


@app.cell(hide_code=True)
def _load_cache(Path, json):
    def load_cache(cache_file: Path):
        """Load data from a JSON cache file if it exists."""
        if cache_file.exists():
            with cache_file.open("r") as f:
                return json.load(f)
        return []

    return (load_cache,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### UI""")
    return


@app.cell
def _(mo):
    radio_color = mo.ui.radio(
        options={"None": None, "content": "content", "version": "extracted_version"}
    )
    return (radio_color,)


@app.cell
def _(mo, radio_color):
    radio_color_element = mo.vstack([mo.md("Color"), radio_color])
    return (radio_color_element,)


@app.cell
def _(mo):
    radio_timebin = mo.ui.radio(
        options={"1 month": "M1", "3 months": "M3", "6 months": "M6"}, value="3 months"
    )
    return (radio_timebin,)


@app.cell
def _(mo, radio_timebin):
    radio_timebin_element = mo.vstack([mo.md("Time bin"), radio_timebin])
    return (radio_timebin_element,)


@app.cell
def _(config, data_poi, mo):
    radio_include = mo.ui.radio(
        options={
            "All": None,
            f"mention version {config['PACKAGE_OF_INTEREST']}": data_poi["has_version"],
            f"import {config['PACKAGE_OF_INTEREST']}": data_poi["use_imports"],
            f"mention version of but does not import {config['PACKAGE_OF_INTEREST']}": (
                data_poi["has_version"]
            )
            & (~data_poi["use_imports"]),
            f"mention version and import {config['PACKAGE_OF_INTEREST']}": (data_poi["has_version"])
            & (data_poi["use_imports"]),
        },
        value="All",
    )
    return (radio_include,)


@app.cell
def _(mo, radio_include):
    radio_include_element = mo.vstack([mo.md("Include"), radio_include])
    return (radio_include_element,)


@app.cell
def _(mo):
    switch = mo.ui.switch()
    return (switch,)


@app.cell
def _(mo, switch):
    switch_weigthed_element = mo.vstack(
        [mo.md("Weighted by number of occurrences in each repo"), switch]
    )
    return (switch_weigthed_element,)


@app.cell
def _(mo):
    radio_x = mo.ui.radio(
        options=["n_notebook", "n_python_file", "n_notebook_parsed", "n_python_file_parsed"],
        value="n_notebook",
    )
    radio_y = mo.ui.radio(
        options=["n_notebook", "n_python_file", "n_notebook_parsed", "n_python_file_parsed"],
        value="n_notebook",
    )
    return radio_x, radio_y


@app.cell
def _(mo, radio_x, radio_y):
    radio_x_element = mo.vstack([mo.md("X"), radio_x])
    radio_y_element = mo.vstack([mo.md("X"), radio_y])
    return radio_x_element, radio_y_element


@app.cell
def _(radio_color):
    explanation_version_0 = ""
    if radio_color.value == "extracted_version":
        explanation_version_0 = """
    Version "0.0.0" is used here for cases when no version was pinned
    or the exact version could not be determined.
    """
    return (explanation_version_0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Data extraction""")
    return


@app.cell(hide_code=True)
def _(pd):
    def extract_object_count(df, col):
        """Extract sub dataframe for count of modules / classes / functions."""
        object_list = []
        for x in df.iterrows():
            if x[1][col] is None:
                continue
            if isinstance(x[1][col], str):
                x[1][col] = eval(x[1][col])
            for object, n in x[1][col].items():
                if object == "nilearn":
                    continue
                object_list.append(
                    {
                        "name": x[1]["name"],
                        "object": object,
                        "n": n,
                        "extracted_version": x[1]["extracted_version"],
                        "content": x[1]["content"],
                    }
                )
        return pd.DataFrame(object_list)

    return (extract_object_count,)


@app.cell
def _(
    extract_data_repo,
    get_last_commit_date,
    get_version,
    json,
    load_cache,
    logger,
    mo,
    shutil,
):
    def extract_data(config):
        ignore_list = load_cache(
            config["OUTPUT"]["DIR"] / config["PACKAGE_OF_INTEREST"] / config["OUTPUT"]["IGNORE"]
        )

        content_cache_file = (
            config["OUTPUT"]["DIR"] / config["PACKAGE_OF_INTEREST"] / config["OUTPUT"]["CONTENT"]
        )

        data_projects = load_cache(content_cache_file)
        repo_already_done = {x["name"] for x in data_projects}

        all_repos = []
        for user in config["CACHE"]["DIR"].iterdir():
            if user.is_dir():
                all_repos.extend(list(user.iterdir()))

        with mo.status.progress_bar(total=len(all_repos)) as bar:
            for d in all_repos:
                if not d.is_dir() or not (d / ".git").exists():
                    logger.error(f"{d} is not a git repo.")
                    bar.update()
                    continue

                repo_full_name = f"{d.parent.name}/{d.name}"
                logger.info(f"{repo_full_name}")

                if f"https://github.com/{repo_full_name}" in ignore_list:
                    logger.info("\tRepo in ignore list")
                    bar.update()
                    continue
                if repo_full_name in repo_already_done:
                    logger.info("\tRepo already analyzed.")
                    bar.update()
                    continue

                last_commit = get_last_commit_date(d)

                if last_commit is None:
                    logger.info("\tCould not get date last commit. Deleting repo.")
                    shutil.rmtree(d)
                    bar.update()
                    continue

                versions = get_version(directory=d, config=config)

                data_this_repo = extract_data_repo(repo_path=d, config=config)

                data_projects.append(
                    {
                        "name": repo_full_name,
                        "last_commit": last_commit,
                        "versions": versions,
                        **data_this_repo,
                    }
                )

                with content_cache_file.open("w") as f:
                    json.dump(data_projects, f, indent=2)

                logger.info("\tDONE")

                bar.update()

        logger.info("Data extraction done.")

    return (extract_data,)


@app.cell
def _(Path, count_functions, count_imports, find_files_with_string, logger):
    import nbformat
    from nbconvert import PythonExporter
    from nbformat.reader import NotJSONError

    def extract_data_repo(repo_path: Path, config):
        exporter = PythonExporter()

        import_counts: dict[str, int] = {}
        function_counts: dict[str, int] = {}
        n_python_file = 0
        n_python_file_parsed = 0
        n_notebook = 0
        n_notebook_parsed = 0
        contains_python_2 = False

        if not repo_path.exists():
            return {
                "n_notebook": n_notebook,
                "n_notebook_parsed": n_notebook_parsed,
                "n_python_file": n_python_file,
                "n_python_file_parsed": n_python_file_parsed,
                "import_counts": import_counts,
                "function_counts": function_counts,
                "contains_python_2": contains_python_2,
            }

        for ext in config["EXTENSIONS"]:
            files = find_files_with_string(repo_path, config, pattern=f"*.{ext}")

            logger.info(f"\t{len(files)} *.{ext} files containing {config['PACKAGE_OF_INTEREST']}")

            if ext == "py":
                n_python_file = len(files)
            else:
                n_notebook = len(files)

            for py_file in files:
                py_file = Path(py_file)

                if any(part in config["EXCLUDED_DIRS"] for part in py_file.parts):
                    logger.debug(f"File in excluded dir : {py_file.relative_to(repo_path)}")
                    continue

                try:
                    with py_file.open("r") as f:
                        if py_file.suffix == ".py":
                            content = f.read()
                        else:
                            notebook_node = nbformat.read(f, as_version=4)
                            content, _ = exporter.from_notebook_node(notebook_node)
                except UnicodeDecodeError:
                    logger.error(f"\tCould not decode file: {py_file.relative_to(repo_path)}")
                    continue
                except NotJSONError:
                    logger.error(
                        f"\tNotebook does not appear to be JSON: {py_file.relative_to(repo_path)}"
                    )
                    continue
                except Exception:
                    logger.error(f"\tError when reading: {py_file.relative_to(repo_path)}")
                    continue

                if config["PACKAGE_OF_INTEREST"] not in content:
                    logger.debug(
                        f"\t{config['PACKAGE_OF_INTEREST']} not in file: "
                        f"{py_file.relative_to(repo_path)}"
                    )
                    continue

                found_imports = count_imports(content, config)
                if not found_imports:
                    logger.debug(f"\tNo import of POI in file: {py_file.relative_to(repo_path)}")
                    continue
                if isinstance(found_imports, SyntaxError):
                    if "Missing parentheses in call to 'print'." in found_imports.msg:
                        contains_python_2 = True
                        logger.error("\tthis seems to be a python<3 file")
                    else:
                        logger.error(f"\tError parsing content: {py_file.relative_to(repo_path)}")
                        logger.error(f"\t\t {found_imports.msg}")
                    continue
                if isinstance(found_imports, Exception):
                    logger.error(f"\tError parsing content: {py_file.relative_to(repo_path)}")
                    logger.error(f"\t\t {found_imports.msg}")
                    continue

                for k, v in found_imports.items():
                    if k not in import_counts:
                        import_counts[k] = v
                    else:
                        import_counts[k] += v

                found_classes = count_functions(content, config)
                for k, v in found_classes.items():
                    if k not in function_counts:
                        function_counts[k] = v
                    else:
                        function_counts[k] += v

                if py_file.suffix == ".py":
                    n_python_file_parsed += 1
                else:
                    n_notebook_parsed += 1

        return {
            "n_notebook": n_notebook,
            "n_notebook_parsed": n_notebook_parsed,
            "n_python_file": n_python_file,
            "n_python_file_parsed": n_python_file_parsed,
            "import_counts": import_counts,
            "function_counts": function_counts,
            "contains_python_2": contains_python_2,
        }

    return NotJSONError, PythonExporter, extract_data_repo, nbformat


@app.cell(hide_code=True)
def _(logger, subprocess):
    def find_files_with_string(search_dir, config, pattern="*.py"):
        command = [
            "grep",
            "-rl",
            f"--include={pattern}",
            f"{config['PACKAGE_OF_INTEREST']}",
            str(search_dir),
        ]

        try:
            result = subprocess.run(
                command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if not result.stdout:
                return []

            files = [x for x in result.stdout.split("\n") if x != ""]
            return files

        except subprocess.CalledProcessError as e:
            # grep returns exit code 1 if no matches are found
            if e.returncode == 1:
                return []
            else:
                logger.error("An error occurred")
                return []

    return (find_files_with_string,)


@app.cell(disabled=True, hide_code=True)
def test_extract_data_repo(config, extract_data_repo):
    data_this_repo = repo_to_test = config["CACHE"]["DIR"] / "poldrack" / "myconnectome"
    extract_data_repo(repo_path=repo_to_test, config=config)
    return data_this_repo, repo_to_test


@app.cell(hide_code=True)
def _(Path, logger, subprocess):
    def get_last_commit_date(directory: Path | str) -> str | None:
        """Get date last commit of a repo on disk."""
        cmd = f'git -C "{directory}" log -1 --format="%at"'

        try:
            # Get the Unix timestamp of the last commit
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            timestamp = result.stdout.strip()

            # Convert to readable date format
            date_cmd = f"date -d @{timestamp} +%Y/%m/%d"
            date_result = subprocess.run(
                date_cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )

            return date_result.stdout.strip()

        except subprocess.CalledProcessError as e:
            logger.error(f"Error: {e}")
            return None

    return (get_last_commit_date,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Version handling""")
    return


@app.cell
def _(
    Path,
    extract_version,
    get_version_from_pyproject,
    get_version_from_setup_cfg,
    get_version_from_setup_py,
    logger,
):
    def get_version(directory: Path, config):
        versions = []
        for lockfile in config["LOCKFILES"]:
            for file in directory.glob(f"**/{lockfile}"):
                if any(part in config["EXCLUDED_DIRS"] for part in file.parts):
                    logger.debug(f"File in excluded dir : {file.relative_to(directory)}")
                    continue
                try:
                    if lockfile == "pyproject.toml":
                        version = get_version_from_pyproject(file, config)
                    elif lockfile == "setup.cfg":
                        version = get_version_from_setup_cfg(file, config)
                    elif lockfile == "setup.py":
                        version = get_version_from_setup_py(file, config)
                    else:
                        # Extract version of POI from generic text file.
                        version = None
                        with file.open("r", encoding="utf-8") as f:
                            lines = f.readlines()
                        for line in lines:
                            if config["PACKAGE_OF_INTEREST"] in line:
                                version = line
                                break
                except UnicodeDecodeError:
                    logger.error(
                        f"Could not decode file: {file.relative_to(config['CACHE']['DIR'])}"
                    )
                    version = None

                versions.append(
                    {
                        "file": str(file.relative_to(directory)),
                        "version": version,
                        "extracted_version": extract_version(version),
                    }
                )

        return versions

    return (get_version,)


@app.cell
def get_extracted_version():
    def get_extracted_version(version_list):
        tmp = [x["extracted_version"] for x in version_list if x["extracted_version"] is not None]
        if not tmp:
            return None
        elif len(set(tmp)) > 1:
            return "several_versions_detected"
        else:
            return next(iter(set(tmp)))

    return (get_extracted_version,)


@app.cell
def _(Path):
    def get_lock_file(version_list):
        tmp = [Path(x["file"]).name for x in version_list]
        if not tmp:
            return None
        elif len(set(tmp)) > 1:
            return "several_lockfile_detected"
        else:
            return next(iter(set(tmp)))

    return (get_lock_file,)


@app.cell(hide_code=True)
def _(re):
    def extract_version(version):
        """Extract actual version from a string.

        Supports X.Y and X.Y.Z formats
        """
        if not isinstance(version, str):
            return None
        if version.count(",") == 0:
            match = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", version)
            if match:
                return match.group(1)
        return "0.0.0"

    return (extract_version,)


@app.cell(hide_code=True)
def test_extract_version(extract_version):
    assert extract_version("") == "0.0.0"
    assert extract_version("nilearn==0.1") == "0.1"
    assert extract_version("nilearn><1.0") == "1.0"
    return


@app.cell(hide_code=True)
def _(Path, print):
    import toml

    def get_version_from_pyproject(pyproject_path: Path, config) -> str | None:
        """Extract version of POI from pyproject.toml."""
        default = "0.0.0"
        try:
            with pyproject_path.open("r") as f:
                pyproject = toml.load(f)

            # --- PEP 621 style ---
            project = pyproject.get("project", {})
            for dep in project.get("dependencies", []):
                if dep.lower().startswith(config["PACKAGE_OF_INTEREST"]):
                    parts = dep.split(" ", 1)
                    return parts[1].strip() if len(parts) > 1 else default

            for group_deps in project.get("optional-dependencies", {}).values():
                for dep in group_deps:
                    if dep.lower().startswith(config["PACKAGE_OF_INTEREST"]):
                        parts = dep.split(" ", 1)
                        return parts[1].strip() if len(parts) > 1 else default

            # --- Poetry style ---
            poetry = pyproject.get("tool", {}).get("poetry", {})
            deps = poetry.get("dependencies", {})

            if config["PACKAGE_OF_INTEREST"] in deps:
                entry = deps[config["PACKAGE_OF_INTEREST"]]
                if isinstance(entry, str):
                    return entry.strip() or default
                elif isinstance(entry, dict):
                    return entry.get("version", default).strip()

            return None
        except Exception as e:
            print(f"Error reading pyproject.toml: {e}")
            return None

    return get_version_from_pyproject, toml


@app.cell(hide_code=True)
def _(configparser, print):
    def get_version_from_setup_cfg(setup_cfg, config) -> str | None:
        """Extract version of POI from setup.cfg."""
        default = "0.0.0"
        try:
            cfg = configparser.ConfigParser()
            cfg.read(setup_cfg)

            # --- Check install_requires ---
            if cfg.has_option("options", "install_requires"):
                requires = cfg.get("options", "install_requires").splitlines()
                for dep in requires:
                    dep = dep.strip()
                    if dep.lower().startswith(config["PACKAGE_OF_INTEREST"]):
                        parts = dep.split(" ", 1)
                        return parts[1].strip() if len(parts) > 1 else default

            # --- Check extras_require ---
            for section in cfg.sections():
                if section.startswith("options.extras_require"):
                    for _, value in cfg.items(section):
                        deps = value.splitlines()
                        for dep in deps:
                            dep = dep.strip()
                            if dep.lower().startswith(config["PACKAGE_OF_INTEREST"]):
                                parts = dep.split(" ", 1)
                                return parts[1].strip() if len(parts) > 1 else default

            return None
        except Exception as e:
            print(f"Error reading setup.cfg: {e}")
            return None

    return (get_version_from_setup_cfg,)


@app.cell
def _(Path, ast, logger):
    def get_version_from_setup_py(setup_py: Path, config) -> str | None:
        with setup_py.open("r") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                logger.error("SyntaxError in setup.py file.")
                return None

        # Look for setup() and its keyword arguments
        class SetupVisitor(ast.NodeVisitor):
            def __init__(self):
                self.install_requires = []

            def visit_Call(self, node):
                if getattr(node.func, "id", "") == "setup":
                    for kw in node.keywords:
                        if kw.arg == "install_requires" and isinstance(
                            kw.value, (ast.List | ast.Tuple)
                        ):
                            self.install_requires = [ast.literal_eval(elt) for elt in kw.value.elts]
                self.generic_visit(node)

        visitor = SetupVisitor()
        visitor.visit(tree)

        # Print dependencies and check for nilearn
        for dep in visitor.install_requires:
            if config["PACKAGE_OF_INTEREST"] in dep:
                return dep
        return None

    return (get_version_from_setup_py,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Configuration""")
    return


@app.cell
def _set_config(GITHUB_TOKEN, Path, mo):
    config: dict[str, bool | dict[str, Path | bool | str | int | dict[str, str]]] = {
        "PACKAGE_OF_INTEREST": "nilearn",
        "PACKAGE_OF_INTEREST_REPO": "nilearn/nilearn",
        "OUTPUT": {
            "DIR": mo.notebook_dir() / "public",
            "DEPENDENTS": "dependents.json",  # repositories found by github-dependents-info
            "REPOSITORIES": "repositories.json",  # repositories to investigate
            "CONTENT": "content.json",  # content found
            "DATA": "content.csv",  # content cleaned
            "IGNORE": "ignore.json",  # repositories to ignore
        },
        "CACHE": {
            "DIR": mo.notebook_dir() / "tmp",
            "REFRESH": False,
        },
        "LOCKFILES": [
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "Pipfile",
            "environment.yml",
            "conda.yml",
            "Pipfile.lock",
            "poetry.lock",
        ],
        "DEBUG": False,
        "EXCLUDED_DIRS": {
            ".venv",
            "env",
            "venv",
            "ENV",
            "env.bak",
            "venv.bak",
            ".ipynb_checkpoints",
            "doc",
            "docs",
            "externals",
            "lib",
            "Lib",
            "__pycache__",
            "build",
            "dist",
        },
        "EXTENSIONS": [
            "ipynb",
            "py",
        ],
        "GITHUB_API": {
            "HEADERS": {
                "Authorization": f"token {GITHUB_TOKEN}",
                "Accept": "application/vnd.github.v3+json",
            },
            # Time to wait if GitHub API rate limit is hit,
            "RATE_LIMIT_SLEEP_TIME": 60,
            "SEARCH": "https://api.github.com/search/code?q={query}&per_page=100&page={page}",
            # Global sleep time between API requests (to prevent rate limits)
            "SLEEP_TIME": 2,
        },
        "N_JOBS": 4,
    }
    return (config,)


@app.cell
def _(config, itertools):
    QUERIES = [
        f'"{expression} {config["PACKAGE_OF_INTEREST"]}" AND extension:{extension}'
        for expression, extension in itertools.product(["from", "import"], config["EXTENSIONS"])
    ]

    QUERIES.extend(
        [f"{config['PACKAGE_OF_INTEREST']} in:file filename:{x}" for x in config["LOCKFILES"]]
    )
    return (QUERIES,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Imports""")
    return


@app.cell
def _():
    import argparse
    import ast
    import base64
    import collections
    import configparser
    import itertools
    import json
    import os
    import re
    import shutil
    import subprocess
    import sys
    import time
    import warnings
    from ast import literal_eval
    from pathlib import Path

    import marimo as mo
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import pandas as pd
    import plotly.express as px
    import requests
    from marimo import md
    from matplotlib import cm
    from matplotlib_venn import venn2
    from packaging.version import Version
    from rich import print

    return (
        Path,
        Version,
        argparse,
        ast,
        base64,
        cm,
        collections,
        configparser,
        itertools,
        json,
        literal_eval,
        mcolors,
        md,
        mo,
        mpl,
        os,
        pd,
        plt,
        print,
        px,
        re,
        requests,
        shutil,
        subprocess,
        sys,
        time,
        venn2,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Globals""")
    return


@app.cell
def _(os):
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    return (GITHUB_TOKEN,)


@app.cell
def _(poia_logger):
    logger = poia_logger()
    return (logger,)


if __name__ == "__main__":
    app.run()
