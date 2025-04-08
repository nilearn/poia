import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
        # POIA: Package Of Interest Audit

        Look for usage of a package of interest (POI) on public repos on github.

        Will list repo that:
        - contain the POI in one the common files used to declare depdendencies (pyproject.toml, setup.cfg, requirements.txt...)
        - import the POI in a python module or a ipython notebook.

        Then it will clone those repos and collect information about them.
        """
    )


@app.cell
def _():
    import argparse
    import ast
    import base64
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
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from urllib.parse import quote

    import marimo as mo
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import pandas as pd
    import plotly.express as px
    import requests
    import toml
    from loguru import logger
    from marimo import md
    from matplotlib import cm
    from matplotlib_venn import venn2
    from packaging.version import Version
    from rich import print
    from tqdm import tqdm

    return (
        Path,
        ThreadPoolExecutor,
        Version,
        argparse,
        ast,
        base64,
        cm,
        configparser,
        itertools,
        json,
        logger,
        mcolors,
        md,
        mo,
        mpl,
        os,
        pd,
        plt,
        print,
        px,
        quote,
        re,
        requests,
        shutil,
        subprocess,
        sys,
        time,
        toml,
        tqdm,
        venn2,
        warnings,
    )


@app.cell
def _(logger, sys):
    logger.remove()  # Remove the default handler.
    logger.add(sys.stderr, level="DEBUG")


@app.cell
def _(os):
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    return (GITHUB_TOKEN,)


@app.cell
def _(mo):
    mo.md(r"""## Configuration""")


@app.cell
def _set_config(GITHUB_TOKEN, Path, __file__):
    config: dict[
        str, bool | dict[str, Path | bool | str | int | dict[str, str]]
    ] = {
        "CACHE": {
            "DIR": Path(__file__).parent / "tmp",
            "REPOSITORIES": "repositories.json",  # repositories to investigate
            "IGNORE": "ignore.json",  # repositories to ignore
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
        "PACKAGE_OF_INTEREST": "nilearn",
        "N_JOBS": 18,
    }
    return (config,)


@app.cell
def _(config, itertools):
    EXTENSIONS = [
        "extension:ipynb",
        "extension:py",
    ]

    QUERIES = [
        f'"{expression} {config["PACKAGE_OF_INTEREST"]}" AND {extension}'
        for expression, extension in itertools.product(
            ["from", "import"], EXTENSIONS
        )
    ]

    QUERIES.extend(
        [
            f"{config['PACKAGE_OF_INTEREST']} in:file filename:{x}"
            for x in config["LOCKFILES"]
        ]
    )
    return EXTENSIONS, QUERIES


@app.cell
def _(mo):
    mo.md(r"""## Search repos on github""")


@app.cell
def _(QUERIES, config, search_repositories):
    repos = search_repositories(
        queries=QUERIES,
        config=config,
        cache_file=config["CACHE"]["REPOSITORIES"],
    )
    return (repos,)


@app.cell
def _(Path, config, logger, repos, shutil):
    logger.info(f"{len(set(repos))} found.")

    cloned_repos = set()

    for r in set(repos):
        url_path = Path(r)

        repo_name = url_path.name
        user_name = url_path.parents[0].name

        if (config["CACHE"]["DIR"] / user_name / repo_name).exists():
            if (
                config["CACHE"]["DIR"] / user_name / repo_name / ".git"
            ).exists():
                cloned_repos.add(r)
            else:
                shutil.rmtree(r)

        if (config["CACHE"]["DIR"] / user_name).exists() and len(
            list((config["CACHE"]["DIR"] / user_name).iterdir())
        ) == 0:
            shutil.rmtree(config["CACHE"]["DIR"] / user_name)

    logger.info(f"{len(cloned_repos)} already cloned.")
    return cloned_repos, r, repo_name, url_path, user_name


@app.cell
def _(mo):
    mo.md(r"""## Cloning repos""")


@app.cell
def _(Path, config, logger, os, subprocess):
    def clone_repo(url):
        url_path = Path(url)

        repo_name = url_path.name
        user_name = url_path.parents[0].name

        (config["CACHE"]["DIR"] / user_name).mkdir(exist_ok=True)

        if (config["CACHE"]["DIR"] / user_name / repo_name).exists():
            logger.info(
                f"Repo {url.replace('https://github.com/', '')} already cloned. Skipping."
            )
            return

        logger.info(f"Cloning {url.replace('https://github.com/', '')}.")

        os.chdir(config["CACHE"]["DIR"] / user_name)

        try:
            subprocess.run(
                ["git", "clone", "--quiet", "--depth", "1", url], check=True
            )
            logger.info(f"Cloned: {url}")
        except subprocess.CalledProcessError:
            logger.error(f"Failed to clone: {url}")

    return (clone_repo,)


@app.cell
def _(ThreadPoolExecutor, clone_repo, config, repos):
    with ThreadPoolExecutor(max_workers=config["N_JOBS"]) as executor:
        executor.map(clone_repo, repos)
    return (executor,)


@app.cell
def _(mo):
    mo.md(r"""## Extracting data from repos""")


@app.cell
def _(
    config,
    count_functions,
    count_imports,
    extract_version,
    get_last_commit_date,
    get_version,
    get_version_from_pyproject,
    get_version_from_setup_cfg,
    load_cache,
    logger,
    shutil,
):
    import nbformat
    from nbconvert import PythonExporter
    from nbformat.reader import NotJSONError

    exporter = PythonExporter()

    data_projects = []

    ignore_list = load_cache(
        config["CACHE"]["DIR"] / config["CACHE"]["IGNORE"]
    )

    excluded_dirs = {"venv", ".ipynb_checkpoints"}

    for user in config["CACHE"]["DIR"].iterdir():
        if not user.is_dir():
            continue

        for d in user.iterdir():
            if not (d / ".git").exists():
                logger.error(f"{d} is not a git repo.")

            if not d.is_dir():
                continue

            repo_full_name = f"{user.name}/{d.name}"

            if f"https://github.com/{repo_full_name}" in ignore_list:
                continue

            last_commit = get_last_commit_date(d)

            if last_commit is None:
                shutil.rmtree(d)
                continue

            versions = []
            for lockfile in config["LOCKFILES"]:
                for file in d.glob(f"**/{lockfile}"):
                    try:
                        if lockfile == "pyproject.toml":
                            version = get_version_from_pyproject(file, config)
                        elif lockfile == "setup.cfg":
                            version = get_version_from_setup_cfg(file, config)
                        else:
                            version = get_version(file)
                    except UnicodeDecodeError:
                        logger.error(
                            f"Could not decode file: {file.relative_to(config['CACHE']['DIR'])}"
                        )
                        version = None

                    versions.append(
                        {
                            "file": file.relative_to(d),
                            "version": version,
                            "extracted_version": extract_version(version),
                        }
                    )

            extracted_version = [
                x["extracted_version"]
                for x in versions
                if x["extracted_version"] is not None
            ]
            if not extracted_version:
                extracted_version = None
            elif len(set(extracted_version)) > 1:
                extracted_version = "several_versions_detected"
            else:
                extracted_version = next(iter(set(extracted_version)))

            import_counts = {}
            class_counts = {}

            for pat in ["**/*.py", "**/*.ipynb"]:
                for py_file in d.glob(pat):
                    if any(
                        part in excluded_dirs
                        for part in py_file.relative_to(
                            config["CACHE"]["DIR"]
                        ).parts
                    ):
                        continue

                    try:
                        with py_file.open("r") as f:
                            if py_file.suffix == ".py":
                                content = f.read()
                            else:
                                notebook_node = nbformat.read(f, as_version=4)
                                content, _ = exporter.from_notebook_node(
                                    notebook_node
                                )

                    except UnicodeDecodeError:
                        logger.error(
                            f"Could not decode file: {py_file.relative_to(config['CACHE']['DIR'])}"
                        )
                        continue

                    except NotJSONError:
                        logger.error(
                            f"Notebook does not appear to be JSON: {py_file.relative_to(config['CACHE']['DIR'])}"
                        )
                        continue

                    except:
                        logger.error(
                            f"Error when reading: {py_file.relative_to(config['CACHE']['DIR'])}"
                        )
                        continue

                    if config["PACKAGE_OF_INTEREST"] not in content:
                        continue

                    found_imports = count_imports(content, config)

                    if not found_imports:
                        continue
                    if isinstance(found_imports, Exception):
                        logger.error(
                            f"Error parsing content: {py_file.relative_to(config['CACHE']['DIR'])}"
                        )
                        continue

                    for k, v in found_imports.items():
                        if k not in import_counts:
                            import_counts[k] = v
                        else:
                            import_counts[k] += v

                    found_classes = count_functions(content, config)
                    for k, v in found_classes.items():
                        if k not in class_counts:
                            class_counts[k] = v
                        else:
                            class_counts[k] += v

            data_projects.append(
                {
                    "name": repo_full_name,
                    "last_commit": last_commit,
                    "versions": versions,
                    "extracted_version": extracted_version,
                    "import_counts": import_counts if import_counts else None,
                    "function_counts": class_counts if class_counts else None,
                }
            )
    return (
        NotJSONError,
        PythonExporter,
        class_counts,
        content,
        d,
        data_projects,
        excluded_dirs,
        exporter,
        extracted_version,
        f,
        file,
        found_classes,
        found_imports,
        ignore_list,
        import_counts,
        k,
        last_commit,
        lockfile,
        nbformat,
        notebook_node,
        pat,
        py_file,
        repo_full_name,
        user,
        v,
        version,
        versions,
    )


@app.cell
def _(data_projects, pd):
    data_projects_df = pd.DataFrame(data_projects)

    data_projects_df["last_commit"] = pd.to_datetime(
        data_projects_df["last_commit"]
    )

    data_projects_df = data_projects_df[
        ~(
            data_projects_df["extracted_version"].eq(
                "several_versions_detected"
            )
        )
    ]

    data_projects_df["has_version"] = data_projects_df[
        "extracted_version"
    ].astype("bool", errors="ignore")

    data_projects_df["has_imports"] = data_projects_df["import_counts"].astype(
        "bool", errors="ignore"
    )

    data_projects_df["use_imports"] = data_projects_df[
        "function_counts"
    ].astype("bool", errors="ignore")

    data_projects_df["include"] = (
        data_projects_df["has_version"]
        | data_projects_df["has_imports"]
        | data_projects_df["use_imports"]
    )

    data_projects_df[["user", "repo"]] = data_projects_df["name"].str.split(
        "/", expand=True
    )

    data_projects_df["duplicated_repo"] = data_projects_df["repo"].duplicated()
    return (data_projects_df,)


@app.cell
def _(data_projects_df):
    data_projects_df[data_projects_df["include"]].drop(["versions"], axis=1)


@app.cell
def _(data_projects_df, mo):
    transformed_df = mo.ui.dataframe(
        data_projects_df.drop(["versions"], axis=1)
    )
    # transformed_df
    return (transformed_df,)


@app.cell
def _(data_projects_df, pd):
    import_list = []
    for x in data_projects_df[data_projects_df["has_imports"]].iterrows():
        if x[1]["import_counts"] is None:
            continue
        for module, count in x[1]["import_counts"].items():
            import_list.append(
                {
                    "name": x[1]["name"],
                    "module": module,
                    "count": count,
                    "extracted_version": x[1]["extracted_version"],
                }
            )
    import_df = pd.DataFrame(import_list)
    return count, import_df, import_list, module, x


@app.cell
def _(import_df, plot_usage):
    plot_usage(import_df)


@app.cell
def _(import_df, plot_usage):
    plot_usage(import_df, color="extracted_version")


@app.cell
def _(data_projects_df, pd):
    function_list = []
    for row in data_projects_df[data_projects_df["use_imports"]].iterrows():
        if row[1]["function_counts"] is None:
            continue
        for function, ct in row[1]["function_counts"].items():
            function_list.append(
                {
                    "name": row[1]["name"],
                    "function": function,
                    "count": ct,
                    "extracted_version": row[1]["extracted_version"],
                }
            )
    function_df = pd.DataFrame(function_list)
    return ct, function, function_df, function_list, row


@app.cell
def _(function_df, plot_usage):
    plot_usage(function_df, col="function")


@app.cell
def _(function_df, plot_usage):
    plot_usage(function_df, col="function", color="extracted_version")


@app.cell
def _(Version, mcolors, plt, px):
    def plot_usage(df, col="module", color=None):
        color_map = None
        if color:
            df = df.dropna(subset=[color])

            # Sort version labels naturally
            if df[color].dtype == "object":
                ordered_versions = sorted(df[color].unique(), key=Version)

                # Get Jet colors for each version using matplotlib
                cmap = plt.get_cmap("jet", len(ordered_versions))
                color_map = [
                    mcolors.to_hex(cmap(i))
                    for i in range(len(ordered_versions))
                ]

        # Aggregate and sort modules by total count
        order = (
            df.groupby(col)["count"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )

        category_orders = {col: order}
        if color:
            category_orders = {col: order, color: ordered_versions}

        fig = px.histogram(
            df,
            x=col,
            title=f"Analysis of {len(df['name'].unique())} projects",
            category_orders=category_orders,
            color_discrete_sequence=color_map,
            color=color,
        )

        fig.update_layout(xaxis_title=col, yaxis_title="Usage Count")

        fig.show()

    return (plot_usage,)


@app.cell
def _(data_projects_df, plot_repos):
    plot_repos(data_projects_df)


@app.cell
def _(data_projects_df, plot_repos):
    plot_repos(data_projects_df, color="extracted_version")


@app.cell
def _(data_projects_df, plot_versions):
    plot_versions(data_projects_df)


@app.cell
def _(mo):
    mo.md(
        r"""
        Now we have list of repos that import the POI
        and those that have it as a dependency.

        Let's try to see how many projects have
        the POI as depedency but do not import it.
        """
    )


@app.cell
def _(config, data_projects_df, plt, venn2):
    as_dependency = data_projects_df["name"][
        data_projects_df["has_version"]
    ].to_list()
    actually_import = data_projects_df["name"][
        data_projects_df["has_imports"]
    ].to_list()
    venn2(
        subsets=(
            set(as_dependency),
            set(actually_import),
        ),
        set_labels=(
            f"{config['PACKAGE_OF_INTEREST']} as dependency",
            f"import {config['PACKAGE_OF_INTEREST']}",
        ),
    )
    plt.show()
    return actually_import, as_dependency


@app.cell
def _(actually_import, config, data_projects_df, plt, venn2):
    use_import = data_projects_df["name"][
        data_projects_df["use_imports"]
    ].to_list()
    venn2(
        subsets=(
            set(actually_import),
            set(use_import),
        ),
        set_labels=(
            f"import {config['PACKAGE_OF_INTEREST']}",
            f"use {config['PACKAGE_OF_INTEREST']}",
        ),
    )
    plt.show()
    return (use_import,)


@app.cell
def _(Version, mcolors, plt, px):
    def plot_repos(df, color=None):
        df.drop_duplicates(subset=["name"])
        df = df[df["include"]]

        category_orders = None
        color_map = None
        if color:
            df = df.dropna(subset=[color])

            # Sort version labels naturally
            if df[color].dtype == "object":
                ordered_versions = sorted(df[color].unique(), key=Version)
                category_orders = {color: ordered_versions}

                # Get Jet colors for each version using matplotlib
                cmap = plt.get_cmap("jet", len(ordered_versions))
                color_map = [
                    mcolors.to_hex(cmap(i))
                    for i in range(len(ordered_versions))
                ]

        start_date = df["last_commit"].min()
        end_date = df["last_commit"].max()

        fig = px.histogram(
            df,
            x="last_commit",
            color=color,
            category_orders=category_orders,
            color_discrete_sequence=color_map,
            title=f"Analysis of {len(df)} projects",
        )

        fig.update_layout(
            xaxis_title="Last Commit Date", yaxis_title="Usage Count"
        )

        fig.update_xaxes(tickformat="%Y-%m")

        # Update the x-axis bin size to 3 months
        fig.update_traces(
            xbins={"start": start_date, "end": end_date, "size": "M3"}
        )

        fig.show()

    return (plot_repos,)


@app.cell
def _(Version, px):
    def plot_versions(df):
        df = df.drop_duplicates(subset=["name"])

        # Drop NaNs in extracted_version
        df = df.dropna(subset=["extracted_version"])

        # Sort versions naturally
        ordered_versions = sorted(
            df["extracted_version"].unique(), key=Version
        )

        fig = px.histogram(
            df,
            x="extracted_version",
            category_orders={"extracted_version": ordered_versions},
            title=f"Analysis of {len(df)} projects",
        )
        fig.update_layout(
            xaxis_title="Version", yaxis_title="Repository Count"
        )
        fig.show()

    return (plot_versions,)


@app.cell
def _(Path, logger, subprocess):
    def get_last_commit_date(directory: Path | str) -> str | None:
        """Get date last commit of a repo on disk."""
        cmd = f'git -C "{directory}" log -1 --format="%at"'

        try:
            # Get the Unix timestamp of the last commit
            result = subprocess.run(
                cmd, shell=True, check=True, capture_output=True, text=True
            )
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


@app.cell
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
                            import_counts[submodules[0]] = (
                                import_counts.get(submodules[0], 0) + 1
                            )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith(config["PACKAGE_OF_INTEREST"])
            ):
                submodules = node.module.split(".")[1:]
                if len(submodules) > 0:
                    import_counts[submodules[0]] = (
                        import_counts.get(submodules[0], 0) + 1
                    )
                else:
                    submodules = [submod.name for submod in node.names]
                    for submodule in submodules:
                        import_counts[submodule] = (
                            import_counts.get(submodule, 0) + 1
                        )

        return import_counts

    return (count_imports,)


@app.cell
def _(ast, warnings):
    def count_functions(content, config):
        """Count usage of classes / functions from the package of interest in a Python file."""
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=SyntaxWarning)
            try:
                tree = ast.parse(content)
            except Exception as e:
                return e

        package = config["PACKAGE_OF_INTEREST"]
        class_usage = {}
        alias_map = {}  # Maps alias to full import path

        # First pass: build a map of imported names and their sources
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(package):
                        alias_map[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(package):
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
                        class_usage[class_name] = (
                            class_usage.get(class_name, 0) + 1
                        )
                self.generic_visit(node)

            def visit_Name(self, node):
                full_ref = alias_map.get(node.id)
                if full_ref and full_ref.startswith(package):
                    # Handle direct usage of imported class
                    class_name = full_ref.split(".")[-1]
                    class_usage[class_name] = (
                        class_usage.get(class_name, 0) + 1
                    )
                self.generic_visit(node)

        ClassVisitor().visit(tree)

        return class_usage

    return (count_functions,)


@app.cell
def _(Path, call_api, load_cache, logger, update_cache):
    def search_repositories(
        queries: list[str], config, cache_file: Path | None = None
    ):
        """Search GitHub for some queries.

        Save responses and list of repos.
        """
        repo_url_cache_file = (
            config["CACHE"]["DIR"]
            / f"{config['PACKAGE_OF_INTEREST']}_{cache_file}"
        )

        if cache_file and not config["CACHE"]["REFRESH"]:
            if repo_url_cache_file.exists():
                logger.info("🔄 Loading data from cache...")
                return load_cache(repo_url_cache_file)
            else:
                logger.info("No cache file found.")

        logger.info(
            f"🔍 Searching repos using '{config['PACKAGE_OF_INTEREST']}'..."
        )

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


@app.cell
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


@app.cell
def _load_cache(Path, json):
    def load_cache(cache_file: Path):
        """Load data from a JSON cache file if it exists."""
        if cache_file.exists():
            with cache_file.open("r") as f:
                return json.load(f)
        return []

    return (load_cache,)


@app.cell
def _(logger, quote, requests, time):
    def call_api(query: str, page: int, config):
        """Wrap github API call and response handling."""
        url = config["GITHUB_API"]["SEARCH"].format(
            query=quote(query), page=page
        )
        response = requests.get(url, headers=config["GITHUB_API"]["HEADERS"])

        if response.status_code == 403:
            logger.warning(
                "GitHub API rate limit exceeded. "
                f"Waiting {config['GITHUB_API']['RATE_LIMIT_SLEEP_TIME']} "
                "seconds before retrying..."
            )
            time.sleep(config["GITHUB_API"]["RATE_LIMIT_SLEEP_TIME"])

        elif response.status_code == 422:
            logger.error(
                f"GitHub API query error (422). Check query format: {query}"
            )

        elif response.status_code != 200:
            logger.error(f"Error {response.status_code}: {response.json()}")

        time.sleep(config["GITHUB_API"]["SLEEP_TIME"])

        return response

    return (call_api,)


@app.cell
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


@app.cell
def _(Path, print, toml):
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

            for group_deps in project.get(
                "optional-dependencies", {}
            ).values():
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

    return (get_version_from_pyproject,)


@app.cell
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
                            if dep.lower().startswith(
                                config["PACKAGE_OF_INTEREST"]
                            ):
                                parts = dep.split(" ", 1)
                                return (
                                    parts[1].strip()
                                    if len(parts) > 1
                                    else default
                                )

            return None
        except Exception as e:
            print(f"Error reading setup.cfg: {e}")
            return None

    return (get_version_from_setup_cfg,)


@app.cell
def _(Path, config):
    def get_version(file: Path) -> str | None:
        """Extract version of POI from generic text file."""
        with file.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if config["PACKAGE_OF_INTEREST"] in line:
                return line
        return None

    return (get_version,)


if __name__ == "__main__":
    app.run()
