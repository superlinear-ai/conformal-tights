[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/radix-ai/conformal-tights) [![Open in GitHub Codespaces](https://img.shields.io/static/v1?label=GitHub%20Codespaces&message=Open&color=blue&logo=github)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=765698489&skip_quickstart=true)

# 👖 Conformal Tights

A [scikit-learn meta-estimator](https://scikit-learn.org/stable/glossary.html#term-meta-estimator) that adds [conformal prediction](https://en.wikipedia.org/wiki/Conformal_prediction) of coherent [quantiles](https://en.wikipedia.org/wiki/Quantile) and [intervals](https://en.wikipedia.org/wiki/Prediction_interval) to any [scikit-learn regressor](https://scikit-learn.org/stable/glossary.html#term-regressor). Features:

1. 🍬 *Meta-estimator*: add prediction of quantiles and intervals to any scikit-learn regressor
2. 🌡️ *Conformally calibrated:* accurate quantiles, and intervals with reliable [coverage](https://en.wikipedia.org/wiki/Coverage_probability)
3. 🚦 *Coherent quantiles:* quantiles increase monotonically instead of [crossing](https://github.com/dmlc/xgboost/issues/9848) [each other](https://github.com/microsoft/LightGBM/issues/3447)
4. 👖 *Tight quantiles:* selects the lowest [dispersion](https://en.wikipedia.org/wiki/Statistical_dispersion) that provides the desired coverage
5. 🎁 *Data efficient:* requires only a small number of calibration examples to fit
6. 🐼 *Pandas support:* optionally predict on DataFrames and receive DataFrame output

## Using

### Installing

First, install this package with:

```sh
pip install conformal-tights
```

### Predicting quantiles

Conformal Tights exposes a meta-estimator called `ConformalCoherentQuantileRegressor` that you can use to wrap any scikit-learn regressor, after which you can use `predict_quantiles` to predict conformally calibrated quantiles. Example usage:

```python
from conformal_tights import ConformalCoherentQuantileRegressor
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Fetch dataset and split in train and test
X, y = fetch_openml("ames_housing", version=1, return_X_y=True, as_frame=True, parser="auto")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Create a regressor, wrap it, and fit on the train set
my_regressor = XGBRegressor(objective="reg:absoluteerror")
conformal_predictor = ConformalCoherentQuantileRegressor(estimator=my_regressor)
conformal_predictor.fit(X_train, y_train)

# Predict with the wrapped regressor
ŷ_test = conformal_predictor.predict(X_test)

# Predict quantiles with the conformal wrapper
ŷ_test_quantiles = conformal_predictor.predict_quantiles(X_test, quantiles=(0.025, 0.05, 0.1, 0.9, 0.95, 0.975))
```

When the input data is a pandas DataFrame, the output is also a pandas DataFrame. For example, printing the head of `ŷ_test_quantiles` yields:

|   house_id |   0.025 |   0.05 |    0.1 |    0.9 |   0.95 |   0.975 |
|-----------:|--------:|-------:|-------:|-------:|-------:|--------:|
|       1357 |  121557 | 130272 | 139913 | 189399 | 211177 |  237309 |
|       2367 |   86005 |  92617 |  98591 | 130236 | 145686 |  164766 |
|       2822 |  116523 | 121711 | 134993 | 175583 | 194964 |  216891 |
|       2126 |  105712 | 113784 | 122145 | 164330 | 183352 |  206224 |
|       1544 |   85920 |  92311 |  99130 | 133228 | 148895 |  167969 |

Let's visualize the predicted quantiles on the test set:

<img src="https://github.com/radix-ai/conformal-tights/assets/4543654/b02b3797-de6a-4e0d-b457-ed8e50e3f42c" width="512">

<details>
<summary>Expand to see the code that generated the graph above</summary>

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
%config InlineBackend.figure_format = "retina"
plt.rcParams["font.size"] = 8
idx = (-ŷ_test.sample(50, random_state=42)).sort_values().index
y_ticks = list(range(1, len(idx) + 1))
plt.figure(figsize=(4, 5))
for j in range(3):
    end = ŷ_test_quantiles.shape[1] - 1 - j
    coverage = round(100 * (ŷ_test_quantiles.columns[end] - ŷ_test_quantiles.columns[j]))
    plt.barh(
        y_ticks,
        ŷ_test_quantiles.loc[idx].iloc[:, end] - ŷ_test_quantiles.loc[idx].iloc[:, j],
        left=ŷ_test_quantiles.loc[idx].iloc[:, j],
        label=f"{coverage}% Prediction interval",
        color=["#b3d9ff", "#86bfff", "#4da6ff"][j],
    )
plt.plot(y_test.loc[idx], y_ticks, "s", markersize=3, markerfacecolor="none", markeredgecolor="#e74c3c", label="Actual value")
plt.plot(ŷ_test.loc[idx], y_ticks, "s", color="blue", markersize=0.6, label="Predicted value")
plt.xlabel("House price")
plt.ylabel("Test house index")
plt.yticks(y_ticks, y_ticks)
plt.tick_params(axis="y", labelsize=6)
plt.grid(axis="x", color="lightsteelblue", linestyle=":", linewidth=0.5)
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.legend()
plt.tight_layout()
plt.show()
```
</details>

### Predicting intervals

In addition to quantile prediction, you can use `predict_interval` to predict conformally calibrated prediction intervals. Compared to quantiles, these focus on reliable coverage over quantile accuracy. Example usage:

```python
# Predict an interval for each example with the conformal wrapper
ŷ_test_interval = conformal_predictor.predict_interval(X_test, coverage=0.95)

# Measure the coverage of the prediction intervals on the test set
coverage = ((ŷ_test_interval.iloc[:, 0] <= y_test) & (y_test <= ŷ_test_interval.iloc[:, 1])).mean()
print(coverage)  # 96.6%
```

When the input data is a pandas DataFrame, the output is also a pandas DataFrame. For example, printing the head of `ŷ_test_interval` yields:

|   house_id |   0.025 |   0.975 |
|-----------:|--------:|--------:|
|       1357 |  108489 |  238396 |
|       2367 |   76043 |  165189 |
|       2822 |  101319 |  220247 |
|       2126 |   94238 |  207501 |
|       1544 |   75976 |  168741 |

## Contributing

<details>
<summary>Prerequisites</summary>

<details>
<summary>1. Set up Git to use SSH</summary>

1. [Generate an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key) and [add the SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).
1. Configure SSH to automatically load your SSH keys:
    ```sh
    cat << EOF >> ~/.ssh/config
    Host *
      AddKeysToAgent yes
      IgnoreUnknown UseKeychain
      UseKeychain yes
    EOF
    ```

</details>

<details>
<summary>2. Install Docker</summary>

1. [Install Docker Desktop](https://www.docker.com/get-started).
    - Enable _Use Docker Compose V2_ in Docker Desktop's preferences window.
    - _Linux only_:
        - Export your user's user id and group id so that [files created in the Dev Container are owned by your user](https://github.com/moby/moby/issues/3206):
            ```sh
            cat << EOF >> ~/.bashrc
            export UID=$(id --user)
            export GID=$(id --group)
            EOF
            ```

</details>

<details>
<summary>3. Install VS Code or PyCharm</summary>

1. [Install VS Code](https://code.visualstudio.com/) and [VS Code's Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers). Alternatively, install [PyCharm](https://www.jetbrains.com/pycharm/download/).
2. _Optional:_ install a [Nerd Font](https://www.nerdfonts.com/font-downloads) such as [FiraCode Nerd Font](https://github.com/ryanoasis/nerd-fonts/tree/master/patched-fonts/FiraCode) and [configure VS Code](https://github.com/tonsky/FiraCode/wiki/VS-Code-Instructions) or [configure PyCharm](https://github.com/tonsky/FiraCode/wiki/Intellij-products-instructions) to use it.

</details>

</details>

<details open>
<summary>Development environments</summary>

The following development environments are supported:

1. ⭐️ _GitHub Codespaces_: click on _Code_ and select _Create codespace_ to start a Dev Container with [GitHub Codespaces](https://github.com/features/codespaces).
1. ⭐️ _Dev Container (with container volume)_: click on [Open in Dev Containers](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/radix-ai/conformal-tights) to clone this repository in a container volume and create a Dev Container with VS Code.
1. _Dev Container_: clone this repository, open it with VS Code, and run <kbd>Ctrl/⌘</kbd> + <kbd>⇧</kbd> + <kbd>P</kbd> → _Dev Containers: Reopen in Container_.
1. _PyCharm_: clone this repository, open it with PyCharm, and [configure Docker Compose as a remote interpreter](https://www.jetbrains.com/help/pycharm/using-docker-compose-as-a-remote-interpreter.html#docker-compose-remote) with the `dev` service.
1. _Terminal_: clone this repository, open it with your terminal, and run `docker compose up --detach dev` to start a Dev Container in the background, and then run `docker compose exec dev zsh` to open a shell prompt in the Dev Container.

</details>

<details>
<summary>Developing</summary>

- This project follows the [Conventional Commits](https://www.conventionalcommits.org/) standard to automate [Semantic Versioning](https://semver.org/) and [Keep A Changelog](https://keepachangelog.com/) with [Commitizen](https://github.com/commitizen-tools/commitizen).
- Run `poe` from within the development environment to print a list of [Poe the Poet](https://github.com/nat-n/poethepoet) tasks available to run on this project.
- Run `poetry add {package}` from within the development environment to install a run time dependency and add it to `pyproject.toml` and `poetry.lock`. Add `--group test` or `--group dev` to install a CI or development dependency, respectively.
- Run `poetry update` from within the development environment to upgrade all dependencies to the latest versions allowed by `pyproject.toml`.
- Run `cz bump` to bump the package's version, update the `CHANGELOG.md`, and create a git tag.

</details>
