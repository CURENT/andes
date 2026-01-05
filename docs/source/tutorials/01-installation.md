# Installation

## Quick Install

::::{tab-set}

:::{tab-item} conda (recommended)
```bash
conda install -c conda-forge andes
```
:::

:::{tab-item} pip
```bash
pip install andes
```
:::

:::{tab-item} uv
```bash
uv pip install andes
```
:::

::::

## New to Python

### Setting Up miniforge

If you are new to Python and want to get started quickly, use miniforge, a conda-like package manager configured with conda-forge.

**Step 1:** Download the latest miniforge for your platform from [miniforge releases](https://github.com/conda-forge/miniforge#miniforge).

- Most users: `x86_64(amd64)` for Intel and AMD processors
- Mac with Apple Silicon: `arm64(Apple Silicon)` for best performance

Complete the installation on your system.

:::{note}
miniforge is a drop-in replacement for conda. If you have an existing conda installation, you can replace all `mamba` commands with `conda`.

If you are using Anaconda or Miniconda on Windows, open `Anaconda Prompt` instead of `Miniforge Prompt`.
:::

**Step 2:** Open Terminal (Linux/macOS) or `Miniforge Prompt` (Windows, **not cmd!**).

You should see `(base)` prepended to the command prompt, e.g., `(base) C:\Users\username>`.

Create an environment for ANDES:

```bash
mamba create --name andes python=3.11
```

Activate the new environment:

```bash
mamba activate andes
```

:::{note}
You will need to activate the `andes` environment every time in a new terminal session.
:::

## Using uv

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver. If you're already familiar with uv, you can use it to install ANDES.

**Install into current environment:**

```bash
uv pip install andes
```

**Create a new virtual environment with ANDES:**

```bash
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install andes
```

**With extras:**

```bash
uv pip install andes[dev]
```

**Development install:**

```bash
git clone https://github.com/curent/andes
cd andes
uv pip install -e .[dev]
```

:::{tip}
For uv installation and detailed usage, see the [uv documentation](https://docs.astral.sh/uv/).
:::

## Extra Packages

Some ANDES features require extra packages not installed by default:

| Group | Description |
|-------|-------------|
| `dev` | Development packages (testing, documentation) |
| `interop` | Interoperability with other power system tools |

Install extras with pip:

```bash
# Development packages
pip install andes[dev]

# All extras
pip install andes[all]
```

:::{note}
Extra packages are not supported by conda/mamba installation. Use `pip` for extras.
:::

## Development Install

For users who want to modify code or develop new models. Changes to source code are reflected immediately without reinstallation.

**Step 1:** Clone the source code:

```bash
git clone https://github.com/curent/andes
```

**Step 2:** Install in development mode:

```bash
cd andes
pip install -e .
```

With extras:

```bash
pip install -e .[dev]
```

:::{note}
ANDES uses `setuptools-scm` for versioning based on git tags. The version updates automatically when you `git pull` new changes.

Check version: `andes` or `python -c "import andes; print(andes.__version__)"`
:::

## Updating ANDES

:::{warning}
If installed in development mode, use `git pull` to update. Do not run `conda install` or `pip install` as this creates duplicate installations.
:::

**conda/mamba:**
```bash
conda install -c conda-forge --yes andes
```

**pip:**
```bash
pip install --upgrade andes
```

**uv:**
```bash
uv pip install --upgrade andes
```

Check [Release Notes](../reference/release-notes.md) before updating for breaking changes.

## Troubleshooting

### Multiple Copies Installed

If you have both development and package installations, uninstall all copies:

```bash
conda remove andes
pip uninstall andes
```

Run both commands multiple times until neither finds the package.

### Windows DLL Error

If you see:
```
ImportError: DLL load failed: The specified module could not be found.
```

This is a Windows Python path issue. The easiest fix is to install ANDES in a Conda/miniforge environment.

## Next Steps

With ANDES installed, you're ready to run your first simulation:

- {doc}`02-first-simulation` - Load a test case and run power flow and time-domain simulation
- {doc}`03-power-flow` - Deep dive into power flow analysis
- {doc}`../reference/cli` - Command-line interface reference
