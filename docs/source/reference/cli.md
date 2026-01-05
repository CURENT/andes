# Command Line Interface

ANDES provides a command-line interface (CLI) for running simulations, plotting results, and managing the software without writing Python code. This reference documents all available commands and options.

## Basic Usage

```bash
andes <command> [options] [arguments]
```

## Commands Overview

| Command | Purpose |
|---------|---------|
| `run` | Run power system simulations |
| `plot` | Plot time-domain simulation results |
| `doc` | View model and routine documentation |
| `prepare` | Generate numerical code from models |
| `selftest` | Verify installation |
| `misc` | Utility functions |

## andes run

Run power system simulations on case files.

### Basic Examples

```bash
# Power flow (default routine)
andes run case.xlsx

# Time-domain simulation
andes run case.xlsx -r tds

# Eigenvalue analysis
andes run case.xlsx -r eig

# PSS/E files (RAW + DYR)
andes run system.raw --addfile system.dyr -r tds
```

### Routine Options

| Option | Routine | Description |
|--------|---------|-------------|
| (none) | PFlow | Newton-Raphson power flow |
| `-r tds` | TDS | Time-domain simulation |
| `-r eig` | EIG | Eigenvalue analysis |
| `-r pflow,tds` | Multiple | Run multiple routines |

### Simulation Control

```bash
# Set simulation end time
andes run case.xlsx -r tds --tf 30

# Set time step
andes run case.xlsx -r tds --tstep 0.01

# Configuration options
andes run case.xlsx -r tds -O TDS.tf=30 -O TDS.tstep=0.01
```

### File Options

```bash
# Add dynamic data file
andes run case.raw --addfile case.dyr

# Convert to XLSX format
andes run case.raw --convert

# Convert to JSON format
andes run case.raw --convert json

# Specify output directory
andes run case.xlsx -o results/

# Disable output files
andes run case.xlsx -n
```

### Parallel Execution

```bash
# Run multiple cases (auto-parallel)
andes run *.xlsx -r tds

# Run with wildcard pattern
andes run kundur_*.xlsx -r tds

# Limit CPU usage
andes run *.xlsx -r tds --ncpu 4
```

### Interactive Shell

Exit to IPython shell after simulation for interactive analysis:

```bash
andes run kundur_full.xlsx -r tds -s -n
```

The System object is available as `system`:

```python
In [1]: system.GENROU.omega.v
Out[1]: array([1.0, 1.0, 1.0, 1.0])
```

### Complete Options

| Option | Description |
|--------|-------------|
| `-r ROUTINE` | Routine to run (pflow, tds, eig) |
| `--tf TF` | Simulation end time [s] |
| `--tstep STEP` | Time step [s] |
| `--addfile FILE` | Additional file (e.g., DYR) |
| `-O KEY=VALUE` | Configuration option |
| `-o PATH` | Output directory |
| `-n, --no-output` | Disable file output |
| `-c, --convert` | Convert file format |
| `--ncpu N` | Number of CPUs |
| `-s, --shell` | Exit to IPython shell |
| `-b SHEET` | Add workbook sheet (e.g., Fault) |

## andes plot

Plot results from time-domain simulation.

### Basic Usage

```bash
# Plot variable index 5 vs time (index 0)
andes plot case_out.lst 0 5

# Plot multiple variables
andes plot case_out.lst 0 5 6 7 8

# Plot range of variables
andes plot case_out.lst 0 2:21:6
```

### Finding Variable Indices

The `.lst` file contains variable names and indices:

```
0, Time [s], $Time\ [s]$
1, delta GENROU 1, $\delta\ GENROU\ 1$
5, omega GENROU 1, $\omega\ GENROU\ 1$
```

Search by name:

```bash
# Find omega variables
andes plot case_out.lst --xargs "omega GENROU"
```

### Plot Options

| Option | Description |
|--------|-------------|
| `--savefig` | Save figure to file |
| `--save FILE` | Save to specific filename |
| `-d` | Disable LaTeX rendering |
| `--to-csv` | Export data to CSV |

## andes doc

View documentation for models and routines.

```bash
# Model documentation
andes doc GENROU

# List all models
andes doc -l
andes doc --list

# Routine documentation
andes doc TDS

# Search models
andes doc --list | grep -i exciter
```

## andes prepare

Generate numerical code from symbolic model definitions. Code generation happens automatically on first use; manual preparation is mainly needed during model development.

```bash
# Full code generation
andes prepare

# Force regeneration
andes prepare -f

# Incremental (only changed models)
andes prepare -i

# Quick mode (skip unchanged)
andes prepare -q
```

## andes selftest

Verify installation by running the test suite.

```bash
# Full test
andes selftest

# Quick test (skip code generation)
andes selftest -q
```

## andes misc

Utility functions.

```bash
# Show version
andes misc --version

# Edit configuration file
andes misc --edit-config

# Clean output files in current directory
andes misc -C

# Clean output files recursively
andes misc -C -r

# Save configuration to file
andes --save-config
```

## Verbosity Levels

Control output detail with `-v LEVEL`:

| Level | Name | Description |
|-------|------|-------------|
| 10 | DEBUG | Detailed debugging info |
| 20 | INFO | Normal output (default) |
| 30 | WARNING | Warnings only |
| 40 | ERROR | Errors only |

```bash
# Debug output
andes -v 10 run case.xlsx

# Quiet mode (warnings only)
andes -v 30 run case.xlsx
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANDES_USE_UMFPACK` | Use UMFPACK sparse solver |
| `ANDES_DISABLE_NUMBA` | Disable Numba JIT compilation |

## Common Workflows

### Standard Analysis

```bash
# Power flow only
andes run case.xlsx

# Power flow + TDS
andes run case.xlsx -r pflow,tds --tf 20

# All analyses
andes run case.xlsx -r pflow,tds,eig --tf 20
```

### Batch Studies

```bash
# Run all cases in parallel
andes run cases/*.xlsx -r tds --ncpu 8

# Limit to 4 processes
andes run cases/*.xlsx -r tds --ncpu 4
```

### Convert and Simulate

```bash
# Convert PSS/E to XLSX
andes run system.raw --addfile system.dyr --convert

# Run simulation on converted file
andes run system.xlsx -r tds --tf 20
```

### Add Disturbance Sheets

```bash
# Add Fault sheet to workbook
andes run case.xlsx -b Fault

# Add multiple sheets
andes run case.xlsx -b Fault,Toggle,Alter
```

## Getting Help

```bash
# General help
andes --help

# Command-specific help
andes run --help
andes plot --help
andes doc --help
```

## See Also

- {doc}`config` - Configuration options
- {doc}`../tutorials/index` - Tutorials with examples
