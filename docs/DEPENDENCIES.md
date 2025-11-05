# Dependencies Management

## Single Source of Truth: `pyproject.toml`

All project dependencies are defined in `pyproject.toml` following modern Python packaging standards (PEP 621).

**Do NOT edit `requirements.txt` or `requirements-extra.txt` directly.** These files are legacy artifacts and have been archived.

## Dependency Groups

Dependencies are organized in `pyproject.toml`:

### Core Dependencies
```toml
[project]
dependencies = [
    "numpy>=1.20.0,<2.3",
    "scipy>=1.7.0,<1.14",
    # ... more core deps
]
```

### Optional Dependencies
```toml
[project.optional-dependencies]
dev = [...]   # Development tools
doc = [...]   # Documentation building
interop = [...] # Interoperability packages
web = [...]   # Web interface
all = [...]   # All optional dependencies
```

## Installation

### For Users
```bash
# Install ANDES with core dependencies
pip install andes

# Install with optional features
pip install andes[dev]      # Development tools
pip install andes[doc]      # Documentation building
pip install andes[all]      # Everything
```

### For Developers
```bash
# Clone repository
git clone https://github.com/CURENT/andes.git
cd andes

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or use uv (faster)
uv pip install -e ".[dev]"
```

## Generating requirements.txt (Legacy Tools)

Some legacy tools require `requirements.txt`. Generate it from `pyproject.toml`:

```bash
# Generate requirements.txt (core dependencies only)
python scripts/generate_requirements.py

# Generate with optional dependencies
python scripts/generate_requirements.py --extra dev
python scripts/generate_requirements.py --extra all

# Custom output file
python scripts/generate_requirements.py --output requirements-custom.txt
```

**Note:** Generated `requirements.txt` files are for legacy tool compatibility only. Always edit `pyproject.toml` as the source.

## Adding Dependencies

### Adding Core Dependencies

Edit `pyproject.toml`:

```toml
[project]
dependencies = [
    "existing-package>=1.0",
    "new-package>=2.0,<3.0",  # ← Add here
]
```

### Adding Optional Dependencies

```toml
[project.optional-dependencies]
dev = [
    "existing-dev-tool",
    "new-dev-tool>=1.0",  # ← Add here
]
```

### Version Constraints

Follow semantic versioning:

```toml
# Recommended: Allow patches and minors, exclude majors
"numpy>=1.20.0,<2.0"

# Allow patches only (very strict)
"requests>=2.28.0,<2.29"

# Exclude specific versions (if broken)
"sympy>=1.6,!=1.10.0,<2.0"

# Minimum version only (not recommended for stability)
"matplotlib>=3.3.0"
```

## Checking Dependencies

```bash
# Check for conflicts
pip check

# Or with uv
uv pip check

# List installed packages
pip list

# Show dependency tree
pip show andes
```

## Updating Dependencies

1. **Update `pyproject.toml`**
   ```bash
   # Edit pyproject.toml to change version constraints
   vim pyproject.toml
   ```

2. **Reinstall**
   ```bash
   pip install -e ".[dev]" --upgrade
   ```

3. **Test**
   ```bash
   pytest
   ```

4. **Commit**
   ```bash
   git add pyproject.toml
   git commit -m "Update dependencies: <description>"
   ```

## CI/CD

GitHub Actions workflows install directly from `pyproject.toml`:

```yaml
# Single source of truth
- run: uv pip install -e ".[dev]"
```

Cache invalidation is automatic when `pyproject.toml` changes:

```yaml
cache:
  key: ${{ hashFiles('pyproject.toml') }}
```

## Migration from requirements.txt

If you have an old workflow using `requirements.txt`:

**Before:**
```bash
pip install -r requirements.txt
pip install -r requirements-extra.txt
pip install -e .
```

**After:**
```bash
pip install -e ".[dev]"
```

That's it! Everything is defined in `pyproject.toml`.

## Troubleshooting

### "requirements.txt not found"

Legacy scripts may look for `requirements.txt`. Generate it:

```bash
python scripts/generate_requirements.py
```

### Dependency conflicts

```bash
# Check for conflicts
uv pip check

# View installed versions
uv pip list

# Reinstall from scratch
pip uninstall andes
pip install -e ".[dev]"
```

### Cache issues in CI

GitHub Actions cache key includes `pyproject.toml` hash. When you update dependencies:
1. Change version constraints in `pyproject.toml`
2. Push changes
3. Cache automatically invalidates
4. Fresh dependencies installed

## Best Practices

1. ✅ **Always edit `pyproject.toml`** - single source of truth
2. ✅ **Use version constraints** - prevent breaking changes
3. ✅ **Pin upper bounds** - ensure compatibility
4. ✅ **Document changes** - explain why dependencies updated
5. ✅ **Test thoroughly** - run full test suite after updates
6. ❌ **Don't edit requirements.txt manually** - it's generated/legacy
7. ❌ **Don't skip version constraints** - causes breakage
8. ❌ **Don't use `>=X` without upper bound** - allows breaking changes

## References

- [PEP 621 - Project Metadata](https://peps.python.org/pep-0621/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)
- [uv Documentation](https://github.com/astral-sh/uv)
