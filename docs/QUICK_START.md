# Quick Start Guide: Modern ANDES Development

## 🚀 30-Second Setup

```bash
# Clone repository
git clone https://github.com/CURENT/andes.git
cd andes

# Install everything (core + dev tools)
pip install -e ".[dev]"

# Run tests
pytest

# You're ready! 🎉
```

That's it! No more juggling requirements files or conda environments.

---

## 💡 What Changed?

### Old Way (Complex)
```bash
# Multiple steps, multiple files
pip install -r requirements.txt
pip install -r requirements-extra.txt
pip install -e .
```

### New Way (Simple)
```bash
# One command, one source of truth
pip install -e ".[dev]"
```

---

## 📦 Installation Options

```bash
# Minimal - just ANDES core
pip install andes

# Development - core + dev tools
pip install -e ".[dev]"

# Documentation - core + doc building tools
pip install -e ".[doc]"

# Everything - all optional features
pip install -e ".[all]"
```

---

## ⚡ Super Fast Option: uv

Want **10x faster** installs? Use `uv`:

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Use it like pip, but way faster
uv pip install -e ".[dev]"  # 30 seconds instead of 5 minutes!

# Run commands
uv run pytest
uv run andes --help
```

---

## 🧪 Development Workflow

```bash
# 1. Make your changes
vim andes/core/model.py

# 2. Run tests
pytest

# Or specific tests
pytest tests/test_model.py

# 3. Check code style
flake8 andes tests

# 4. Commit
git add .
git commit -m "Add awesome feature"
git push
```

---

## 📝 Common Tasks

### Add a New Dependency

Edit `pyproject.toml`:
```toml
[project]
dependencies = [
    "existing-package>=1.0",
    "new-package>=2.0,<3.0",  # Add here
]
```

Then reinstall:
```bash
pip install -e ".[dev]"
```

### Update Dependency Version

Edit `pyproject.toml`:
```toml
# Change version constraint
"numpy>=1.20.0,<2.2"  →  "numpy>=1.20.0,<2.3"
```

Reinstall:
```bash
pip install -e ".[dev]" --upgrade
```

### Run Tests Faster

```bash
# Use multiple cores
pytest -n auto

# Run only failed tests
pytest --lf

# Stop on first failure
pytest -x

# Run specific test
pytest tests/test_model.py::test_function
```

### Build Documentation

```bash
# Install doc dependencies
pip install -e ".[doc]"

# Build docs
cd docs
make html

# View in browser
open build/html/index.html
```

---

## 🐛 Troubleshooting

### "Module not found"

```bash
# Reinstall in editable mode
pip install -e ".[dev]"
```

### "Dependency conflict"

```bash
# Check for conflicts
pip check

# Reinstall from scratch
pip uninstall andes
pip install -e ".[dev]"
```

### "Tests failing"

```bash
# Update all dependencies
pip install -e ".[dev]" --upgrade

# Clear pytest cache
pytest --cache-clear

# Run with verbose output
pytest -vv
```

### "CI failing"

Check:
1. Did you update `pyproject.toml` properly?
2. Are version constraints correct?
3. Check GitHub Actions logs for details
4. See `.github/workflows/README.md`

---

## 📚 Where is Everything?

```
andes/
├── pyproject.toml              # 👈 All dependencies defined here
├── setup.py                    # Minimal shim (don't edit)
├── andes/                      # Source code
│   ├── core/                   # Core functionality
│   ├── models/                 # Power system models
│   └── routines/               # Analysis routines
├── tests/                      # Test suite
├── docs/                       # Documentation
│   ├── DEPENDENCIES.md         # 👈 How to manage dependencies
│   └── SETUP_SUMMARY.md        # 👈 Complete modernization summary
└── .github/
    └── workflows/
        ├── README.md           # 👈 CI/CD documentation
        ├── pythonapp.yml       # Main CI workflow
        └── ci.yml              # Matrix testing

# Legacy files (archived, don't use)
requirements.txt.legacy         # ⚠️ Don't edit
requirements-extra.txt.legacy   # ⚠️ Don't edit
```

---

## 🎓 Learn More

- **Dependencies:** Read `docs/DEPENDENCIES.md`
- **CI/CD:** Read `.github/workflows/README.md`
- **Full Summary:** Read `docs/SETUP_SUMMARY.md`
- **Contributing:** Read `CONTRIBUTING.rst`

---

## 🆘 Need Help?

1. **Check documentation** in `docs/` folder
2. **Search issues** on GitHub
3. **Ask in discussions** on GitHub
4. **File a bug report** if something's broken

---

## ✨ Pro Tips

### Use uv for Speed
```bash
uv pip install -e ".[dev]"  # 10x faster than pip
uv run pytest                # No need to activate venv
```

### Use pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
# Now code checks run automatically on commit
```

### Use Parallel Testing
```bash
pip install pytest-xdist
pytest -n auto  # Use all CPU cores
```

### Use Watch Mode
```bash
pip install pytest-watch
ptw  # Auto-run tests when files change
```

---

## 🎉 You're All Set!

ANDES development is now:
- ✅ **Simple** - One command to install
- ✅ **Fast** - 10x faster with uv
- ✅ **Modern** - Latest Python standards
- ✅ **Reliable** - CI never hangs

Happy coding! 🚀

---

**Questions?** Check the docs or open a GitHub discussion!
