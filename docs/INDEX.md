# ANDES Documentation Index

Complete guide to ANDES development, CI/CD, and modernization.

## 🚀 Getting Started

### For New Users
1. **[Quick Start Guide](QUICK_START.md)** ⭐ START HERE!
   - 30-second setup
   - Common tasks
   - Troubleshooting

### For New Contributors
1. **[Quick Start Guide](QUICK_START.md)** - Get development environment running
2. **[Contributing Guidelines](../CONTRIBUTING.rst)** - How to contribute
3. **[Dependency Management](DEPENDENCIES.md)** - How to manage dependencies

### For Maintainers
1. **[Setup Summary](SETUP_SUMMARY.md)** - What was modernized and why
2. **[Pre-Merge Checklist](../.github/PRE_MERGE_CHECKLIST.md)** - Before merging PRs
3. **[Performance Monitoring](../.github/PERFORMANCE_MONITORING.md)** - Track CI performance

---

## 📚 Documentation Map

### User Documentation
- **[Main README](../README.md)** - Project overview, features, citations
- **[Quick Start](QUICK_START.md)** - Fast setup and common tasks
- **[Official Docs](https://andes.readthedocs.io)** - Complete documentation
- **[Examples](../examples/)** - Example cases and notebooks

### Developer Documentation
- **[Contributing](../CONTRIBUTING.rst)** - Contribution guidelines
- **[Dependencies](DEPENDENCIES.md)** - Managing dependencies
- **[Quick Start](QUICK_START.md)** - Development workflow
- **[Setup Summary](SETUP_SUMMARY.md)** - Modernization overview

### CI/CD Documentation
- **[Workflows README](../.github/workflows/README.md)** - How CI/CD works
- **[Performance Monitoring](../.github/PERFORMANCE_MONITORING.md)** - Track and optimize
- **[pythonapp.yml](../.github/workflows/pythonapp.yml)** - Main CI workflow
- **[ci.yml](../.github/workflows/ci.yml)** - Matrix testing workflow

### Maintainer Documentation
- **[Setup Summary](SETUP_SUMMARY.md)** - Complete modernization overview
- **[PR Description Template](../.github/PR_DESCRIPTION.md)** - For modernization PR
- **[Pre-Merge Checklist](../.github/PRE_MERGE_CHECKLIST.md)** - Before merging
- **[Performance Monitoring](../.github/PERFORMANCE_MONITORING.md)** - CI metrics

---

## 🎯 Common Tasks - Quick Reference

### Installation

```bash
# User installation
pip install andes

# Developer installation (recommended)
pip install -e ".[dev]"

# With uv (10x faster)
uv pip install -e ".[dev]"
```

**Documentation:** [Quick Start](QUICK_START.md)

### Adding Dependencies

1. Edit `pyproject.toml`
2. Reinstall: `pip install -e ".[dev]"`
3. Test: `pytest`

**Documentation:** [Dependencies](DEPENDENCIES.md)

### Running Tests

```bash
# All tests
pytest

# Specific test
pytest tests/test_model.py

# Fast (parallel)
pytest -n auto

# Watch mode
ptw
```

**Documentation:** [Quick Start](QUICK_START.md)

### CI/CD

- **Check status:** [GitHub Actions](https://github.com/CURENT/andes/actions)
- **Understand workflows:** [Workflows README](../.github/workflows/README.md)
- **Monitor performance:** [Performance Guide](../.github/PERFORMANCE_MONITORING.md)

### Contributing

1. Fork repository
2. Create branch: `git checkout -b feature-name`
3. Make changes
4. Run tests: `pytest`
5. Push and create PR

**Documentation:** [Contributing](../CONTRIBUTING.rst)

---

## 📖 Document Descriptions

### Core Documents

#### [Quick Start Guide](QUICK_START.md)
**For:** Everyone
**Purpose:** Fast setup and common tasks
**Contents:**
- 30-second installation
- Development workflow
- Common tasks
- Troubleshooting

#### [Dependencies Guide](DEPENDENCIES.md)
**For:** Developers, Maintainers
**Purpose:** Managing project dependencies
**Contents:**
- Single source of truth (pyproject.toml)
- Adding/updating dependencies
- Version constraints
- Best practices

#### [Setup Summary](SETUP_SUMMARY.md)
**For:** Maintainers
**Purpose:** Understanding modernization changes
**Contents:**
- Complete overview of modernization
- Before/after comparisons
- Performance improvements
- All commits explained

### CI/CD Documents

#### [Workflows README](../.github/workflows/README.md)
**For:** Developers, DevOps
**Purpose:** Understanding CI/CD system
**Contents:**
- Why uv instead of mamba
- How caching works
- Cache invalidation strategy
- Performance comparisons
- Troubleshooting

#### [Performance Monitoring Guide](../.github/PERFORMANCE_MONITORING.md)
**For:** Maintainers, DevOps
**Purpose:** Track and optimize CI performance
**Contents:**
- Key metrics to monitor
- How to check performance
- Common issues and solutions
- Optimization opportunities
- Alerting setup

### Maintainer Documents

#### [PR Description Template](../.github/PR_DESCRIPTION.md)
**For:** Maintainers
**Purpose:** PR description for modernization changes
**Contents:**
- Complete PR description
- Performance impact
- Migration guide
- Review notes

#### [Pre-Merge Checklist](../.github/PRE_MERGE_CHECKLIST.md)
**For:** Maintainers
**Purpose:** Ensure safe merge
**Contents:**
- Code review checklist
- Testing verification
- Backwards compatibility
- Documentation checks
- Rollback plan

---

## 🔄 Modernization Changes

The ANDES project underwent major modernization in November 2025:

### What Changed?

1. **Packaging** - Migrated to modern pyproject.toml (PEP 621)
2. **CI/CD** - Replaced mamba+pip with uv (10-100x faster)
3. **Dependencies** - Single source of truth (no more duplication)
4. **Reliability** - No more hanging workflows (100% reliable)

### Impact?

- ⚡ 3x faster CI runs (8-12 min → 3-5 min)
- ✅ 100% reliable (no more hangs)
- 📝 Single source of truth (1 file vs 3)
- 🚀 Modern standards (PEP 621)

**Full details:** [Setup Summary](SETUP_SUMMARY.md)

---

## 🗺️ Navigation Tips

### I want to...

**Install ANDES for development**
→ [Quick Start](QUICK_START.md) → 30-second setup

**Add a new dependency**
→ [Dependencies](DEPENDENCIES.md) → "Adding Dependencies" section

**Understand CI failures**
→ [Workflows README](../.github/workflows/README.md) → "Troubleshooting" section

**Check CI performance**
→ [Performance Monitoring](../.github/PERFORMANCE_MONITORING.md) → "Key Metrics" section

**Prepare to merge modernization PR**
→ [Pre-Merge Checklist](../.github/PRE_MERGE_CHECKLIST.md)

**Understand what was modernized**
→ [Setup Summary](SETUP_SUMMARY.md)

**Contribute code**
→ [Contributing](../CONTRIBUTING.rst) → [Quick Start](QUICK_START.md)

---

## 📊 Documentation Status

| Document | Status | Last Updated | Maintained By |
|----------|--------|--------------|---------------|
| Quick Start | ✅ Current | 2025-11-05 | Core team |
| Dependencies | ✅ Current | 2025-11-05 | Core team |
| Setup Summary | ✅ Current | 2025-11-05 | Core team |
| Workflows README | ✅ Current | 2025-11-05 | DevOps team |
| Performance Monitoring | ✅ Current | 2025-11-05 | DevOps team |
| PR Description | ✅ Current | 2025-11-05 | Maintainers |
| Pre-Merge Checklist | ✅ Current | 2025-11-05 | Maintainers |

---

## 🤝 Contributing to Documentation

Found an issue or want to improve docs?

1. Edit the relevant `.md` file
2. Follow [Markdown style guide](https://www.markdownguide.org/)
3. Submit a PR with clear description
4. Update this index if adding new docs

---

## 🆘 Getting Help

1. **Search existing docs** - Use this index to find relevant guide
2. **Check issues** - [GitHub Issues](https://github.com/CURENT/andes/issues)
3. **Ask in discussions** - [GitHub Discussions](https://github.com/CURENT/andes/discussions)
4. **Read the FAQ** - Common questions answered
5. **File a bug** - If something's broken

---

## 📅 Document Update Schedule

- **Quick Start** - Updated as needed when setup changes
- **Dependencies** - Updated when dependency strategy changes
- **Setup Summary** - Static (historical record)
- **Workflows README** - Updated when CI/CD changes
- **Performance Monitoring** - Updated quarterly
- **Checklists** - Reviewed before each major release

---

## 🎓 Learning Path

### For New Users

1. Read [Quick Start](QUICK_START.md) (5 min)
2. Install ANDES (1 min)
3. Run examples (10 min)
4. Read [Official Docs](https://andes.readthedocs.io) (as needed)

### For New Contributors

1. Read [Quick Start](QUICK_START.md) (5 min)
2. Read [Contributing](../CONTRIBUTING.rst) (10 min)
3. Read [Dependencies](DEPENDENCIES.md) (5 min)
4. Fork, code, test, submit PR (∞ min)

### For New Maintainers

1. Read [Setup Summary](SETUP_SUMMARY.md) (15 min)
2. Read [Workflows README](../.github/workflows/README.md) (10 min)
3. Read [Performance Monitoring](../.github/PERFORMANCE_MONITORING.md) (10 min)
4. Review [Pre-Merge Checklist](../.github/PRE_MERGE_CHECKLIST.md) (5 min)
5. Understand all above (30-60 min)

---

**Happy developing!** 🚀

*This index is maintained by the ANDES core team. Last updated: 2025-11-05*
