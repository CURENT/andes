# Pre-Merge Checklist for Modernization PR

Use this checklist before merging the modernization changes to main branch.

## ✅ Code Review

- [ ] **Review pyproject.toml**
  - [ ] All dependency versions look reasonable
  - [ ] Version constraints not too strict or too loose
  - [ ] Python version requirement (≥3.9) is acceptable
  - [ ] Optional dependency groups make sense

- [ ] **Review GitHub Actions workflows**
  - [ ] `pythonapp.yml` changes look good
  - [ ] `ci.yml` matrix covers needed platforms
  - [ ] Cache keys are correct
  - [ ] Timeout values are reasonable

- [ ] **Review setup.py**
  - [ ] Minimal shim approach is acceptable
  - [ ] Versioneer integration still works
  - [ ] Error messages are helpful

- [ ] **Review documentation**
  - [ ] README.md accurate (if modified)
  - [ ] CONTRIBUTING.rst updated correctly
  - [ ] New docs (DEPENDENCIES.md, SETUP_SUMMARY.md) are helpful

## ✅ Testing Verification

- [ ] **Local testing**
  - [ ] Clone fresh copy of branch
  - [ ] `pip install -e ".[dev]"` works
  - [ ] `pytest` runs successfully
  - [ ] `andes` CLI works
  - [ ] Can import all major modules

- [ ] **CI testing**
  - [ ] Check latest CI run on branch
  - [ ] All jobs passing (Ubuntu, macOS, Windows if applicable)
  - [ ] No hanging workflows
  - [ ] CI time is faster than before
  - [ ] Cache is working (check logs for "cache hit/miss")

- [ ] **Cross-platform**
  - [ ] Ubuntu tests pass
  - [ ] macOS tests pass (if in matrix)
  - [ ] Windows tests pass (if in matrix)

## ✅ Backwards Compatibility

- [ ] **User installation**
  - [ ] `pip install andes` still works
  - [ ] `pip install andes[dev]` still works
  - [ ] Existing users won't break on upgrade

- [ ] **Developer workflow**
  - [ ] Editable install still works: `pip install -e .`
  - [ ] With extras still works: `pip install -e ".[dev]"`
  - [ ] No breaking changes to setup process

- [ ] **Legacy compatibility**
  - [ ] Decision made on `requirements*.txt.legacy` files
    - [ ] Keep for reference?
    - [ ] Remove entirely?
    - [ ] Add warning in README?

## ✅ Documentation

- [ ] **New docs created**
  - [ ] `.github/workflows/README.md` reviewed
  - [ ] `docs/DEPENDENCIES.md` reviewed
  - [ ] `docs/SETUP_SUMMARY.md` reviewed

- [ ] **Existing docs updated**
  - [ ] Main README.md mentions modern installation
  - [ ] CONTRIBUTING.rst reflects Python 3.9+ requirement
  - [ ] Any installation guides updated

- [ ] **In-code documentation**
  - [ ] pyproject.toml has clear comments
  - [ ] setup.py docstring explains purpose
  - [ ] Workflow files have explanatory comments

## ✅ Performance

- [ ] **CI performance**
  - [ ] Compare CI time before/after
  - [ ] Verify 2-3x speedup achieved
  - [ ] Check cache hit rate is reasonable (70%+)

- [ ] **Installation performance**
  - [ ] Compare local install time
  - [ ] Verify uv is significantly faster
  - [ ] No unexpected slowdowns

## ✅ Migration Plan

- [ ] **Communication**
  - [ ] Announcement draft prepared (see PR_DESCRIPTION.md)
  - [ ] Breaking changes (if any) documented
  - [ ] Migration guide clear and tested

- [ ] **Rollout strategy**
  - [ ] Merge to develop first? Or directly to main?
  - [ ] Tag new version after merge?
  - [ ] Update release notes?

- [ ] **Support**
  - [ ] Ready to answer questions about new setup
  - [ ] Troubleshooting guide in place (workflows/README.md)
  - [ ] Know how to roll back if issues arise

## ✅ Cleanup

- [ ] **Unused files**
  - [ ] Decision made on legacy requirements files
  - [ ] No leftover test files or temp directories
  - [ ] No debugging code left in

- [ ] **Git hygiene**
  - [ ] Commit messages are clear and descriptive
  - [ ] No unnecessary commits (squash if needed)
  - [ ] Branch is up to date with target branch

## ✅ Final Checks

- [ ] **All tests passing**
  ```bash
  # Run locally
  pip install -e ".[dev]"
  pytest

  # Check CI
  # Visit: https://github.com/CURENT/andes/actions
  ```

- [ ] **No conflicts**
  - [ ] Branch merges cleanly with main/master
  - [ ] No merge conflicts

- [ ] **Approvals**
  - [ ] Required reviewers approved
  - [ ] All comments addressed
  - [ ] Any requested changes made

## 🚀 Ready to Merge?

If all boxes above are checked:

1. **Squash and merge** (recommended) or regular merge
2. **Delete branch** after merge
3. **Monitor** next CI run on main
4. **Announce** the changes to team/users
5. **Update** any dependent projects/documentation

## 📋 Post-Merge Tasks

- [ ] Verify CI runs successfully on main branch
- [ ] Check that main branch build status is green
- [ ] Announce changes (Slack, email, release notes)
- [ ] Update external documentation if needed
- [ ] Close related issues
- [ ] Tag a release if appropriate

## ⚠️ Rollback Plan

If issues arise after merge:

1. **Revert the merge commit:**
   ```bash
   git revert -m 1 <merge-commit-sha>
   git push origin main
   ```

2. **Or create hotfix:**
   - Fix issues on new branch
   - Fast-track review
   - Merge fix

3. **Known safe rollback points:**
   - Before: commit `1eecc20`
   - After: commit `f2bdaa4`

---

## ✍️ Sign-off

**Reviewer:** _________________
**Date:** _________________
**Notes:** _________________

**Approved for merge:** ☐ Yes ☐ No ☐ With changes

---

**Last Updated:** $(date)
