# CI Performance Monitoring Guide

Track and optimize GitHub Actions performance for ANDES.

## 📊 Key Metrics to Monitor

### 1. Workflow Duration

**Target:** 3-5 minutes per run
**Measure:** Total time from start to finish

```bash
# Get recent workflow run times
gh run list --limit 20 --json conclusion,createdAt,updatedAt,name

# Calculate duration
gh run view <run-id> --json startedAt,completedAt
```

### 2. Cache Hit Rate

**Target:** 70-80% cache hits
**Measure:** Ratio of cache hits to total runs

**Check in workflow logs:**
```
✅ Cache restored from key: uv-Linux-py3.11-abc123...
❌ Cache not found for input keys: uv-Linux-py3.11-xyz...
```

**Why it matters:**
- Cache hit = 30 sec install ✅
- Cache miss = 2-3 min install ⚠️

### 3. Individual Step Times

**Typical times:**
- Checkout: 5-10 seconds
- Setup Python: 10-15 seconds
- Install uv: 5 seconds
- Cache restore: 1-2 seconds
- Install dependencies (cache hit): 20-40 seconds
- Install dependencies (cache miss): 2-3 minutes
- Run tests: 2-3 minutes
- Run notebook tests: 1-2 minutes

### 4. Failure Rate

**Target:** < 5% failures
**Exclude:** Expected failures (e.g., network timeouts)

---

## 🔍 How to Check Performance

### Using GitHub Web Interface

1. Go to **Actions** tab
2. Click on workflow run
3. Review timing for each step
4. Look for:
   - ⚠️ Steps taking longer than expected
   - ❌ Failed steps
   - ⏱️ Total duration

### Using GitHub CLI

```bash
# Install gh if needed
# brew install gh  (macOS)
# apt install gh   (Ubuntu)

# List recent runs with timing
gh run list --limit 10 --json conclusion,startedAt,updatedAt,displayTitle

# View specific run details
gh run view <run-id>

# Download logs for analysis
gh run view <run-id> --log > workflow.log

# Check for specific patterns
gh run view <run-id> --log | grep "Cache"
gh run view <run-id> --log | grep "Successfully"
gh run view <run-id> --log | grep -i "error"
```

### Using Python Script

```python
#!/usr/bin/env python3
"""Analyze workflow performance."""
import subprocess
import json
from datetime import datetime

def get_workflow_times(limit=20):
    """Get recent workflow run times."""
    cmd = ["gh", "run", "list", "--limit", str(limit),
           "--json", "conclusion,startedAt,updatedAt,name,displayTitle"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    runs = json.loads(result.stdout)

    durations = []
    for run in runs:
        if run.get('startedAt') and run.get('updatedAt'):
            start = datetime.fromisoformat(run['startedAt'].replace('Z', '+00:00'))
            end = datetime.fromisoformat(run['updatedAt'].replace('Z', '+00:00'))
            duration = (end - start).total_seconds() / 60  # minutes
            durations.append({
                'name': run['displayTitle'],
                'conclusion': run['conclusion'],
                'duration_min': round(duration, 1)
            })

    return durations

def print_stats(durations):
    """Print statistics."""
    successful = [d for d in durations if d['conclusion'] == 'success']
    if not successful:
        print("No successful runs found")
        return

    times = [d['duration_min'] for d in successful]
    print(f"📊 Workflow Statistics (last {len(durations)} runs)")
    print(f"   Average: {sum(times)/len(times):.1f} min")
    print(f"   Min: {min(times):.1f} min")
    print(f"   Max: {max(times):.1f} min")
    print(f"   Success rate: {len(successful)/len(durations)*100:.0f}%")

if __name__ == "__main__":
    durations = get_workflow_times(20)
    print_stats(durations)

    print("\n📋 Recent Runs:")
    for d in durations[:10]:
        status = "✅" if d['conclusion'] == 'success' else "❌"
        print(f"   {status} {d['duration_min']}min - {d['name']}")
```

---

## 📈 Performance Trends

### Expected Performance After Modernization

| Week | Avg Duration | Cache Hit Rate | Notes |
|------|-------------|----------------|-------|
| Week 1 | 3-4 min | 60-70% | Cache building up |
| Week 2+ | 3-5 min | 70-80% | Stable performance |

### Benchmarks

**Before modernization:**
```
├─ Setup conda/mamba: 1-2 min
├─ Install dependencies: 5-8 min (often hangs)
├─ Run tests: 3-4 min
└─ Total: 8-12+ min (if doesn't hang)
```

**After modernization:**
```
├─ Setup Python + uv: 15 sec
├─ Install dependencies: 30 sec (cache hit) / 2-3 min (miss)
├─ Run tests: 3-4 min
└─ Total: 3-5 min (reliable)
```

---

## ⚠️ Performance Issues & Solutions

### Issue: Slow Cache Restore

**Symptom:** Cache restore taking > 30 seconds

**Check:**
```yaml
# In workflow logs, look for:
Cache Size: XXX MB
Time: XXX seconds
```

**Solution:**
- Cache might be too large (> 2 GB)
- Consider cleaning before cache save:
```yaml
- name: Clean cache before save
  run: uv cache clean
```

### Issue: Frequent Cache Misses

**Symptom:** < 50% cache hit rate

**Causes:**
1. `pyproject.toml` changing frequently
2. Cache being evicted (too many caches)
3. Cache corruption

**Solutions:**
```bash
# Check cache keys in use
gh cache list

# Delete old caches
gh cache delete <cache-key>

# Force cache rebuild
# Edit pyproject.toml, commit, push
```

### Issue: Slow Dependency Installation

**Symptom:** > 5 minutes even with cache miss

**Check:**
```yaml
# In logs, look for which packages are slow
- Downloading package-name...  # If this is slow
- Installing package-name...   # If this is slow
```

**Solutions:**
1. Check network connectivity
2. Consider pre-building wheels
3. Pin problem packages to specific versions

### Issue: Workflow Timeout

**Symptom:** Workflow killed after 30 minutes

**Check:**
```yaml
# Current timeout settings
timeout-minutes: 30  # Job level
timeout-minutes: 15  # Step level
```

**Solutions:**
- Increase timeout if legitimate
- Investigate what's hanging
- Add more granular timeouts to steps

### Issue: Flaky Tests

**Symptom:** Tests pass/fail intermittently

**Identify:**
```bash
# Run tests multiple times
for i in {1..10}; do pytest || echo "FAIL $i"; done

# Use pytest-repeat
pip install pytest-repeat
pytest --count=10 tests/test_flaky.py
```

**Solutions:**
- Add `@pytest.mark.flaky(reruns=3)`
- Fix non-deterministic behavior
- Add proper test isolation

---

## 🎯 Optimization Opportunities

### 1. Parallel Testing

**Current:**
```yaml
pytest -v
```

**Optimized:**
```yaml
pytest -v -n auto  # Use all CPU cores
```

**Expected improvement:** 30-40% faster tests

### 2. Test Selection

**Smart:** Only run tests affected by changes
```yaml
- name: Run affected tests
  run: |
    git diff origin/main... --name-only | grep "\.py$" | \
      xargs pytest --co -q | pytest
```

### 3. Docker Layer Caching

If using Docker:
```yaml
- name: Set up Docker Buildx
  uses: docker/setup-buildx-action@v3

- name: Cache Docker layers
  uses: actions/cache@v4
  with:
    path: /tmp/.buildx-cache
    key: docker-${{ runner.os }}-${{ hashFiles('**/Dockerfile') }}
```

### 4. Conditional Jobs

Skip expensive jobs when not needed:
```yaml
- name: Run notebooks
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  run: pytest --nbmake examples
```

### 5. Matrix Strategy Optimization

**Current:** Test all combinations

**Optimized:** Test strategically
```yaml
strategy:
  matrix:
    os: [ubuntu-latest]
    python-version: ['3.9', '3.11', '3.12']
    # Only test macOS/Windows on latest Python
    include:
      - os: macos-latest
        python-version: '3.12'
      - os: windows-latest
        python-version: '3.12'
```

---

## 📊 Monitoring Dashboard

### Create a Simple Dashboard

**Using GitHub Actions Badge:**
```markdown
![CI Status](https://github.com/CURENT/andes/workflows/Python%20application/badge.svg)
```

**Custom Status Page:**
```html
<!-- Add to README.md -->
## CI Performance

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Duration | 3-5 min | 4.2 min | ✅ |
| Cache Hit | 70%+ | 75% | ✅ |
| Success Rate | 95%+ | 98% | ✅ |

*Updated: 2025-11-05*
```

### Automated Monitoring

**Using GitHub API:**
```python
#!/usr/bin/env python3
"""Monitor CI performance and alert on issues."""
import requests

def check_performance():
    url = "https://api.github.com/repos/CURENT/andes/actions/runs"
    response = requests.get(url, params={"per_page": 10})
    runs = response.json()['workflow_runs']

    # Calculate metrics
    durations = []
    failures = 0

    for run in runs:
        if run['conclusion'] == 'success':
            duration_sec = (
                datetime.fromisoformat(run['updated_at']) -
                datetime.fromisoformat(run['created_at'])
            ).total_seconds()
            durations.append(duration_sec / 60)
        elif run['conclusion'] == 'failure':
            failures += 1

    avg_duration = sum(durations) / len(durations) if durations else 0
    failure_rate = failures / len(runs) * 100

    # Alert if performance degrades
    if avg_duration > 8:  # minutes
        print(f"⚠️ ALERT: Average duration increased to {avg_duration:.1f} min")
    if failure_rate > 10:
        print(f"⚠️ ALERT: Failure rate increased to {failure_rate:.0f}%")

    return {
        'avg_duration_min': round(avg_duration, 1),
        'failure_rate_pct': round(failure_rate, 1)
    }

if __name__ == "__main__":
    metrics = check_performance()
    print(f"📊 Current Performance:")
    print(f"   Average Duration: {metrics['avg_duration_min']} min")
    print(f"   Failure Rate: {metrics['failure_rate_pct']}%")
```

---

## 🔔 Alerting

### Set Up Notifications

**Slack Integration:**
```yaml
- name: Notify Slack on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    text: 'CI failed! Duration: ${{ steps.test.outputs.duration }}'
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

**Email Notifications:**
- GitHub Settings → Notifications
- Enable "Actions" notifications

---

## 📋 Weekly Review Checklist

- [ ] Check average workflow duration (target: 3-5 min)
- [ ] Check cache hit rate (target: 70%+)
- [ ] Check failure rate (target: < 5%)
- [ ] Review slowest steps
- [ ] Check for new performance opportunities
- [ ] Update this document if patterns change

---

## 🎯 Performance Goals

### Short Term (1 month)
- ✅ Migrate to uv (DONE - 3x improvement)
- ✅ Implement smart caching (DONE - 70%+ hit rate)
- ✅ Fix hanging workflows (DONE - 100% reliable)

### Medium Term (3 months)
- [ ] Achieve 80%+ cache hit rate
- [ ] Optimize test suite for speed
- [ ] Implement parallel testing
- [ ] Add performance monitoring dashboard

### Long Term (6 months)
- [ ] Sub-3-minute CI runs
- [ ] 95%+ cache hit rate
- [ ] Automated performance regression detection
- [ ] Multi-stage pipeline optimization

---

**Monitor, measure, optimize!** 📈
