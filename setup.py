"""
Minimal setup.py shim for backwards compatibility.

Modern Python packaging now uses pyproject.toml (PEP 517/518/621).
All configuration is defined in pyproject.toml.

This file is kept for:
1. Backwards compatibility with older pip versions
2. Editable installs (pip install -e .)
3. Versioneer integration

For new installations, pyproject.toml is the source of truth.
"""

import sys
import os

# Enforce minimum Python version
if sys.version_info < (3, 9):
    error = """
ANDES requires Python 3.9 or later.

Python 3.8 and earlier are no longer supported as they have reached
end-of-life. Please upgrade your Python installation.

Current Python version: {}.{}
Required: Python >= 3.9

Check your Python version:
    python3 --version

To upgrade pip (if needed):
    pip install --upgrade pip
""".format(sys.version_info.major, sys.version_info.minor)
    sys.exit(error)

# Import setuptools
from setuptools import setup

# Try to import versioneer for dynamic versioning
# If versioneer is not available (e.g., in isolated build), use fallback
try:
    # Add current directory to path to find versioneer.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import versioneer
    version = versioneer.get_version()
    cmdclass = versioneer.get_cmdclass()
except (ImportError, ModuleNotFoundError):
    # Fallback: read version from _version.py if it exists
    version_file = os.path.join(os.path.dirname(__file__), 'andes', '_version.py')
    if os.path.exists(version_file):
        with open(version_file) as f:
            for line in f:
                if line.startswith('__version__'):
                    version = line.split('=')[1].strip().strip('"').strip("'")
                    break
            else:
                version = "0.0.0+unknown"
    else:
        version = "0.0.0+unknown"
    cmdclass = {}

# Call setup with versioneer integration
# All other configuration is in pyproject.toml
setup(
    version=version,
    cmdclass=cmdclass,
)
