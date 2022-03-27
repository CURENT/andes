import re
import sys
import os
from collections import defaultdict

from setuptools import find_packages, setup

import versioneer

if sys.version_info < (3, 6):
    error = """
andes does not support Python <= {0}.{1}.
Python 3.6 and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(3, 6)
    sys.exit(error)

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    readme = readme_file.read()


def parse_requires(filename):
    with open(os.path.join(here, filename)) as requirements_file:
        reqs = [
            line for line in requirements_file.read().splitlines()
            if not line.startswith('#')
        ]
    return reqs


def get_extra_requires(filename, add_all=True):
    """
    Build ``extras_require`` from an invert requirements file.

    See:
    https://hanxiao.io/2019/11/07/A-Better-Practice-for-Managing-extras-require-Dependencies-in-Python/
    """

    with open(os.path.join(here, filename)) as fp:
        extra_deps = defaultdict(set)
        for k in fp:
            if k.strip() and not k.startswith('#'):
                tags = set()
                if '#' in k:
                    if k.count("#") > 1:
                        raise ValueError("Invalid line: {}".format(k))

                    k, v = k.split('#')
                    tags.update(vv.strip() for vv in v.split(','))

                tags.add(re.split('[<=>]', k)[0])
                for t in tags:
                    extra_deps[t].add(k)

        # add tag `all` at the end
        if add_all:
            extra_deps['all'] = set(vv for v in extra_deps.values() for vv in v)

    return extra_deps


extras_require = get_extra_requires("requirements-extra.txt")

# --- update `extras_conda` to include packages only available in PyPI ---
extras_require["interop"].add("pypowsybl")
extras_require["all"].add("pypowsybl")

setup(
    name='andes',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Python software for symbolic power system modeling and numerical analysis.",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Hantao Cui",
    author_email='cuihantao@gmail.com',
    url='https://github.com/cuihantao/andes',
    packages=find_packages(exclude=[]),
    entry_points={
        'console_scripts': [
            'andes = andes.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'andes': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
        ]
    },
    install_requires=parse_requires('requirements.txt'),
    extras_require=extras_require,
    license="GNU Public License v3",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Environment :: Console',
    ],
)
