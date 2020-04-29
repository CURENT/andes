from os import path
from setuptools import setup, find_packages
import sys
import versioneer

# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
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

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [
        line for line in requirements_file.read().splitlines()
        if not line.startswith('#')
    ]

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
    install_requires=requirements,
    license="GNU Public License v3",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Environment :: Console',
    ],
)
