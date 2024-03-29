from setuptools import setup, find_packages
from codecs import open
import os
import glob

__author__ = "Nongnuch Artrith, Alexander Urban"
__email__ = "nartrith@atomistic.net, aurban@atomistic.net"

here = os.path.abspath(os.path.dirname(__file__))
package_name = 'howru'
package_description = 'Optimize Hubbard U parameters for DFT+U'

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as fp:
    long_description = fp.read()

# Get version number from the VERSION file
with open(os.path.join(here, package_name, 'VERSION')) as fp:
    version = fp.read().strip()

setup(
    name=package_name,
    version=version,
    description=package_description,
    long_description=long_description,
    url='howru.atomistic.net',
    author=__author__,
    author_email=__email__,
    license='MPL2',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
    keywords=['percolation', 'materials science', 'monte carlo'],
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy', 'scipy', 'pymatgen'],
    scripts=glob.glob(os.path.join("scripts", "*.py"))
)
