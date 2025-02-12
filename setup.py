"""
Setuptools based setup module
"""
from setuptools import setup, find_packages
import versioneer

setup(
    name='pyiron-gpl',
    version=versioneer.get_version(),
    description='pyiron - an integrated development environment (IDE) for computational materials science.',
    long_description='http://pyiron.org',

    url='https://github.com/pyiron/pyiron_gpl',
    author='Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department',
    author_email='janssen@mpie.de',
    license='GPLv3',

    classifiers=['Development Status :: 5 - Production/Stable',
                 'Topic :: Scientific/Engineering :: Physics',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10',
                 'Programming Language :: Python :: 3.11',
    ],

    keywords='pyiron',
    packages=find_packages(exclude=["*tests*", "*docs*", "*binder*", "*conda*", "*notebooks*", "*.ci_support*"]),
    install_requires=[
        'numpy==1.26.4',
        'pyiron_atomistics==0.6.22',
        'pyiron_snippets==0.1.4',
        'qc-iodata==v1.0.0a5'
    ],
    cmdclass=versioneer.get_cmdclass(),

    )
