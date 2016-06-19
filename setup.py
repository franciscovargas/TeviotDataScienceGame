from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path
from pip.req import parse_requirements

here = path.abspath(path.dirname(__file__))
install_reqs = parse_requirements(here + '/requirements.txt',session=True)
reqs = [str(ir.req) for ir in install_reqs]
# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='teviot',

    version='1.0',

    description='dsg 2k16 UoE entry imagenet like challenge repo',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/franciscovargas/TeviotDataScienceGame.git',

    author='Francisco Vargas',
    author_email='vargfran@gmail.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers, Users',
        'Topic :: Data Science Game, UoE entry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',

    ],

    # What does your project relate to?
    keywords='datascience machine learning probability imagenet convolutional',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=[ 'test']),

    # List run-time dependencies here.  These will be installed by pip when your
    # project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/technical.html#install-requires-vs-requirements-files
    install_requires=reqs,


    extras_require = {
        'dev': ['check-manifest']
    },

)
