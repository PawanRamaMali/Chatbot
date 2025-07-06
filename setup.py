#!/usr/bin/env python3
"""
Setup script for Neural Chatbot
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Get version from __init__.py
def get_version():
    init_file = here / 'src' / 'neural_chatbot' / '__init__.py'
    with open(init_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '0.1.0'

setup(
    name='neural-chatbot',
    version=get_version(),
    description='An intelligent AI chatbot using neural networks and natural language processing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/PawanRamaMali/neural-chatbot',
    author='Your Name',
    author_email='your.email@example.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Communications :: Chat',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='chatbot, ai, neural network, nlp, tensorflow, machine learning',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.8, <4',
    install_requires=[
        'numpy>=1.21.0',
        'tensorflow>=2.8.0,<3.0.0',
        'nltk>=3.7',
        'scikit-learn>=1.0.0',
        'pandas>=1.3.0',
        'flask>=2.0.0',
        'flask-cors>=3.0.10',
        'pyyaml>=6.0',
        'click>=8.0.0',
        'colorlog>=6.0.0',
        'python-dotenv>=0.19.0',
        'marshmallow>=3.14.0',
        'gunicorn>=20.1.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-mock>=3.8.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=0.991',
            'pre-commit>=2.20.0',
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
        'plot': [
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'plotly>=5.0.0',
        ],
        'production': [
            'redis>=4.5.0',
            'celery>=5.2.0',
            'prometheus-client>=0.15.0',
            'sentry-sdk[flask]>=1.15.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'neural-chatbot=neural_chatbot.main:main',
            'neural-chatbot-train=neural_chatbot.core.trainer:main',
            'neural-chatbot-serve=neural_chatbot.api.app:main',
        ],
    },
    include_package_data=True,
    package_data={
        'neural_chatbot': [
            'data/*.json',
            'config/*.yaml',
            'web/templates/*.html',
            'web/static/css/*.css',
            'web/static/js/*.js',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/PawanRamaMali/neural-chatbot/issues',
        'Source': 'https://github.com/PawanRamaMali/neural-chatbot',
        'Documentation': 'https://neural-chatbot.readthedocs.io/',
    },
)