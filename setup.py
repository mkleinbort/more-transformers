from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='more-transformers',
    version='0.0.11',
    description='More transformers for scikit-learn pipelines',  # Optional
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mkleinbort/more-transformers',
    author='Mycchaka Kleinbort',
    author_email='mkleinbort@gmail.com',
    packages=['more_transformers'],
    include_package_data=True,
    platforms='any',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    keywords='sklearn pipeline ml ai pandas',
    python_requires='>=3.6',
    
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
    ],

    project_urls={ 
        'Bug Reports': 'https://github.com/mkleinbort/more-transformers/issues',
        'Source': 'https://github.com/mkleinbort/more-transformers/',
    },
)
