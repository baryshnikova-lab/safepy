"""
## Development

### release new version

    git commit -am "version bump";git push origin master
    python setup.py --version
    git tag -a v$(python setup.py --version) -m "upgrage";git push --tags

"""

import sys
if (sys.version_info[0]) != (3):
     raise RuntimeError('Python 3 required ')

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

## dependencies/requirements
requirements = {    
'base':  """ipykernel>=5.1.0
            networkx>=2.2
            pandas>=0.23.4
            statsmodels>=0.9.0
            matplotlib>=2.0.1
            scipy>=1.0.0
            tqdm>=4.11.2
            numpy>=1.14.1
            xlrd >=1.1.0""".split('\n'),
}
extras_require={k:l for k,l in requirements.items() if not k=='base'}
## all: extra except dev
extras_require['all']=[l for k,l in extras_require.items() if not k=='dev']
### flatten
extras_require['all']=[s for l in extras_require['all'] for s in l]
### unique
extras_require['all']=list(set(extras_require['all']))

setuptools.setup(
    name='safepy',
    version='0.0.1',
    description='',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    author='',
    author_email='',
    license='General Public License v. 3',
    packages=setuptools.find_packages('.',exclude=['test', 'unit','deps', 'data']),
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements['base'],
    extras_require=extras_require,
    entry_points={
    'console_scripts': ['safepy = safepy.run:parser.dispatch',],
    },    
    python_requires='>=3.7, <4',
)
