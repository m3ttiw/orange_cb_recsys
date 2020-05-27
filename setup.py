from setuptools import setup


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='orange_cb_recsys',
      version='0.1',
      author='Roberto Barile, Francesco Benedetti, Carlo Parisi, Mattia Patruno',
      install_requires=[
            'pandas'
            'PyYAML==5.3.1',
            'numpy==1.18.4',
            'gensim==3.8.3',
            'nltk==3.5',
            'babelpy==1.0.1',
            'mysql==0.0.2',
            'mysql-connector-python==8.0.20',
            'wikipedia2vec==1.0.4'],
      description='Python Framework for Content-Based Recommeder Systems',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/m3ttiw/orange_cb_recsys',
      packages=['orange_cb_recsys', 'test', ],
      package_dir={'orange_cb_recsys': 'orange_cb_recsys', 'test': 'test', },
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
            "Operating System :: OS Independent",
      ],
      python_requires='>=3.8',
      )
