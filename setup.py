from setuptools import setup


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='orange_cb_recsys',
      version='1.1.2.1',
      author='Roberto Barile, Francesco Benedetti, Carlo Parisi, Mattia Patruno',
      install_requires=[
            'PyYAML==5.3.1',
            'numpy==1.18.4',
            'gensim==3.8.3',
            'nltk==3.5',
            'babelpy==1.0.1',
            'mysql==0.0.2',
            'mysql-connector-python==8.0.20',
            'wikipedia2vec==1.0.4'],
      description='Python Framework for Content-Based Recommeder Systems',
      url='https://github.com/m3ttiw/orange_cb_recsys',
      download_url='https://github.com/m3ttiw/orange_cb_recsys/archive/1.1.2.tar.gz',
      packages=['orange_cb_recsys', 'test', ],
      package_dir={'orange_cb_recsys': 'orange_cb_recsys', 'test': 'test', },
      )
