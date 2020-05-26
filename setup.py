from setuptools import setup


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='orange_cb_recsys',
      version='1.1',
      install_requires=requirements,
      description='Python Framework for Content-Based Recommeder Systems',
      url='https://github.com/m3ttiw/orange_cb_recsys',
      download_url='https://github.com/m3ttiw/orange_cb_recsys/archive/1.0.tar.gz',
      packages=['orange_cb_recsys', 'test', ],
      package_dir={'orange_cb_recsys': 'orange_cb_recsys', 'test': 'test', },
      )
