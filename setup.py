from distutils.core import setup

setup(name='Framework_CBRS_py',
      version='1.0',
      description='Python Framework for Content-Based Recommeder Systems',
      url='https://github.com/m3ttiw/Framework_CBRS_py',
      packages=['src', 'offline', 'online', 'test', ],
      package_dir={'src': 'src', 'offline': 'src/offline', 'online': 'src/online', 'test': 'test', },
      )

"""
PER EVENTUALI DATASET
setup(...,
      packages=['mypkg'],
      package_dir={'mypkg': 'src/mypkg'},
      package_data={'mypkg': ['data/*.dat']},
      )
"""
