from distutils.core import setup

setup(name='orange_cb_recsys',
      version='1.0',
      description='Python Framework for Content-Based Recommeder Systems',
      url='https://github.com/m3ttiw/Framework_CBRS_py',
      packages=['orange_cb_recsys', 'test', ],
      package_dir={'orange_cb_recsys': 'orange_cb_recsys', 'test': 'test', },
      )

"""
PER EVENTUALI DATASET
setup(...,
      packages=['mypkg'],
      package_dir={'mypkg': 'orange_cb_recsys/mypkg'},
      package_data={'mypkg': ['data/*.dat']},
      )
"""
