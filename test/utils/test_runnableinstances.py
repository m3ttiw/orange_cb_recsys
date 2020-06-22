from unittest import TestCase
from orange_cb_recsys.utils.runnableinstances import *


class Test(TestCase):
    def test_runnable_instances(self):
        show()

        get()

        add('test', 'ciao')

        remove('test')

        show()
