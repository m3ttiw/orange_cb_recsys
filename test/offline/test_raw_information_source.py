from unittest import TestCase

from offline.raw_data_extractor.raw_information_source import SQLDatabase


class TestSQLDatabase(TestCase):

    def test_iter(self):
        sql = SQLDatabase('localhost', 'root', 'password', 'prova', 'tabella')
        my_iter = iter(sql)
        d1 = {'campo1': 'Francesco', 'campo2': 'Benedetti', 'campo3': 'Polignano'}
        d2 = {'campo1': 'Mario', 'campo2': 'Rossi', 'campo3': 'Roma'}
        d3 = {'campo1': 'Gigio', 'campo2': 'Donnarumma', 'campo3': 'Milano'}
        self.assertDictEqual(next(my_iter), d1)
        self.assertDictEqual(next(my_iter), d2)
        self.assertDictEqual(next(my_iter), d3)
