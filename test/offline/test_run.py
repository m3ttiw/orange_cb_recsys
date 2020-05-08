import json
from typing import Dict, List
from unittest import TestCase

#from src.offline.run import check_for_available


class Test(TestCase):

    def test_dict_key(self,
                      test_dict: Dict,
                      keys: List[str],
                      msg: str = "You have to put the {} in the {}",
                      context: str = "dictionary"):
        for key in keys:
            self.assertIn(key, test_dict.keys(), msg.format(key, context))

    def test_config(self):
        self.skipTest("FIX TEST")
        # test only if the key in the config.json are valid
        config_list = json.load(open("src\offline\config.json"))
        with self.assertRaises(FileNotFoundError):
            self.assert_(True, "Try to use double back_slashes '\\' instead of a single slash")
        self.assertEqual(type(config_list), type(list()), "the config must contain a list of dict")
        for content_config in config_list:
            self.assertEqual(type(content_config), type(dict()),
                             "the config must contain a list of dict")
            self.test_dict_key(content_config, ["content_type", "raw_source_path", "source_type", "id_field_name",
                                                "fields"], context="content config")
            for field in content_config['fields']:
                if field is not None:
                    self.test_dict_key(field,
                                       ["field_name", "memory_interface", "pipeline_list"], context="field config")
                    for pipeline in field["pipeline_list"]:
                        if pipeline is not None:
                            self.test_dict_key(field,
                                               ["field_content_production", "preprocessing_list"],
                                               context="pipeline config")
                            self.test_dict_key(field["field_content_production"],
                                               ["class"], context="field_content_production config")
                            for preprocessing in pipeline["preprocessing_list"]:
                                self.test_dict_key(preprocessing, ["class"], context="preprocessing config")

    def test_run(self):
        self.skipTest("test in the submodules.")

    def test_check_for_available(self):
        self.skipTest("FIX TEST")
        in_dict = [{"source_type": "text"}]
        self.assertFalse(check_for_available(in_dict))
        in_dict = [{"source_type": "json", "fields": [{"memory_interface": "not-index"}]}]
        self.assertFalse(check_for_available(in_dict))
        in_dict = [{"source_type": "json", "fields": [{"memory_interface": "index", "pipeline_list": [{
                        "field_content_production": {"class": "no-class"}}]}]}]
        self.assertFalse(check_for_available(in_dict))
        in_dict = [{"source_type": "json", "fields": [{"memory_interface": "index", "pipeline_list": [{
            "field_content_production": {"class": "babelpy"}, "preprocessing_list": [{"class": "no-class"}]}]}]}]
        self.assertFalse(check_for_available(in_dict))
        in_dict = [{"source_type": "json", "fields": [{"memory_interface": "index", "pipeline_list": [{
            "field_content_production": {"class": "babelpy"}, "preprocessing_list": [{"class": "open_nlp"}]}]}]}]
        self.assertTrue(check_for_available(in_dict))
