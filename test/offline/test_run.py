import json
from typing import Dict, List
from unittest import TestCase


class Test(TestCase):

    def test_dict_key(self,
                      test_dict: Dict,
                      keys: List[str],
                      msg: str = "You have to put the {} in the {}",
                      context: str = "dictionary"):
        for key in keys:
            self.assertIn(key, test_dict.keys(), msg.format(key, context))

    def test_config(self):
        config_list = json.load(open("src\offline\config.json"))
        with self.assertRaises(FileNotFoundError) as cm:
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
