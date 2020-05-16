import json
from typing import Dict, List
from unittest import TestCase

# from src.offline.run import check_for_available
from offline.run import config_run, check_for_available

config_dict = {
    "content_type": "ITEM",
    "raw_source_path": "movies_info.json",
    "source_type": "json",
    "id_field_name": "imdbID",
    "start_from": "0",
    "end_up_at": "all",
    "fields": [
        {
            "field_name": "Title",
            "memory_interface": "None",
            "memory_interface_path": "./test-index-plot",
            "pipeline_list": [
                {
                    "field_content_production": {"class": "babelpy"},
                    "preprocessing_list": []
                }
            ]
        }
    ]
}


class Test(TestCase):
    """
    def test_dict_key(self,
                      test_dict: Dict,
                      keys: List[str],
                      msg: str = "You have to put the {} in the {}",
                      context: str = "dictionary"):
        for key in keys:
            self.assertIn(key, test_dict.keys(), msg.format(key, context))
    """

    def test_config(self):
        self.skipTest("FIX TEST")
        # test only if the key in the config.json are valid
        config_list = json.load(open("config.json"))
        msg: str = "You have to put the {} in the {}"
        print(type(config_list))
        self.assertEqual(type(config_list), type(list()), "the config must contain a list of dict")
        for content_config in config_list:
            self.assertEqual(type(content_config), type(dict()),
                             "the config must contain a list of dict")
            """
            self.test_dict_key(content_config, ["content_type", "raw_source_path", "source_type", "id_field_name",
                                                "fields"], context=)
            """
            for key in ["content_type", "raw_source_path", "source_type", "id_field_name", "fields"]:
                self.assertIn(key, content_config.keys(), msg.format(key, "content config"))
            for field in content_config['fields']:
                if field is not None:
                    """
                    self.test_dict_key(field,
                                       ["field_name", "memory_interface", "pipeline_list"], context="field config")
                    """
                    for key in ["field_name", "memory_interface", "pipeline_list"]:
                        self.assertIn(key, field.keys(), msg.format(key, "field config"))
                    for pipeline in field["pipeline_list"]:
                        if pipeline is not None:
                            """
                            self.test_dict_key(pipeline,
                                               ["field_content_production", "preprocessing_list"],
                                               context="pipeline config")
                            self.test_dict_key(pipeline["field_content_production"],
                                               ["class"], context="field_content_production config")
                            """
                            for key in ["field_content_production", "preprocessing_list"]:
                                self.assertIn(key, pipeline.keys(), msg.format(key, "pipeline config"))
                            self.assertIn("class", pipeline["field_content_production"].keys(),
                                          msg.format(key, "field_content_production config"))
                            for preprocessing in pipeline["preprocessing_list"]:
                                """
                                self.test_dict_key(preprocessing, ["class"], context="preprocessing config")
                                """
                                self.assertIn("class", preprocessing.keys(), msg.format(key, "preprocessing config"))

    def test_check_for_available(self):
        global config_dict
        self.assertEqual(check_for_available([{}]), False)
        self.assertEqual(check_for_available(config_dict), True)

    def test_run(self):
        #self.skipTest("test in the submodules.")
        global config_dict
        self.assertEqual(config_run(config_dict), 0, "The configuration should run without problems ok")

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
