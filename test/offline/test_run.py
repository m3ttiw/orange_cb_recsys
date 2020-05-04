import json
from unittest import TestCase


class Test(TestCase):
    def test_config(self):
        config_list = json.load(open("src\offline\config.json"))
        with self.assertRaises(FileNotFoundError) as cm:
            self.assert_(True, "Try to use double back_slashes '\\' instead of a single slash")
        self.assertEqual(type(config_list), type(list()), "the config must contain a list of dict")
        for content_config in config_list:
            self.assertEqual(type(content_config), type(dict()),
                             "the config must contain a list of dict")
            self.assertIn("content_type", content_config.keys(),
                          "You have to put the 'content_type' in the content config")
            self.assertIn("raw_source_path", content_config.keys(),
                          "You have to put the 'raw_source_path' in the content config")
            self.assertIn("source_type", content_config.keys(),
                          "You have to put the 'source_type' in the content config")
            self.assertIn("id_field_name", content_config.keys(),
                          "You have to put the 'id_field_name' in the content config")
            self.assertIn("fields", content_config.keys(),
                          "You have to put the 'fields' in the content config")
            for field in content_config['fields']:
                if field is not None:
                    self.assertIn("field_name", field.keys(),
                                  "You have to put the 'field_name' in the field config")
                    self.assertIn("memory_interface", field.keys(),
                                  "You have to put the 'memory_interface' in the field config")
                    self.assertIn("pipeline_list", field.keys(),
                                  "You have to put the 'pipeline_list' in the field config")
                    for pipeline in field["pipeline_list"]:
                        if pipeline is not None:
                            self.assertIn("field_content_production", pipeline.keys(),
                                          "You have to put the 'field_content_production' in the pipeline config")
                            self.assertIn("class", pipeline["field_content_production"].keys(),
                                          "You have to put the 'class' in the field_content_production config")
                            self.assertIn("preprocessing_list", pipeline.keys(),
                                          "You have to put the 'preprocessing_list' in the pipeline config")
                            for preprocessing in pipeline["preprocessing_list"]:
                                pass

    def test_run(self):
        self.skipTest("test in the submodules.")
