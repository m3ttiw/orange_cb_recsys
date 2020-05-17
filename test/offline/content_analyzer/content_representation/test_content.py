import pickle
from unittest import TestCase

from src.offline.content_analyzer.content_representation.content import Content
from src.offline.content_analyzer.content_representation.content_field import FeaturesBagField, ContentField


class TestContent(TestCase):
    def test_load_serialize(self):
        content_field_repr = FeaturesBagField("test")
        content_field_repr.append_feature("test_key", "test_value")
        content_field = ContentField("test_field", "0000")
        content_field.append(content_field_repr)
        content = Content("001")
        content.append(content_field)
        try:
            content.serialize(".")
        except:
            pass

        with open('001.bin', 'rb') as file:
            self.assertEqual(content, pickle.load(file))

    def test_append_remove(self):
        content_field_repr = FeaturesBagField("test")
        content_field_repr.append_feature("test_key", "test_value")
        content_field = ContentField("test_field", "0000")
        content_field.append(content_field_repr)
        content1 = Content("001")
        content1.append(content_field)

        content2 = Content("002")
        content2.append(content_field)
        content_field_repr = FeaturesBagField("test")
        content_field_repr.append_feature("test_key", "test_value")
        content_field2 = ContentField("test_field2", "0000")
        content_field2.append(content_field_repr)
        content2.append(content_field2)
        content2.remove("test_field2")
        self.assertTrue(content1.get_field_list(), content2.get_field_list())
