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
        content_not_serialized = content
        content.append(content_field)
        try:
            content.serialize("test_dir")
            content.load("test_dir")
        except:
            pass
        self.assertEqual(content_not_serialized, content)

    def test_append_remove(self):
        content_field_repr = FeaturesBagField("test")
        content_field_repr.append_feature("test_key", "test_value")
        content_field = ContentField("test_field", "0000")
        content_field.append(content_field_repr)
        content = Content("001")
        content.append(content_field)
        self.assertEqual(content.remove("test_field"), content_field)
