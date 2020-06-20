from orange_cb_recsys.content_analyzer.content_representation.content_field import FeaturesBagField
from orange_cb_recsys.content_analyzer.field_content_production_techniques import SingleContentTechnique
from orange_cb_recsys.utils.check_tokenization import check_not_tokenized
from pywsd import disambiguate
from collections import Counter


class SynsetDocumentFrequency(SingleContentTechnique):
    def produce_content(self, field_representation_name: str, field_data) -> FeaturesBagField:
        """
        Produces a bag of features whose key is a wordnet synset
        and whose value is the frequency of the synset in the
        field data text
        """

        field_data = check_not_tokenized(field_data)

        synsets = disambiguate(field_data)
        synsets = [synset for word, synset in synsets if synset is not None]

        fb = FeaturesBagField(field_representation_name, Counter(synsets))
        print(fb)
        return fb
