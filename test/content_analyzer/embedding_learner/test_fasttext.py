from unittest import TestCase
from src.content_analyzer.information_processor.nlp import NLTK
from src.content_analyzer.embedding_learner.fasttext import GensimFastText
from src.content_analyzer.memory_interfaces.memory_interfaces import TextInterface
from src.content_analyzer.raw_information_source import JSONFile

class TestGensimFastText(TestCase):
    def test_start_learning(self):
        list = [[[-0.01764452, 0.09988707, -0.07479963, 0.18985124]],
                [[-0.23016433, -0.15074177, -0.1580534, 0.1000268]],
                [[0.07058926, 0.10564964, -0.20231633, -0.17358421]],
                [[0.24803016, 0.06857626, -0.09041961, -0.06555818]],
                [[0.14719704, -0.1785951, -0.17852865, -0.1936684]],
                [[-0.21094991, -0.08666641, 0.07959844, 0.02585224]],
                [[0.2021303, 0.20597672, 0.06250537, -0.11165955]],
                [[0.20791063, -0.13759944, -0.15303749, 0.14919399]],
                [[-0.20543897, 0.05149202, 0.0828627, -0.07295058]],
                [[-0.22641116, -0.17118621, 0.05215447, -0.08410117]],
                [[0.02842869, 0.02290572, 0.03643247, 0.06447929]],
                [[-0.04531404, 0.01236728, -0.00496479, 0.11727818]],
                [[0.01859922, -0.06659139, -0.10700723, 0.08015627]],
                [[-0.1578243, 0.06482039, -0.00113674, -0.1465923]],
                [[-0.05484185, -0.15408944, 0.01424197, -0.11396042]],
                [[-0.07760792, -0.06334326, -0.0091289, 0.13192533]],
                [[-0.08769695, 0.05628382, -0.08081185, 0.12275832]],
                [[0.12177688, -0.06153355, -0.09278309, -0.00073456]],
                [[0.01448179, -0.03265472, -0.03582956, -0.08525026]],
                [[0.04067247, 0.00251827, -0.03531238, -0.04543722]]]
        result = GensimFastText(source=JSONFile("movies_info_reduced.json"),
                                preprocessor=NLTK(),
                                loader= TextInterface,
                                field_name="Genre").start_learning()
        for i, res in enumerate(result):
            self.assertEqual(list[i], res, "Fail in Doc {} - Vector = {}".format(str(i), res))
