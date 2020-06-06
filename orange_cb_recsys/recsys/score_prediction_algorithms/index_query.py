import os

from orange_cb_recsys.content_analyzer.memory_interfaces import IndexInterface
from orange_cb_recsys.recsys.score_prediction_algorithms.score_prediction_algorithm import ScorePredictionAlgorithm

import pandas as pd

from orange_cb_recsys.utils.const import DEVELOPING, home_path
from orange_cb_recsys.utils.load_content import load_content_instance

from java.nio.file import Paths

from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause, BoostQuery
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.search.similarities import ClassicSimilarity
from org.apache.lucene.analysis.en import EnglishAnalyzer


class IndexQuery(ScorePredictionAlgorithm):
    def __recs_query(self, positive_rated_document_list, scores, recs_number, items_directory):
        BooleanQuery.setMaxClauseCount(2000000)
        searcher = IndexSearcher(DirectoryReader.open(SimpleFSDirectory(Paths.get(items_directory))))
        searcher.setSimilarity(ClassicSimilarity())

        field_list = searcher.doc(positive_rated_document_list[0]).getFields()
        user_fields = {}
        field_parsers = {}
        analyzer = EnglishAnalyzer()
        for field in field_list:
            if field.name() == 'content_id':
                continue
            user_fields[field.name()] = field.stringValue()
            field_parsers[field.name()] = QueryParser(field.name(), analyzer)

        positive_rated_document_list.remove(positive_rated_document_list[0])

        for _ in positive_rated_document_list:
            for field in field_list:
                if field.name() == 'content_id':
                    continue
                user_fields[field.name()] += field.stringValue()

        query_builder = BooleanQuery.Builder()
        for score in scores:
            for field_name in user_fields.keys():
                if field_name == 'content_id':
                    continue
                field_parsers[field_name].setDefaultOperator(QueryParser.Operator.OR)

                field_query = field_parsers[field_name].escape(user_fields[field_name])
                field_query = field_parsers[field_name].parse(field_query)
                field_query = BoostQuery(field_query, score)
                query_builder.add(field_query, BooleanClause.Occur.SHOULD)

        query = query_builder.build()
        docs_to_search = len(positive_rated_document_list) + recs_number
        scoreDocs = searcher.search(query, docs_to_search).scoreDocs

        recorded_items = 0
        columns = ['item_id', 'rating']
        score_frame = pd.DataFrame(columns=columns)
        for scoreDoc in scoreDocs:
            if recorded_items >= recs_number:
                break
            if scoreDoc.doc not in positive_rated_document_list:
                doc = searcher.doc(scoreDoc.doc)
                item_id = doc.getField("content_id").stringValue()
                recorded_items += 1

                score_frame = pd.concat([score_frame, pd.DataFrame.from_records([(item_id, scoreDoc.score)], columns=columns)])

        return score_frame

    def predict(self, ratings: pd.DataFrame, items_directory: str):
        index_path = os.path.join(items_directory, 'search_index')
        if not DEVELOPING:
            index_path = os.path.join(home_path, items_directory, 'search_index')

        scores = []
        rated_document_list = []
        for item_id, score in zip(ratings.to_id, ratings.score):
            item = load_content_instance(items_directory, item_id)

            if score > 0:
                rated_document_list.append(item.get_index_document_id())
                scores.append(score)

        return self.__recs_query(rated_document_list,
                                 scores,
                                 len([filename for filename in os.listdir(items_directory) if filename != 'search_index']),
                                 index_path)