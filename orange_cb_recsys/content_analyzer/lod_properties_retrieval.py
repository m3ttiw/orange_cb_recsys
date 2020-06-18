from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
from orange_cb_recsys.utils.string_cleaner import clean_with_unders, clean_no_unders


class LODPropertiesRetrieval(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def retrieve(self) -> Dict[str, str]:
        pass


class DBPediaMappingTechnique(LODPropertiesRetrieval):
    """
    Class that creates a list of couples like this:
        <property: property value URI>
    """

    def __init__(self, entity_type: str, lang: str, label_field: str, additional_filters=None):
        super().__init__()

        if additional_filters is None:
            additional_filters = {}

        self.__additional_filters = additional_filters
        self.__entity_type = entity_type
        self.__lang = lang
        self.__label_field = label_field

        self.__sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        self.__sparql.setReturnFormat(JSON)

        self.__has_label = self.__check_has_label()

    def set_label_field(self, label_field: str):
        self.__label_field = label_field

    def __check_has_label(self):
        query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbo: <http://dbpedia.org/ontology/>  "
        query += "SELECT DISTINCT "

        query += ', '.join("?%s" % field_name.lower() for field_name in self.__additional_filters)

        query += " WHERE { "

        query += '. '.join(
            ["?uri dbo:" + property_name + ' ?' + field_name.lower() + "_tmp" for field_name, property_name in
             self.__additional_filters.items()]) + '. '

        query += ' '.join(
            ["OPTIONAL { ?" + field_name.lower() + "_tmp" + " rdfs:label" ' ?' + field_name.lower() + " }" for
             field_name, property_name in
             self.__additional_filters.items()])

        query += " } LIMIT 1 OFFSET 0"

        self.__sparql.setQuery(query)
        results = self.__sparql.query().convert()

        result = results["results"]["bindings"][0]

        return result.keys()

    def __mapping_query(self, raw_content):
        query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbo: <http://dbpedia.org/ontology/>  "
        query += "SELECT DISTINCT ?uri  "

        query += "WHERE { "

        # type matching
        query += "?uri rdf:type dbo:%s . " % self.__entity_type

        # label matching
        query += "?uri rdfs:label " + '?' + self.__label_field.lower() + '. '

        # filter fields assignments
        query += '. '.join(["?uri dbo:%s ?%s. " % (property_name, field_name.lower()) +
                            "FILTER (" +
                            ' || '.join(["regex(?%s" % field_name.lower() +
                                         ("_label" if field_name.lower() in self.__has_label else '') +
                                         ', \"' + clean_no_unders(value) + '\", "i")'
                                         for value in (raw_content[field_name].split(', '))]) +
                            ")" for field_name, property_name in
                            self.__additional_filters.items()])

        if len(self.__has_label) != 0:
            query += '. '

        # label retrieval for fields with label
        query += '. '.join(
            ["?%s rdfs:label ?%s_label" % (field_name.lower(), field_name.lower())
             for field_name in self.__has_label])

        # lang filter
        query += ". FILTER langMatches(lang(?%s), \"%s\"). " % (self.__label_field.lower(), self.__lang)

        # label filter
        query += "FILTER regex(?%s, \"%s\", \"i\"). " % (self.__label_field.lower(), clean_no_unders(raw_content[self.__label_field]))

        query += " } "

        self.__sparql.setQuery(query)
        results = self.__sparql.query().convert()

        if len(results["results"]["bindings"]) == 0:
            raise ValueError("No mapping found")

        result = results["results"]["bindings"][0]
        uri = result["uri"]["value"]
        return uri

    def __get_properties_query(self):
        query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbo: <http://dbpedia.org/ontology/>  "
        query += "SELECT DISTINCT ?property_label WHERE { "
        query += "{ "
        query += "?property rdfs:domain ?class. "
        query += "dbo:%s rdfs:subClassOf+ ?class. " % self.__entity_type
        query += "} UNION {"
        query += "?property rdfs:domain dbo:%s" % self.__entity_type
        query += "} "
        query += "?property rdfs:label ?property_label. "
        query += "FILTER (langMatches(lang(?property_label), \"EN\")). }"

        self.__sparql.setQuery(query)
        results = self.__sparql.query().convert()

        if len(results["results"]["bindings"]) == 0:
            return None
        property_labels = [clean_with_unders(row["property_label"]["value"])
                           for row in results["results"]["bindings"]]

        return property_labels

    def __retrieve_property_values(self, uri, new_property_labels):
        if uri is None:
            return None
        query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
        query += "SELECT ?p ?o WHERE { <%s> ?p_tmp ?o. ?p_tmp rdfs:label ?p }" % uri

        self.__sparql.setQuery(query)
        results = self.__sparql.query().convert()

        result_dict = {}
        for row in results["results"]["bindings"]:
            property_label = clean_with_unders(row["p"]["value"])

            if property_label in new_property_labels:
                result_dict[property_label] = row["o"]["value"]

        return result_dict

    def produce_content(self, name, raw_content):
        new_property_labels = self.__get_properties_query()
        original_property_labels = []

        for key in raw_content.keys():
            original_property_labels.append(key)

        properties = {}

        try:
            uri = self.__mapping_query(raw_content)
            result_dict = self.__retrieve_property_values(uri, new_property_labels)
        except ValueError:
            result_dict = {}

        for property_label in new_property_labels:
            if property_label in result_dict.keys():
                properties[property_label] = result_dict[property_label]
            else:
                properties[property_label] = ""

        for property_label in original_property_labels:
            if property_label in raw_content.keys():
                properties[property_label] = raw_content[property_label]
            else:
                properties[property_label] = ""

        print(properties)