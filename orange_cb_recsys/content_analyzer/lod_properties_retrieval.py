from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
from orange_cb_recsys.utils.string_cleaner import clean_with_unders, clean_no_unders


class LODPropertiesRetrieval(ABC):

    def __init__(self, mode: str = 'only_retrieved_evaluated'):
        self.__mode = self.__check_mode(mode)

    @staticmethod
    def __check_mode(mode):
        modalities = [
            'all',
            'all_retrieved',
            'only_retrieved_evaluated',
            'original_retrieved',
        ]
        if mode in modalities:
            return mode
        else:
            return 'all'

    def set_mode(self, mode):
        self.__mode = self.__check_mode(mode)

    def get_mode(self):
        return self.__mode

    @abstractmethod
    def get_properties(self, raw_content: Dict[str, object]) -> Dict[str, str]:
        pass


class DBPediaMappingTechnique(LODPropertiesRetrieval):
    """
    Class that creates a list of couples like this:
        <property: property value URI>
    """

    def __init__(self, entity_type: str, lang: str, label_field: str, additional_filters=None,
                 mode: str = 'only_retrieved_evaluated'):
        super().__init__(mode)

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
        if len(self.__additional_filters) > 0:
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
        else:
            return []

    def __mapping_query(self, raw_content):
        query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbo: <http://dbpedia.org/ontology/>  "
        query += "SELECT DISTINCT ?uri  "

        query += "WHERE { "

        # type matching
        query += "?uri rdf:type dbo:%s . " % self.__entity_type

        # label matching
        query += "?uri rdfs:label " + '?' + self.__label_field.lower()

        if (len(self.__additional_filters)) > 0:
            query += '. '

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
        query += "FILTER regex(?%s, \"%s\", \"i\"). " % (
            self.__label_field.lower(), clean_no_unders(raw_content[self.__label_field]))

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

    def __get_only_retrieved_evaluated(self, raw_content: Dict[str, object]) -> Dict[str, str]:
        new_property_labels = self.__get_properties_query()
        try:
            uri = self.__mapping_query(raw_content)
            result_dict = self.__retrieve_property_values(uri, new_property_labels)
        except ValueError:
            result_dict = {}
        return result_dict

    def __get_all_properties_retrieved(self, raw_content: Dict[str, object]) -> Dict[str, str]:
        new_property_labels = self.__get_properties_query()
        result_dict = self.__get_only_retrieved_evaluated(raw_content)
        properties = {}
        for property_label in new_property_labels:
            if property_label in result_dict.keys():
                properties[property_label] = result_dict[property_label]
            else:
                properties[property_label] = ""
        return properties

    def __get_original_retrieved(self, raw_content: Dict[str, object]) -> Dict[str, str]:
        original_property_labels = []
        original_properties = {}
        for key in raw_content.keys():
            original_property_labels.append(key)

        retrieved_properties = self.__get_only_retrieved_evaluated(raw_content)

        for property_label in original_property_labels:
            if property_label in retrieved_properties.keys():
                original_properties[property_label] = retrieved_properties[property_label]
            else:
                original_properties[property_label] = ""

        return original_properties

    def __get_all_properties(self, raw_content: Dict[str, object]) -> Dict[str, str]:
        all_prop_retrieved = self.__get_all_properties_retrieved(raw_content)
        property_labels = self.__get_properties_query()
        properties = {}
        for key in raw_content.keys():
            property_labels.append(key)

        for property_label in property_labels:
            if property_label in all_prop_retrieved.keys() and all_prop_retrieved[property_label] != '':
                properties[property_label] = all_prop_retrieved[property_label]
            elif property_label in raw_content.keys():
                properties[property_label] = raw_content[property_label]
            else:
                properties[property_label] = ""
        return properties

    def get_properties(self, raw_content: Dict[str, object]) -> Dict[str, str]:
        if self.get_mode() == 'only_retrieved_evaluated':
            return self.__get_only_retrieved_evaluated(raw_content)

        if self.get_mode() == 'all_retrieved':
            return self.__get_all_properties_retrieved(raw_content)

        if self.get_mode() == 'original_retrieved':
            return self.__get_original_retrieved(raw_content)

        if self.get_mode() == 'all':
            return self.__get_all_properties(raw_content)


raw_content = {"Title": "Jumanji", "Year": "1995", "Rated": "PG", "Released": "15 Dec 1995", "Runtime": "104 min",
               "Genre": "Adventure, Family, Fantasy", "Director": "Joe Johnston",
               "Writer": "Jonathan Hensleigh (screenplay by), Greg Taylor (screenplay by), Jim Strain (screenplay by), Greg Taylor (screen story by), Jim Strain (screen story by), Chris Van Allsburg (screen story by), Chris Van Allsburg (based on the book by)",
               "Actors": "Robin Williams, Jonathan Hyde, Kirsten Dunst, Bradley Pierce",
               "Plot": "After being trapped in a jungle board game for 26 years, a Man-Child wins his release from the game. But, no sooner has he arrived that he is forced to play again, and this time sets the creatures of the jungle loose on the city. Now it is up to him to stop them.",
               "Language": "English, French", "Country": "USA", "Awards": "4 wins & 9 nominations.",
               "Poster": "https://m.media-amazon.com/images/M/MV5BZTk2ZmUwYmEtNTcwZS00YmMyLWFkYjMtNTRmZDA3YWExMjc2XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SX300.jpg",
               "Ratings": [{"Source": "Internet Movie Database", "Value": "6.9/10"},
                           {"Source": "Rotten Tomatoes", "Value": "53%"}, {"Source": "Metacritic", "Value": "39/100"}],
               "Metascore": "39", "imdbRating": "6.9", "imdbVotes": "260,909", "imdbID": "tt0113497", "Type": "movie",
               "DVD": "25 Jan 2000", "BoxOffice": "N/A", "Production": "Sony Pictures Home Entertainment",
               "Website": "N/A", "Response": "True"}

mapp = DBPediaMappingTechnique('Film', 'EN', 'Title')
prop = mapp.get_properties(raw_content)
print(prop)

"""mapp.set_mode('all')
prop = mapp.get_properties(raw_content)
print(prop)

mapp.set_mode('all_retrieved')
prop = mapp.get_properties(raw_content)
print(prop)

mapp.set_mode('original_retrieved')
prop = mapp.get_properties(raw_content)
print(prop)"""