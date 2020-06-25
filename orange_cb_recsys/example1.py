# CONTENT ANALYZER
from orange_cb_recsys.content_analyzer import ContentAnalyzer, ContentAnalyzerConfig
from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import DATFile
from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import DBPediaMappingTechnique, \
    PropertiesFromDataset

"""movies_filename = os.path.join(home_path, '/Documents/ml-1m/movies.dat')
user_filename = os.path.join(home_path, '/Documents/ml-1m/users.dat')
ratings_filename = os.path.join(home_path, '/Documents/ml-1m/ratings.dat')"""

movies_filename = '/home/Mattia/Documents/ml-1m/movies.dat'
user_filename = '/home/Mattia/Documents/ml-1m/users.dat'
ratings_filename = '/home/Mattia/Documents/ml-1m/ratings.dat'

output_dir = '../../contents/test_1m_'

movies_ca_config = ContentAnalyzerConfig(
    content_type='Item',
    source=DATFile(movies_filename),
    id_field_name_list=['0'],
    output_directory=output_dir
)

movies_ca_config.append_exogenous_properties_retrieval(
    DBPediaMappingTechnique(
        entity_type='Film',
        lang='EN',
        label_field='1'
    )
)

content_analyzer = ContentAnalyzer(movies_ca_config).fit()

users_ca_config = ContentAnalyzerConfig(
    content_type='User',
    source=DATFile(user_filename),
    id_field_name_list=['0'],
    output_directory=output_dir
)

users_ca_config.append_exogenous_properties_retrieval(
    PropertiesFromDataset()
)

content_analyzer.set_config(users_ca_config).fit()

ratings_import = RatingsImporter(
    source=DATFile(ratings_filename),
    rating_configs=[RatingsFieldConfig(field_name='2', processor=NumberNormalizer(min_=1, max_=5))],
    from_field_name='0',
    to_field_name='1',
    timestamp_field_name='3'
).import_ratings()

"""
# PLOT GRAPH
from orange_cb_recsys.recsys.graphs.tripartite_graphs import NXTripartiteGraph
import networkx as nx
import matplotlib.pyplot as plt

g = NXTripartiteGraph(source_frame=ratings_import, contents_dir=output_dir)

G = g.graph

pos = nx.bipartite_layout(G, g.get_from_nodes(), align='horizontal')
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 500)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, arrows=True)

colors=range(20)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
nx.draw(G, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.Blues)
plt.show()
"""