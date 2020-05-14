import numpy as np
from src.offline.content_analyzer.field_content_production_technique import CombiningTechnique


class Centroid(CombiningTechnique):
    """"
    Class that implements the Abstract Class CombiningTechnique,
    this class implements the centroid vector of a matrix.
    """
    def combine(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """"
        Calculates the centroid of the input matrix

        Args:
            embedding_matrix (np.ndarray): np bi-dimensional array whose centroid will be calculated

        Returns:
            np.ndarray: centroid vector of the input matrix
        """
        return np.average(embedding_matrix, axis=0)

# your combining technique
