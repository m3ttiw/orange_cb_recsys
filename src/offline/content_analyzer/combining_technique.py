import numpy as np
from src.offline.content_analyzer.field_content_production_technique import CombiningTechnique


class Centroid(CombiningTechnique):
    """"
    Class that implements the Abstract Class CombiningTechnique,
    this class implements the centroid vector of a matrix.
    """
    def combine(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """"
        Implements the Abstract Method combine in Combining Technique,
        calculate centroid of the input matrix

        Args:
            embedding_matrix (np.ndarray): np bi-dimensional array of which calculate the centroid

        Returns:
            np.ndarray: centroid vector of input matrix
        """
        return np.average(embedding_matrix, axis=0)

    def __str__(self):
        return "Centroid"

    def __repr__(self):
        return "< Centroid >"

# your combining technique
