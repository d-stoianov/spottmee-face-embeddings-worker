from pydantic import BaseModel
import numpy as np
from typing import List

Embedding = List[float]

class FaceEmbedding(BaseModel):
    name: str
    embedding: Embedding
    
    class Config:
        arbitrary_types_allowed = True # needed for numpy arrays

    def to_numpy_array(self) -> np.ndarray:
        return np.array(self.embedding, dtype=np.float32)

    @classmethod
    def from_numpy_array(cls, name: str, np_array: np.ndarray) -> 'FaceEmbedding':
        return cls(name=name, embedding=np_array.tolist())