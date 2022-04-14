from sentence_transformers import SentenceTransformer, util
from packaging import version
from platform import python_version
import torch

model = SentenceTransformer('pritamdeka/S-Biomed-Roberta-snli-multinli-stsb')

sent = model.encode(['Hi how are you!', "Im doing fine :P"])

print('hi')