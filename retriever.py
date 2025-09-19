from timetrack import TimingContext 
from transformers import BitsAndBytesConfig
from sentence_transformers.models import Transformer, Pooling, Normalize
from sentence_transformers import SentenceTransformer
import faiss
import torch
TOP_K = 10

class EvidenceRetriever:
    def __init__(self, model_name: str, quantize: bool = False):
        self.quantize = quantize
        self.timings = {}

        with TimingContext("model_initialization", self.timings):
            if self.quantize:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )

                transformer = Transformer(
                    model_name_or_path=model_name,
                    model_args={
                        "trust_remote_code": True,
                        "quantization_config": quantization_config,
                        "device_map": "auto"
                    }
                )
                print('quantized')
                pooling = Pooling(
                    transformer.get_word_embedding_dimension(), pooling_mode="mean")
                normalize = Normalize()

                self.model = SentenceTransformer(
                    modules=[transformer, pooling, normalize],
                    trust_remote_code=True
                )
            else:
                print('not quantized')
                self.model = SentenceTransformer(
                    model_name, trust_remote_code=True)

        self.index = None
        self.sentences = None

    def _compute_embeddings(self, sentences):
        with TimingContext("embedding_computation", self.timings):
            embeddings = self.model.encode(
                sentences,
                show_progress_bar=True,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().numpy()
        return embeddings

    def build_index(self, sentences):
        self.sentences = sentences
        embeddings = self._compute_embeddings(sentences)
        dimension = embeddings.shape[1]

        with TimingContext("faiss_index_building", self.timings):
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings.astype('float32'))

    def retrieve_evidence(self, query: str, k: int = TOP_K):
        with TimingContext("evidence_retrieval", self.timings):
            query_embedding = self._compute_embeddings([query])
            distances, indices = self.index.search(
                query_embedding.astype('float32'), k
            )

            results = []
            for idx, distance in zip(indices[0], distances[0]):
                results.append((idx, float(distance), self.sentences[idx]))
        return results
