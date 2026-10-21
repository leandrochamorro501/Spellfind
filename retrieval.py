"""Tuple-compatible retrieval results with explicit per-modality diagnostics."""


class SearchBatch(tuple):
    def __new__(cls, lyrics, audio, ranked, details):
        result = super().__new__(cls, (lyrics, audio, ranked))
        result.details = details
        return result


class RetrievalError(RuntimeError):
    def __init__(self, details):
        super().__init__("No usable retrieval results; at least one branch failed")
        self.details = details
