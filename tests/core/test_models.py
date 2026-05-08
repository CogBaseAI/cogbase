from cogbase.core.models import Chunk


def test_chunk_embedding_optional():
    c = Chunk(chunk_id="doc1_0", doc_id="doc1", text="hello")
    assert c.embedding is None
