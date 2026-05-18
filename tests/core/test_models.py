from api.models import ChatMessage, QueryRequest
from cogbase.core.models import Chunk


def test_chunk_embedding_optional():
    c = Chunk(chunk_id="doc1_0", doc_id="doc1", text="hello")
    assert c.embedding is None


def test_to_storable_metadata_non_core_fields_spill():
    c = Chunk(chunk_id="c1", doc_id="d1", text="t", char_offset=10, char_length=5)
    stored = c.to_storable_metadata()
    assert stored["char_offset"] == 10
    assert stored["char_length"] == 5


def test_to_storable_metadata_none_fields_omitted():
    c = Chunk(chunk_id="c1", doc_id="d1", text="t")
    stored = c.to_storable_metadata()
    assert "char_offset" not in stored
    assert "char_length" not in stored


def test_to_storable_metadata_typed_field_wins_over_metadata_bag():
    # Top-level Chunk field should take precedence over a colliding metadata key.
    c = Chunk(chunk_id="c1", doc_id="d1", text="t", char_offset=10, metadata={"char_offset": 999})
    stored = c.to_storable_metadata()
    assert stored["char_offset"] == 10


def test_from_stored_roundtrip():
    c = Chunk(chunk_id="c1", doc_id="d1", text="t", char_offset=10, char_length=5,
              metadata={"source": "upload"})
    stored_meta = c.to_storable_metadata()
    restored = Chunk.from_stored(
        chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text,
        embedding=None, metadata=stored_meta,
    )
    assert restored.char_offset == 10
    assert restored.char_length == 5
    assert restored.metadata == {"source": "upload"}


def test_query_request_parses_chat_history():
    req = QueryRequest.model_validate(
        {
            "text": "what is the notice period?",
            "history": [
                {"role": "user", "content": "summarize the contract"},
                ChatMessage(role="assistant", content="The contract is 12 pages long."),
            ],
        }
    )

    assert req.text == "what is the notice period?"
    assert len(req.history) == 2
    assert req.history[0].role == "user"
    assert req.history[0].content == "summarize the contract"
    assert req.history[1].role == "assistant"
    assert req.history[1].content == "The contract is 12 pages long."
