from api.models import ChatMessage, QueryRequest
from cogbase.core.models import Chunk


def test_chunk_embedding_optional():
    c = Chunk(chunk_id="doc1_0", doc_id="doc1", text="hello")
    assert c.embedding is None


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
