import pytest

import services.llm.openai_service as openai_module
from services.llm.openai_service import OpenAILLM


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", [None, "https://llm.example.test/v1"])
async def test_openai_client_uses_voice_request_timeout(monkeypatch, base_url) -> None:
    captured_options = {}

    class FakeAsyncOpenAI:
        def __init__(self, **options) -> None:
            captured_options.update(options)

    monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAI)
    llm = OpenAILLM(api_key="test-key", base_url=base_url)

    await llm._get_client()

    assert captured_options["api_key"] == "test-key"
    assert captured_options["timeout"] == 30.0
    if base_url:
        assert captured_options["base_url"] == base_url
    else:
        assert "base_url" not in captured_options
