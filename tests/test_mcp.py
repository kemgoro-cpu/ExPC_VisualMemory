import json

import pytest
from conftest import add_event

from visual_memory import mcp_server


def test_mcp_lists_new_context_documents_without_manual_approval(monkeypatch, settings, service):
    monkeypatch.setenv("VISUAL_MEMORY_DATA_DIR", str(settings.data_dir))
    event_id = add_event(service, "approved MCP evidence")
    document_pack = service.packs.create("ready context", [event_id])
    listed = json.loads(mcp_server.list_context_packs())
    assert "artifact_path" not in listed[0]
    assert "manifest_sha256" not in listed[0]
    assert [pack["id"] for pack in listed] == [document_pack["id"]]
    manifest = json.loads(mcp_server.get_context_pack(document_pack["id"]))
    assert manifest["items"][0]["event_id"] == event_id
    document = mcp_server.get_context_document(document_pack["id"], "html")
    assert document.resource.mimeType == "text/html"
    assert "data:image/webp;base64," in document.resource.text

    service.packs.revoke(document_pack["id"])
    assert json.loads(mcp_server.list_context_packs()) == []
    with pytest.raises(PermissionError):
        mcp_server.get_context_document(document_pack["id"], "html")
