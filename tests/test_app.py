from fastapi.testclient import TestClient


def test_health_returns_ok(monkeypatch):
    # Monkeypatch predict_bio to avoid loading the real ONNX model during tests
    from repo.app import main

    def _stub_predict_bio(texts, apply_regex_postprocess=True):
        return [[] for _ in texts]

    monkeypatch.setattr(main, "predict_bio", _stub_predict_bio)

    with TestClient(main.api) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_predict_basic(monkeypatch):
    from repo.app import main

    def _stub_predict_bio(texts, apply_regex_postprocess=True):
        # Emit a single BRAND span for each input text
        return [[(0, 3, "BRAND")] for _ in texts]

    monkeypatch.setattr(main, "predict_bio", _stub_predict_bio)

    with TestClient(main.api) as client:
        resp = client.post("/api/predict", json={"input": "abc"})
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 1
        item = body[0]
        assert set(item.keys()) == {"start_index", "end_index", "entity"}
        assert item["start_index"] == 0
        assert item["end_index"] == 3
        assert item["entity"] == "BRAND"
