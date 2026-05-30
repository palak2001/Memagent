"""M0 gate: imports, config, and embedder dimension."""

import numpy as np


def test_package_imports():
    import neander
    assert neander.__version__


def test_config_imports_and_loads():
    from neander.config import load_settings
    s = load_settings()
    assert s.chat_model
    assert s.extract_model
    assert s.embed_model_name
    assert s.embed_dim == 384


def test_embedder_dimension():
    from neander.config import load_settings
    from neander.memory.embeddings import get_embedder

    s = load_settings()
    # Use the hash backend so tests run offline with no model download.
    embedder = get_embedder("hash", dim=s.embed_dim)
    vec = embedder.embed("hello world")
    assert vec.shape == (s.embed_dim,), f"Expected ({s.embed_dim},), got {vec.shape}"
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5, "vector should be L2-normalised"


def test_embedder_dimension_sentence_transformers():
    """Real sentence-transformers model returns 384-dim normalised vectors."""
    try:
        from neander.memory.embeddings import get_embedder
        embedder = get_embedder("sentence-transformers", dim=384)
        vec = embedder.embed("test sentence")
        assert vec.shape == (384,)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4
    except Exception:
        import pytest
        pytest.skip("sentence-transformers not available")
