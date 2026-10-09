def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU (skipped in CI: pytest -m 'not gpu')")
