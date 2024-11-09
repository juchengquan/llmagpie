from prometheus_client import make_asgi_app, CollectorRegistry, multiprocess


def make_metrics_app():
    """Using multiprocess collector for registry.
    """
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return make_asgi_app(registry=registry)
