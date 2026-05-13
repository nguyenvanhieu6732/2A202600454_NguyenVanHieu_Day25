from __future__ import annotations

from dataclasses import dataclass
import time

from reliability_lab.cache import ResponseCache, SharedRedisCache
from reliability_lab.circuit_breaker import CircuitBreaker, CircuitOpenError
from reliability_lab.providers import FakeLLMProvider, ProviderError, ProviderResponse

try:
    from prometheus_client import Counter, Histogram
    AGENT_REQUESTS = Counter('agent_requests_total', 'Total agent requests', ['route'])
    AGENT_LATENCY = Histogram('agent_latency_seconds', 'Agent request latency')
    CACHE_HITS = Counter('cache_hits_total', 'Total cache hits')
except ImportError:
    AGENT_REQUESTS = None
    AGENT_LATENCY = None
    CACHE_HITS = None


@dataclass(slots=True)
class GatewayResponse:
    text: str
    route: str
    provider: str | None
    cache_hit: bool
    latency_ms: float
    estimated_cost: float
    error: str | None = None


class ReliabilityGateway:
    """Routes requests through cache, circuit breakers, and fallback providers."""

    def __init__(
        self,
        providers: list[FakeLLMProvider],
        breakers: dict[str, CircuitBreaker],
        cache: ResponseCache | SharedRedisCache | None = None,
    ):
        self.providers = providers
        self.breakers = breakers
        self.cache = cache

    def complete(self, prompt: str) -> GatewayResponse:
        """Return a reliable response or a static fallback.

        TODO(student): Improve route reasons, cache safety checks, and error handling.
        TODO(student): Add cost budget check — if cumulative cost exceeds a threshold,
        skip expensive providers and route to cache or cheaper fallback.
        """
        start_time = time.perf_counter()
        if self.cache is not None:
            cached, score = self.cache.get(prompt)
            if cached is not None:
                latency = (time.perf_counter() - start_time) * 1000
                if CACHE_HITS: CACHE_HITS.inc()
                if AGENT_REQUESTS: AGENT_REQUESTS.labels(route="cache").inc()
                if AGENT_LATENCY: AGENT_LATENCY.observe(latency / 1000.0)
                return GatewayResponse(
                    cached, 
                    f"cache_hit:{score:.2f}", 
                    None, 
                    True, 
                    latency, 
                    0.0
                )

        last_error: str | None = None
        for provider in self.providers:
            breaker = self.breakers[provider.name]
            try:
                response: ProviderResponse = breaker.call(provider.complete, prompt)
                if self.cache is not None:
                    self.cache.set(prompt, response.text, {"provider": provider.name})
                route = f"primary:{provider.name}" if provider == self.providers[0] else f"fallback:{provider.name}"
                latency = (time.perf_counter() - start_time) * 1000
                if AGENT_REQUESTS: AGENT_REQUESTS.labels(route=route).inc()
                if AGENT_LATENCY: AGENT_LATENCY.observe(latency / 1000.0)
                return GatewayResponse(
                    text=response.text,
                    route=route,
                    provider=provider.name,
                    cache_hit=False,
                    latency_ms=latency,
                    estimated_cost=response.estimated_cost,
                )
            except (ProviderError, CircuitOpenError) as exc:
                last_error = str(exc)
                continue

        latency = (time.perf_counter() - start_time) * 1000
        if AGENT_REQUESTS: AGENT_REQUESTS.labels(route="static_fallback").inc()
        if AGENT_LATENCY: AGENT_LATENCY.observe(latency / 1000.0)
        return GatewayResponse(
            text="The service is temporarily degraded. Please try again soon.",
            route="static_fallback",
            provider=None,
            cache_hit=False,
            latency_ms=latency,
            estimated_cost=0.0,
            error=last_error,
        )
