"""Shared HTTP helper: an httpx client that trusts the OS certificate store.

Centralises the TLS-inspecting-proxy (e.g. Zscaler) handling used by both the LLM
client and the EDGAR client.
"""

from __future__ import annotations

import logging
import ssl

import httpx

logger = logging.getLogger(__name__)


def build_http_client(
    *,
    use_os_truststore: bool = True,
    timeout_s: float = 60.0,
    headers: dict[str, str] | None = None,
) -> httpx.Client:
    """Build an httpx client, optionally trusting the OS certificate store."""
    if use_os_truststore:
        try:
            import truststore

            ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            return httpx.Client(verify=ctx, timeout=timeout_s, headers=headers)
        except Exception:  # pragma: no cover - truststore missing/unsupported
            logger.warning("truststore unavailable; using default SSL verification")
    return httpx.Client(timeout=timeout_s, headers=headers)
