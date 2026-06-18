from functools import lru_cache
from ipaddress import ip_address, ip_network

from fastapi import Request

from config.settings import get_settings


class ClientIPResolver:
    def __init__(self, trusted_proxy_cidrs: list[str]) -> None:
        self._trusted_proxies = tuple(
            ip_network(cidr.strip()) for cidr in trusted_proxy_cidrs if cidr.strip()
        )

    def resolve(self, request: Request) -> str:
        peer = request.client.host if request.client else "unknown"
        if not self._is_trusted_proxy(peer):
            return peer

        forwarded = request.headers.get("X-Forwarded-For", "").split(",", 1)[0].strip()
        return forwarded if self._is_valid_address(forwarded) else peer

    def _is_trusted_proxy(self, peer: str) -> bool:
        try:
            peer_address = ip_address(peer)
        except ValueError:
            return False
        return any(peer_address in network for network in self._trusted_proxies)

    @staticmethod
    def _is_valid_address(candidate: str) -> bool:
        try:
            ip_address(candidate)
        except ValueError:
            return False
        return True


@lru_cache
def _configured_resolver() -> ClientIPResolver:
    cidrs = get_settings().trusted_proxy_cidrs.split(",")
    return ClientIPResolver(cidrs)


def get_client_ip(request: Request) -> str:
    return _configured_resolver().resolve(request)
