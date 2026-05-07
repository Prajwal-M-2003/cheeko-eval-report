import os


_PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def disable_broken_loopback_proxy_env() -> bool:
    """
    Clear obviously broken local proxy settings such as 127.0.0.1:9.

    Some environments export dead loopback proxies to block outbound network
    access. That breaks normal API calls with WinError 10061.
    """
    changed = False
    for key in _PROXY_KEYS:
        value = (os.getenv(key, "") or "").strip().lower()
        if "127.0.0.1:9" in value or "localhost:9" in value:
            os.environ.pop(key, None)
            changed = True
    return changed
