import hmac
import os
import threading
import time
from collections import defaultdict, deque


LOCAL_CLIENT_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
ADMIN_PATHS = {"/upload", "/delete", "/profiles"}


def extract_bearer_token(authorization: str) -> str:
    """从 Authorization 请求头中提取 Bearer Token。"""
    value = (authorization or "").strip()
    scheme, separator, token = value.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return ""
    return token.strip()


def token_matches(provided: str, expected: str) -> bool:
    """使用常量时间比较两个 Token。"""
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided, expected)


def is_public_path(path: str, docs_public: bool = True) -> bool:
    """判断路径是否允许匿名访问。"""
    if path in {"/", "/health"}:
        return True
    if docs_public and (
        path in {"/docs", "/redoc", "/openapi.json"}
        or path.startswith("/docs/")
        or path.startswith("/openapi")
    ):
        return True
    return False


def path_requires_admin(path: str) -> bool:
    """判断路径是否属于知识库和全局管理操作。"""
    return path in ADMIN_PATHS


def is_local_client(host: str) -> bool:
    """判断客户端是否来自本机。"""
    return (host or "").strip().lower() in LOCAL_CLIENT_HOSTS


def external_bind_enabled(host: str) -> bool:
    """判断服务是否绑定到了非本机地址。"""
    return (host or "").strip().lower() not in LOCAL_CLIENT_HOSTS


def validate_network_security(host: str, api_token: str, admin_token: str) -> None:
    """检查对外监听时是否配置了相互独立的管理 Token。"""
    if not external_bind_enabled(host):
        return
    if not api_token:
        raise ValueError("对外监听时必须设置 RAG_API_TOKEN")
    if not admin_token:
        raise ValueError("对外监听时必须设置独立的 RAG_ADMIN_TOKEN")
    if token_matches(api_token, admin_token):
        raise ValueError("RAG_API_TOKEN 与 RAG_ADMIN_TOKEN 不能相同")


def docs_are_public() -> bool:
    """读取 API 文档是否允许匿名访问的配置。"""
    return os.environ.get("RAG_PUBLIC_DOCS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


class SlidingWindowRateLimiter:
    """线程安全的单进程滑动窗口限流器。"""

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = max(1, int(window_seconds))
        self._events = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int) -> tuple[bool, int]:
        """记录一次请求并返回是否放行及建议等待秒数。"""
        if limit <= 0:
            return True, 0
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                retry_after = max(
                    1,
                    int(self.window_seconds - (now - events[0])) + 1,
                )
                return False, retry_after
            events.append(now)
        return True, 0


def configured_rate_limit(admin: bool) -> int:
    """读取普通或管理接口的每分钟请求上限。"""
    name = (
        "RAG_ADMIN_RATE_LIMIT_PER_MINUTE"
        if admin
        else "RAG_RATE_LIMIT_PER_MINUTE"
    )
    default = "30" if admin else "120"
    try:
        return max(0, int(os.environ.get(name, default)))
    except ValueError:
        return int(default)
