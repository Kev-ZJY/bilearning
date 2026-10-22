"""Bounded public-web research with pinned DNS, no credentials and auditable text.

Search snippets are discovery aids, never read-page evidence. All page content is
untrusted data; callers must not execute instructions found in it.
"""
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import queue
import re
import socket
import ssl
import sys
import threading
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import getproxies, proxy_bypass
from xml.etree import ElementTree

try:
    from urllib.request import proxy_bypass_environment
except ImportError:  # Older or embedded Python builds may omit this helper.
    proxy_bypass_environment = None

try:
    from urllib.request import proxy_bypass_macosx_sysconf
except ImportError:  # Available only on Python builds with macOS proxy support.
    proxy_bypass_macosx_sysconf = None

from app.core.cancel import TaskCancelled

MAX_BYTES = 1_500_000
MAX_TEXT = 24_000
TIMEOUT = 18.0
MAX_REDIRECTS = 3
DNS_JSON_MAX_BYTES = 65_536
PUBLIC_DNS_JSON_ENDPOINT = "https://dns.google/resolve"


class _ProxyConnectionFailure(OSError):
    """A proxy socket connection failed before an origin response existed."""


class _PinnedConnection(http.client.HTTPConnection):
    # HTTPConnection normally closes its socket as soon as a Connection: close
    # response arrives, while HTTPResponse keeps the file alive. Keep the handle
    # until reading finishes so cancellation/deadline shutdown can reach it.
    hold_socket = True

    def close(self):
        if not self.hold_socket:
            super().close()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _public(address):
    ip = ipaddress.ip_address(address)
    return ip.is_global and not (ip.is_multicast or ip.is_reserved or ip.is_unspecified)


def _parse_google_dns_a_response(body, hostname):
    """Validate a bounded Google JSON-DNS A response without filtering answers."""
    if not isinstance(body, bytes) or len(body) > DNS_JSON_MAX_BYTES:
        raise ValueError("公共DNS响应超过体积上限")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("公共DNS响应格式无效") from None
    if not isinstance(payload, dict) or type(payload.get("Status")) is not int or payload["Status"] != 0:
        raise ValueError("公共DNS查询未成功")
    if payload.get("TC") is not False:
        raise ValueError("公共DNS响应被截断或标记无效")

    questions = payload.get("Question")
    requested_name = str(hostname).lower().rstrip(".")
    if not isinstance(questions, list) or len(questions) != 1 or not isinstance(questions[0], dict):
        raise ValueError("公共DNS问题字段无效")
    question = questions[0]
    question_name = question.get("name")
    if not isinstance(question_name, str):
        raise ValueError("公共DNS问题字段无效")
    try:
        question_name = question_name.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError:
        raise ValueError("公共DNS问题字段无效") from None
    if question_name != requested_name or type(question.get("type")) is not int or question["type"] != 1:
        raise ValueError("公共DNS问题与请求不匹配")

    answers = payload.get("Answer", [])
    if not isinstance(answers, list):
        raise ValueError("公共DNS答案字段无效")
    addresses = []
    for record in answers:
        if not isinstance(record, dict) or type(record.get("type")) is not int:
            raise ValueError("公共DNS答案字段无效")
        if record["type"] != 1:
            continue
        value = record.get("data")
        if not isinstance(value, str):
            raise ValueError("公共DNS A记录无效")
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            raise ValueError("公共DNS A记录无效") from None
        if address.version != 4 or not _public(str(address)):
            raise ValueError("公共DNS包含非公网A记录")
        normalized = address.compressed
        if normalized not in addresses:
            addresses.append(normalized)
    if not addresses:
        raise ValueError("公共DNS没有可用A记录")
    return addresses


def _google_dns_public_a(hostname, *, deadline, checkpoint):
    """Resolve one already-validated origin name through the fixed pinned DoH URL."""
    query = urlencode({
        "name": hostname,
        "type": "A",
        "edns_client_subnet": "0.0.0.0/0",
    })
    resolver_url = f"{PUBLIC_DNS_JSON_ENDPOINT}?{query}"
    parsed, resolver_addresses = resolve_public_url(
        resolver_url, checkpoint=checkpoint, deadline=deadline,
    )
    status, headers, body = _request(
        parsed,
        resolver_addresses,
        deadline=deadline,
        checkpoint=checkpoint,
        max_body_bytes=DNS_JSON_MAX_BYTES,
        accept="application/dns-json",
    )
    if status != 200:
        raise ValueError("公共DNS服务返回非成功状态")
    content_type = headers.get("content-type", "").lower().split(";", 1)[0].strip()
    if content_type not in {"application/dns-json", "application/json"}:
        raise ValueError("公共DNS服务返回非JSON内容")
    return _parse_google_dns_a_response(body, hostname)


def resolve_public_url(url, *, checkpoint=lambda: None, deadline=None):
    """Resolve once, validate every answer, and return addresses for direct connect."""
    if not isinstance(url, str) or len(url) > 4096 or any(ord(c) <= 32 for c in url):
        raise ValueError("无效网页URL")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("网页仅允许公网 HTTP/HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("网页URL禁止包含凭证")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in {80, 443}:
        raise ValueError("网页仅允许80/443端口")
    host = parsed.hostname.encode("idna").decode("ascii").lower().rstrip(".")
    if "%" in host or host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("网页地址必须属于公网")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None and not _public(str(literal)):
        raise ValueError("网页地址必须属于公网")
    deadline = min(deadline or time.monotonic() + TIMEOUT, time.monotonic() + 5)
    result = queue.Queue(maxsize=1)

    def resolve():
        try:
            result.put(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except Exception as exc:
            result.put(exc)

    threading.Thread(target=resolve, daemon=True).start()
    while True:
        checkpoint()
        if time.monotonic() >= deadline:
            raise TimeoutError("网页DNS解析超时")
        try:
            answers = result.get(timeout=min(0.1, deadline - time.monotonic()))
            break
        except queue.Empty:
            continue
    if isinstance(answers, Exception):
        raise answers
    addresses = list(dict.fromkeys((row[0], row[4][0]) for row in answers))
    if not addresses or any(not _public(ip) for _, ip in addresses):
        raise ValueError("网页DNS包含非公网地址")
    addresses.sort(key=lambda row: row[0] != socket.AF_INET)
    netloc = f"[{host}]" if ":" in host else host
    if port != (443 if parsed.scheme == "https" else 80):
        netloc += f":{port}"
    path = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
    query = quote(parsed.query, safe="%:@!$&'()*+,;=/?-._~")
    normalized = urlunsplit((parsed.scheme, netloc, path, query, ""))
    return urlsplit(normalized), addresses


def _proxy_for_url(parsed):
    """Return a validated HTTP proxy endpoint from system/environment config."""
    if _proxy_bypassed(parsed.hostname):
        return None
    proxies = getproxies()
    value = proxies[parsed.scheme] if parsed.scheme in proxies else proxies.get("all")
    if not value:
        return None
    if not isinstance(value, str):
        raise ValueError("网页代理配置无效")
    value = value.strip()
    if not value:
        return None
    if "://" not in value:
        value = "http://" + value
    try:
        proxy = urlsplit(value)
        if proxy.username is not None or proxy.password is not None:
            raise ValueError("网页代理认证不受支持")
        if proxy.scheme.lower() != "http":
            raise ValueError("网页代理协议不受支持")
        if not proxy.hostname or proxy.path not in {"", "/"} or proxy.query or proxy.fragment:
            raise ValueError("网页代理配置无效")
        port = proxy.port or 80
        if not 1 <= port <= 65535:
            raise ValueError("网页代理配置无效")
    except ValueError as exc:
        if str(exc) in {
            "网页代理认证不受支持", "网页代理协议不受支持", "网页代理配置无效",
        }:
            raise
        raise ValueError("网页代理配置无效") from None
    return proxy.hostname, port


def _proxy_bypassed(host):
    """Honor environment bypasses and macOS SystemConfiguration bypasses."""
    environment_check = proxy_bypass_environment
    system_check = proxy_bypass_macosx_sysconf
    if callable(environment_check) and environment_check(host):
        return True
    if sys.platform == "darwin" and callable(system_check) and system_check(host):
        return True
    return bool(proxy_bypass(host))


def _resolve_proxy_endpoint(host, port, *, deadline, checkpoint):
    """Resolve a configured proxy with the request deadline, without public-IP rules."""
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        address = str(literal)
        family = socket.AF_INET6 if literal.version == 6 else socket.AF_INET
        sockaddr = (address, port, 0, 0) if literal.version == 6 else (address, port)
        return [(family, sockaddr)]

    resolve_deadline = min(deadline, time.monotonic() + 5)
    result = queue.Queue(maxsize=1)

    def resolve():
        try:
            result.put(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except Exception as exc:
            result.put(exc)

    threading.Thread(target=resolve, daemon=True).start()
    while True:
        checkpoint()
        remaining = resolve_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("网页代理解析超时")
        try:
            answers = result.get(timeout=min(0.1, remaining))
            break
        except queue.Empty:
            continue
    if isinstance(answers, Exception):
        raise OSError("网页代理连接失败") from None
    addresses = list(dict.fromkeys((row[0], row[4]) for row in answers))
    if not addresses:
        raise OSError("网页代理连接失败")
    addresses.sort(key=lambda row: row[0] != socket.AF_INET)
    return addresses


def _numeric_host(ip):
    address = ipaddress.ip_address(ip)
    return f"[{address.compressed}]" if address.version == 6 else address.compressed


def _connect_tunnel(sock, ip, port, *, deadline, checkpoint, max_headers=32_768):
    """Tunnel to the pinned numeric address; never expose the origin DNS name."""
    authority = f"{_numeric_host(ip)}:{port}"
    request = (
        f"CONNECT {authority} HTTP/1.1\r\n"
        f"Host: {authority}\r\n"
        "Proxy-Connection: keep-alive\r\n\r\n"
    ).encode("ascii")
    checkpoint()
    sock.sendall(request)
    checkpoint()

    response = bytearray()
    boundary = -1
    while boundary < 0:
        checkpoint()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("网页读取超时")
        if len(response) >= max_headers:
            raise ValueError("网页代理响应头超过限制")
        sock.settimeout(min(4, remaining))
        chunk = sock.recv(min(4096, max_headers + 1 - len(response)))
        if not chunk:
            raise ValueError("网页代理隧道响应无效")
        response.extend(chunk)
        if len(response) > max_headers:
            raise ValueError("网页代理响应头超过限制")
        boundary = response.find(b"\r\n\r\n")
    if boundary + 4 != len(response):
        raise ValueError("网页代理隧道响应含意外数据")

    lines = bytes(response[:boundary]).split(b"\r\n")
    status = lines[0].split()
    if len(status) < 2 or status[0] not in {b"HTTP/1.0", b"HTTP/1.1"}:
        raise ValueError("网页代理隧道响应无效")
    try:
        status_code = int(status[1])
    except ValueError:
        raise ValueError("网页代理隧道响应无效") from None
    if status_code != 200:
        raise ValueError("网页代理隧道建立失败")

    content_lengths = []
    transfer_encodings = []
    for line in lines[1:]:
        if not line or line[:1] in b" \t":
            raise ValueError("网页代理隧道响应无效")
        name, separator, value = line.partition(b":")
        if not separator or not name or any(byte < 33 or byte > 126 for byte in name):
            raise ValueError("网页代理隧道响应无效")
        lower_name = name.lower()
        if lower_name == b"content-length":
            try:
                content_lengths.append(int(value.strip()))
            except ValueError:
                raise ValueError("网页代理隧道响应无效") from None
        elif lower_name == b"transfer-encoding":
            transfer_encodings.append(value.strip())
    if any(length != 0 for length in content_lengths) or transfer_encodings:
        raise ValueError("网页代理隧道响应包含正文")


def _request(parsed, addresses, *, deadline, checkpoint, max_body_bytes=MAX_BYTES, accept=None):
    """Fetch through a configured HTTP proxy without delegating origin DNS."""
    checkpoint()
    if type(max_body_bytes) is not int or not 1 <= max_body_bytes <= MAX_BYTES:
        raise ValueError("网页响应体积上限无效")
    family, ip = addresses[0]
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    proxy = _proxy_for_url(parsed)
    if proxy is not None:
        proxy_host, proxy_port = proxy
        proxy_addresses = _resolve_proxy_endpoint(
            proxy_host, proxy_port, deadline=deadline, checkpoint=checkpoint,
        )
        connect_family, proxy_sockaddr = proxy_addresses[0]
    else:
        connect_family, proxy_sockaddr = family, (ip, port)

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("网页读取超时")
    sock = socket.socket(connect_family, socket.SOCK_STREAM)
    conn = _PinnedConnection(parsed.hostname, port, timeout=min(8, remaining))
    active = [sock]
    deadline_expired = threading.Event()
    cancel_stop = threading.Event()
    cancel_error = []

    def shutdown_active():
        try:
            active[0].shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def expire():
        deadline_expired.set()
        shutdown_active()

    def watch_cancel():
        while not cancel_stop.wait(0.05):
            try:
                checkpoint()
            except BaseException as exc:
                cancel_error.append(exc)
                shutdown_active()
                return

    def check():
        if cancel_error:
            raise cancel_error[0]
        checkpoint()
        if time.monotonic() >= deadline:
            raise TimeoutError("网页读取超时")

    timer = threading.Timer(remaining, expire)
    timer.daemon = True
    timer.start()
    cancel_watcher = threading.Thread(target=watch_cancel, daemon=True)
    cancel_watcher.start()
    try:
        check()
        sock.settimeout(min(8, remaining))
        try:
            sock.connect(proxy_sockaddr)
        except OSError:
            check()
            if proxy is not None:
                raise _ProxyConnectionFailure("网页代理连接失败") from None
            raise
        check()
        if proxy is not None and parsed.scheme == "https":
            _connect_tunnel(sock, ip, port, deadline=deadline, checkpoint=check)
            check()
        if parsed.scheme == "https":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("网页读取超时")
            sock.settimeout(min(8, remaining))
            sock = ssl.create_default_context().wrap_socket(
                sock, server_hostname=parsed.hostname, do_handshake_on_connect=False,
            )
            active[0] = sock
            check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("网页读取超时")
            sock.settimeout(min(8, remaining))
            sock.do_handshake()
            check()
        conn.sock = sock
        origin_target = parsed.path + ("?" + parsed.query if parsed.query else "")
        if proxy is not None and parsed.scheme == "http":
            authority = _numeric_host(ip)
            if port != 80:
                authority += f":{port}"
            target = f"http://{authority}{origin_target}"
        else:
            target = origin_target
        check()
        conn.request("GET", target, headers={
            "Host": parsed.netloc,
            "User-Agent": "VideoKnowledgeReview/1.0 (public source verification)",
            "Accept": accept or "text/html,application/xhtml+xml,text/plain,application/rss+xml,application/xml;q=0.8",
            "Accept-Encoding": "identity", "Connection": "close",
        })
        response = conn.getresponse()
        check()
        headers = {k.lower(): v for k, v in response.getheaders()}
        if response.status in {301, 302, 303, 307, 308}:
            return response.status, headers, b""
        if headers.get("content-encoding", "identity").lower() != "identity":
            raise ValueError("不读取压缩或未知编码的网页响应")
        if headers.get("content-length") and int(headers["content-length"]) > max_body_bytes:
            raise ValueError("网页响应超过体积上限")
        body = bytearray()
        while True:
            check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("网页读取超时")
            sock.settimeout(min(4, remaining))
            piece = response.read1(min(65536, max_body_bytes + 1 - len(body)))
            check()
            if not piece:
                break
            body.extend(piece)
            if len(body) > max_body_bytes:
                raise ValueError("网页响应超过体积上限")
        return response.status, headers, bytes(body)
    except BaseException:
        if cancel_error:
            raise cancel_error[0]
        if deadline_expired.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("网页读取超时") from None
        raise
    finally:
        cancel_stop.set()
        timer.cancel()
        cancel_watcher.join(timeout=0.2)
        conn.hold_socket = False
        conn.close()
        sock.close()


def fetch_public(url, *, checkpoint=lambda: None):
    deadline = time.monotonic() + TIMEOUT
    visited = set()
    dns_fallback_used = False
    for _ in range(MAX_REDIRECTS + 1):
        checkpoint()
        parsed, addresses = resolve_public_url(url, checkpoint=checkpoint, deadline=deadline)
        normalized = parsed.geturl()
        if normalized in visited:
            raise ValueError("网页重定向循环")
        visited.add(normalized)
        proxy_configured = parsed.scheme == "https" and _proxy_for_url(parsed) is not None
        try:
            status, headers, body = _request(parsed, addresses, deadline=deadline, checkpoint=checkpoint)
        except (TimeoutError, ConnectionError, _ProxyConnectionFailure) as exc:
            if (
                dns_fallback_used
                or not proxy_configured
                or isinstance(exc, (TaskCancelled, ssl.SSLError))
            ):
                raise
            dns_fallback_used = True
            fallback_addresses = _google_dns_public_a(
                parsed.hostname, deadline=deadline, checkpoint=checkpoint,
            )
            status, headers, body = _request(
                parsed,
                [(socket.AF_INET, address) for address in fallback_addresses],
                deadline=deadline,
                checkpoint=checkpoint,
            )
        if status in {301, 302, 303, 307, 308}:
            if not headers.get("location"):
                raise ValueError("网页重定向缺少目标")
            url = urljoin(normalized, headers["location"])
            continue
        if status != 200:
            raise ValueError(f"网页返回 HTTP {status}")
        content_type = headers.get("content-type", "").lower()
        if content_type.split(";")[0].strip() not in {
            "text/html", "text/plain", "application/xhtml+xml", "application/rss+xml",
            "application/xml", "text/xml",
        }:
            raise ValueError("网页不是可读取的文本类型")
        if len(body) > MAX_BYTES:
            raise ValueError("网页响应超过体积上限")
        return {"url": normalized, "content_type": content_type, "body": body}
    raise ValueError("网页重定向次数超过上限")


class _PageText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = []
        self.title_depth = 0
        self.title = []
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "noscript", "svg", "form", "footer"}:
            self.skip.append(tag)
        if tag == "title":
            self.title_depth += 1
        if not self.skip and tag in {"p", "div", "section", "article", "h1", "h2", "h3", "li", "br", "tr", "pre"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.skip and tag == self.skip[-1]:
            self.skip.pop()
        if tag == "title":
            self.title_depth = max(0, self.title_depth - 1)

    def handle_data(self, data):
        if self.title_depth:
            self.title.append(data)
        elif not self.skip:
            self.parts.append(data)


def _decode(page):
    charset = re.search(r"charset\s*=\s*[\"']?([\w-]+)", page["content_type"])
    encoding = charset.group(1) if charset else "utf-8"
    try:
        return page["body"].decode(encoding, errors="replace")
    except LookupError:
        return page["body"].decode("utf-8", errors="replace")


class _SearchLinks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results, self.current = [], None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "result__a" in attrs.get("class", ""):
            href = urljoin("https://html.duckduckgo.com", attrs.get("href", ""))
            target = parse_qs(urlsplit(href).query).get("uddg", [href])[0]
            if urlsplit(target).scheme in {"http", "https"}:
                self.current = {"url": target, "title": "", "snippet": ""}

    def handle_data(self, data):
        if self.current is not None:
            self.current["title"] += data

    def handle_endtag(self, tag):
        if tag == "a" and self.current is not None:
            self.results.append(self.current)
            self.current = None


class WebResearch:
    def __init__(self, *, checkpoint=lambda: None):
        self.checkpoint = checkpoint
        self.sources = {}
        self.search_attempts = []
        self._pages = {}

    def search(self, query, limit=5):
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("检索词必须为1至500字符")
        if type(limit) is not int or not 1 <= limit <= 8:
            raise ValueError("检索结果数必须为1至8")
        providers = [
            ("bing_rss", "https://www.bing.com/search?" + urlencode({"q": query, "format": "rss"})),
            ("duckduckgo_html", "https://html.duckduckgo.com/html/?" + urlencode({"q": query})),
        ]
        for provider, url in providers:
            self.checkpoint()
            attempt = {"query": query, "provider": provider, "started_at": _now(), "status": "running"}
            self.search_attempts.append(attempt)
            try:
                page = fetch_public(url, checkpoint=self.checkpoint)
                if provider == "bing_rss":
                    tree = ElementTree.fromstring(page["body"])
                    results = [{"url": item.findtext("link", ""), "title": item.findtext("title", ""),
                                "snippet": item.findtext("description", "")[:1000]}
                               for item in tree.findall("./channel/item")]
                else:
                    parser = _SearchLinks()
                    parser.feed(_decode(page))
                    results = parser.results
                results = [r for r in results if urlsplit(r["url"]).scheme in {"http", "https"}][:limit]
                if not results:
                    raise ValueError("搜索未返回可用结果，可能被限制或需要验证码")
                attempt.update(status="ok", count=len(results))
                return {"query": query, "provider": provider, "results": results, "fetched_at": _now()}
            except TaskCancelled:
                attempt["status"] = "cancelled"
                raise
            except Exception as exc:
                attempt.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        raise RuntimeError("公网检索失败；未取得外部证据")

    def read(self, url, *, start=0, find=None):
        self.checkpoint()
        if type(start) is not int or start < 0 or (find is not None and (not isinstance(find, str) or not 1 <= len(find) <= 200)):
            raise ValueError("网页正文范围或查找词无效")
        cached = self._pages.get(url)
        if cached is None:
            page = fetch_public(url, checkpoint=self.checkpoint)
            raw = _decode(page)
            if "html" in page["content_type"]:
                parser = _PageText()
                parser.feed(raw)
                text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in "".join(parser.parts).splitlines() if line.strip())
                title = "".join(parser.title).strip()
            else:
                text, title = raw.strip(), ""
            cached = {"url": page["url"], "title": title[:300], "text": text,
                      "body_sha256": _sha256(text), "fetched_at": _now()}
            self._pages[url] = self._pages[page["url"]] = cached
        text = cached["text"]
        if len(text.strip()) < 30:
            raise ValueError("网页没有足够正文，不能作为已读取证据")
        if find is not None:
            position = text.lower().find(find.lower(), start)
            if position < 0:
                raise ValueError("网页正文未找到查找词")
            start = max(0, position - 500)
        if start >= len(text):
            raise ValueError("网页正文起点超出范围")
        excerpt = text[start:start + MAX_TEXT]
        result = {"url": cached["url"], "title": cached["title"], "text": excerpt,
                  "start": start, "next_start": start + len(excerpt), "total_chars": len(text),
                  "truncated": start > 0 or start + MAX_TEXT < len(text),
                  "body_sha256": cached["body_sha256"], "excerpt_sha256": _sha256(excerpt),
                  "fetched_at": cached["fetched_at"]}
        previous = self.sources.get(result["url"], {})
        segments = list(previous.get("segments", []))
        segment_sha256 = [_sha256(segment) for segment in segments]
        if excerpt not in segments:
            segments.append(excerpt)
            segment_sha256.append(result["excerpt_sha256"])
        self.sources[result["url"]] = {**result, "text": "\n".join(segments),
                                        "segments": segments, "segment_sha256": segment_sha256}
        return result
