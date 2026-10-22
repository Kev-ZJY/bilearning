"""Bounded public HTML inspection. Pin validated DNS addresses for each redirect."""

import http.client
import ipaddress
import re
import socket
import ssl
from html.parser import HTMLParser
from urllib.parse import urlsplit, urljoin


class _PublicHTTPS(http.client.HTTPSConnection):
    def connect(self):
        addresses = socket.getaddrinfo(self.host, self.port, type=socket.SOCK_STREAM)
        if not addresses or any(
            not ipaddress.ip_address(a[4][0]).is_global for a in addresses
        ):
            raise ValueError("只允许读取公开网页地址")
        family, kind, proto, _, address = addresses[0]
        sock = socket.socket(family, kind, proto)
        sock.settimeout(self.timeout)
        try:
            sock.connect(address)
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


class _Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = []
        self.text = []
        self.in_title = False
        self.hidden = 0
        self.video = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        if tag == "title":
            self.in_title = True
        if tag == "video":
            self.video = True
        values = dict(attrs)
        if (
            tag == "meta"
            and values.get("property") == "og:type"
            and values.get("content", "").startswith("video")
        ):
            self.video = True
        if tag == "meta" and values.get("name") == "description":
            self.text.append(values.get("content", ""))

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.hidden:
            return
        if self.in_title:
            self.title.append(data)
        self.text.append(data)


def inspect_page(url):
    for _ in range(4):
        parts = urlsplit(url)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.port not in (None, 443)
        ):
            raise ValueError("网页检查仅支持公开 HTTPS 地址")
        conn = _PublicHTTPS(
            parts.hostname, timeout=12, context=ssl.create_default_context()
        )
        try:
            conn.request(
                "GET",
                (parts.path or "/") + ("?" + parts.query if parts.query else ""),
                headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"},
            )
            response = conn.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                url = urljoin(url, response.getheader("Location", ""))
                continue
            if response.status != 200:
                raise ValueError(f"网页读取失败（HTTP {response.status}）")
            if "text/html" not in response.getheader("Content-Type", ""):
                return {
                    "url": url,
                    "title": "",
                    "text": "",
                    "video_hint": False,
                    "content_type": response.getheader("Content-Type", ""),
                }
            raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("网页超出检查大小上限")
            page = _Page()
            page.feed(raw.decode("utf-8", errors="replace"))
            return {
                "url": url,
                "title": "".join(page.title)[:300],
                "text": re.sub(r"\s+", " ", " ".join(page.text))[:6000],
                "video_hint": page.video,
            }
        finally:
            conn.close()
    raise ValueError("网页重定向次数过多")


def nonvideo_urls(text):
    urls = re.findall(r'https?://[^\s<>"\'，。；、（）()\[\]]+', text)
    return list(
        dict.fromkeys(
            u.rstrip(".,;!?")
            for u in urls
            if (urlsplit(u).hostname or "").lower()
            not in ("bilibili.com", "www.bilibili.com", "b23.tv")
        )
    )
