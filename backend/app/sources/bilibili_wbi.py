"""B站 WBI 签名与元信息解析。

实现 B站 Web 端 WBI 鉴权签名（nav 接口 wbi_img -> mixin_key），
以及视频元信息(view)、播放地址(playurl)、字幕(player/v2)三个客户端方法。

设计背景（均为真实请求验证结论，请勿"优化"掉任何一条）：
1. yt-dlp 对 B站报 HTTP 412，必须自实现 WBI 签名，这是唯一可行路。
2. x/player/v2 字幕接口免登录返回空数组（实测 4 个 BV 号全部 0 条），
   字幕只是加分项，绝不作为主路径或必需依赖。
3. CDN 下载必须带全请求头（UA+Referer+Origin+Accept+Accept-Language），
   只带 User-Agent 会 412。
4. nav 接口返回 code=-101（未登录）但密钥仍有效，必须无视 code 直接读 wbi_img，
   很多项目写 `if code != 0: raise` 于是永远拿不到密钥。

清晰度（免登录上限）：accept_quality=[116,80,64,32,16]=1080P60/1080P/720P/480P/360P，
默认 64=1080P。4K/8K 需登录。
"""
from __future__ import annotations

import hashlib
import re
import time
from urllib.parse import parse_qs, urlparse

import requests

# 64 位混合序列表（B站官方固定），用于从 img_key+sub_key 里挑字符组成 mixin_key。
MIXIN_TABLE = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_BV_RE = re.compile(r"BV[0-9A-Za-z]+")


def mixin_key(img_key: str, sub_key: str) -> str:
    """由 nav 接口的 img_key/sub_key 混合出 32 位 wbi 签名密钥。

    两个 key 各约 32 位，拼接成 64 位；按 MIXIN_TABLE 挑位后取前 32 位。
    纯函数、确定性：相同输入恒得相同输出。
    """
    raw = img_key + sub_key
    mixed = "".join(raw[i] for i in MIXIN_TABLE)
    return mixed[:32]


def sign_params(params: dict, key: str, wts: int | None = None) -> dict:
    """给参数加上 wts 时间戳和 w_rid 签名，返回新字典（不修改入参）。

    w_rid = md5(按 key 排序拼接的 query + mixin_key)，B站服务端据此验签。
    """
    wts = wts if wts is not None else int(time.time())
    signed = dict(params)
    signed["wts"] = wts
    query = "&".join(
        f"{k}={v}" for k, v in sorted(signed.items())
        if v is not None and v != ""
    )
    signed["w_rid"] = hashlib.md5((query + key).encode("utf-8")).hexdigest()
    return signed


def parse_bv_input(text: str) -> str:
    """从任意输入提取 BV 号；b23.tv 短链无法用正则提取，原样返回交给后续步骤。

    支持：带尾斜杠 URL、不带尾斜杠 URL、裸 BV 号、短链（原样返回）。
    """
    text = text.strip()
    if "b23.tv" in text:
        return text
    m = _BV_RE.search(text)
    if m:
        return m.group(0)
    return text


def parse_bv_input_ex(text: str) -> dict:
    """解析 B站 链接，返回 {"bvid": str, "page": int | None, "page_invalid": bool}。

    踩坑背景（请勿"优化"回旧逻辑）：旧 parse_bv_input 只抓 BV 号，URL 里的
    `?p=N` 选集参数被直接丢弃，用户想导入 P2 也一律当成 P1，导致选集完全失效。
    这里把 `?p=` 也解析出来一并返回。

    - bvid: 提取到的 BV 号；若文本里没有 BV（如 b23.tv 短链）则原样返回整段文本，
            交给 videos.py 后续步骤区分提示。
    - page: 用户通过 `?p=N` 指定的分P 序号（1-based 正整数）；未指定或语法无效时为
            None。注意：越界（p > 实际总P数）需要拿到 pages 后才能判定，本函数只做
            URL/语法层面的解析，越界判定留在 videos.py。
    - page_invalid: 用户写了 `?p=` 但内容无效（非数字、<=0），用于上层区分
            「用户没指定 p」与「用户指定了 p 但无效」；后者必须拒绝导入，不得回退 P1。
    """
    # Extract the URL before parsing its query. Quotation marks and request prose
    # are not part of a page parameter (e.g. ?p=11」从p=11到23).
    link = re.search(r'https?://[^\s<>"\'，。；、（）()\[\]「」『』“”‘’]+|BV[0-9A-Za-z]+(?:\?[^\s「」『』“”‘’]+)?', text)
    text = (link.group(0) if link else text).strip().rstrip(".,;!")
    if "b23.tv" in text:
        return {"bvid": text, "page": None, "page_invalid": False}

    m = _BV_RE.search(text)
    bvid = m.group(0) if m else text

    page = None
    page_invalid = False
    # 只解析 query 里的 p 参数；path 里的其它片段忽略。
    parsed = urlparse(text)
    if parsed.query:
        qs = parse_qs(parsed.query, keep_blank_values=True)
        if "p" in qs:
            raw_vals = qs["p"]
            raw = raw_vals[0] if raw_vals else ""
            try:
                n = int(raw)
            except (ValueError, TypeError):
                n = None
            # 非数字、空值、重复参数或 <=0：明确标记无效，交给 API 拒绝。
            # 让上层能提示"选集参数无效"。绝不在这里默认填 1 还假装用户没指定。
            if n is None or n < 1 or len(raw_vals) != 1:
                page_invalid = True
            else:
                page = n
    return {"bvid": bvid, "page": page, "page_invalid": page_invalid}


def build_headers(bvid: str) -> dict:
    """CDN 下载/接口请求必需的五个头，缺一不可（否则 412，坑 3）。"""
    return {
        "User-Agent": UA,
        "Referer": f"https://www.bilibili.com/video/{bvid}",
        "Origin": "https://www.bilibili.com",
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }


class BilibiliError(Exception):
    """B站接口返回非 0 code 时抛出，携带服务端 message。"""


class BilibiliClient:
    """B站 Web 端元信息客户端，内置 WBI 签名与密钥缓存。"""

    BASE = "https://api.bilibili.com"
    NAV = "/x/web-interface/nav"
    VIEW = "/x/web-interface/view"
    PLAYURL = "/x/player/wbi/playurl"
    PLAYER = "/x/player/v2"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._session = requests.Session()
        self._key: str | None = None

    def get_wbi_key(self) -> str:
        """获取 wbi 混合密钥，缓存到 self._key。

        坑 4：nav 接口可能返回 code=-101（未登录），但 wbi_img 仍有效，
        因此必须无视 code，直接读 data.wbi_img 提取文件名（去扩展名）后混合。
        """
        if self._key is not None:
            return self._key
        resp = self._session.get(
            self.BASE + self.NAV,
            headers={"User-Agent": UA},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        wbi_img = data["data"]["wbi_img"]
        img_url = wbi_img["img_url"]
        sub_url = wbi_img["sub_url"]
        img_key = urlparse(img_url).path.rsplit("/", 1)[-1].split(".")[0]
        sub_key = urlparse(sub_url).path.rsplit("/", 1)[-1].split(".")[0]
        self._key = mixin_key(img_key, sub_key)
        return self._key

    def get_view(self, bvid: str) -> dict:
        """视频元信息：title/duration/cid/pages/pic。code!=0 抛 BilibiliError。"""
        data = self._get_json(self.VIEW, {"bvid": bvid})
        return data["data"]

    def get_playurl(self, bvid: str, cid: int, quality: int = 64) -> dict:
        """音视频流地址（DASH）。需 WBI 签名；code!=0 抛 BilibiliError。

        quality 默认 64=720P；fnval=16 表示请求 DASH 格式。
        """
        key = self.get_wbi_key()
        params = {"bvid": bvid, "cid": cid, "qn": quality, "fnval": 16}
        data = self._get_json(self.PLAYURL, sign_params(params, key))
        return data["data"]

    def get_subtitles(self, bvid: str, cid: int) -> list:
        """字幕列表（免登录多为空，坑 2）。异常一律吞掉返回 []，绝不影响主流程。"""
        try:
            key = self.get_wbi_key()
            params = {"bvid": bvid, "cid": cid}
            data = self._get_json(self.PLAYER, sign_params(params, key))
            return list(data.get("data", {}).get("subtitle", {}).get("subtitles", []))
        except Exception:
            return []

    def _get_json(self, path: str, params: dict) -> dict:
        """GET 接口并解析 JSON；code!=0 抛 BilibiliError（携带服务端 message）。"""
        resp = self._session.get(
            self.BASE + path,
            params=params,
            headers={"User-Agent": UA},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise BilibiliError(
                f"bilibili API error code={data.get('code')} "
                f"message={data.get('message')}"
            )
        return data
