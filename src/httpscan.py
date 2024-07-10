#!/usr/bin/env python3
import argparse
import ast
import asyncio
import collections
import contextlib
import copy
import dataclasses
import itertools
import json
import logging
import os
import pathlib
import random
import re
import sys
import time
import typing
import urllib.parse
from functools import cached_property

import aiohttp
import aiohttp.abc
import yaml
from aiohttp_socks import ProxyConnector

__version__ = "0.2.1"
__author__ = "s3rgeym"

# При запуске отладчика VS Code устанавливает переменную PYDEVD_USE_FRAME_EVAL=NO
DEBUGGER_ON = any(name.startswith("PYDEVD_") for name in os.environ)

HEADER_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
HEADER_ACCEPT_LANGUAGE = "en-US,en;q=0.9"
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
GOOGLE_REFERER = "https://www.google.com/"

Fail = typing.NewType("Fail", None)

FAIL = Fail(0)


class ANSI:
    CSI = "\x1b["
    RESET = f"{CSI}m"
    RED = f"{CSI}31m"
    GREEN = f"{CSI}32m"
    YELLOW = f"{CSI}33m"
    BLUE = f"{CSI}34m"
    PURPLE = f"{CSI}35m"
    CYAN = f"{CSI}36m"


class ColorHandler(logging.StreamHandler):
    _level_colors = {
        "DEBUG": ANSI.GREEN,
        "INFO": ANSI.YELLOW,
        "WARNING": ANSI.RED,
        "ERROR": ANSI.RED,
        "CRITICAL": ANSI.RED,
    }

    _fmt = logging.Formatter("[ %(asctime)s ] %(levelname)8s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        message = self._fmt.format(record)
        return f"{self._level_colors[record.levelname]}{message}{ANSI.RESET}"


log = logging.getLogger(__name__)
# stderr = functools.partial(print, file=sys.stderr, flush=True)


def parse_range(s: str) -> list[str | int]:
    range_parts = s.split("..", 1)
    is_num_range = all(x.isdigit() for x in range_parts)
    first, last = map((ord, int)[is_num_range], range_parts)
    # first, last = min(first, last), max(first, last)
    res = range(first, last + 1)
    return list(res if is_num_range else map(chr, res))


# Модифицированная версия отсюда https://algo.monster/liteproblems/1096
def expand(s: str) -> set[str]:
    rv = set()

    def dfs(exp: str) -> None:
        try:
            inner, after = exp.split("}", 1)
        except ValueError:
            rv.add(exp)
            return

        before, inner = inner.rsplit("{", 1)

        for item in parse_range(inner) if ".." in inner else inner.split(","):
            dfs(f"{before}{item}{after}")

    dfs(s)

    return rv


@dataclasses.dataclass
class ExpressionExecutor:
    source: str

    def execute(self, vars: dict[str, typing.Any] = {}) -> typing.Any:
        self.vars = copy.deepcopy(vars)
        self.tokens_it = self.tokenize(self.source)
        self.next_tok = None
        self.advance()
        return self.stmt()

    class Token(typing.NamedTuple):
        name: str
        value: str
        pos: int

    TOKEN_PATTERNS: typing.ClassVar[dict[str, str]] = {
        "NULL": r"(?:null|none)",
        "BOOLEAN": r"(?:true|false)",
        "ID": r"[a-z_][a-z0-9_]*",
        "NUMBER": r"[-+]?\d+(\.\d+)?",
        "STRING": r'(?:"(?:\\"|[^"])*"|\'(?:\\\'|[^\'])*\')',
        "COMPARE": r"(?:[=!]=|[<>]=?)",
        "NOT": r"(?:not|!)",
        "AND": r"(?:and|&&)",
        "OR": r"(?:or|\|\|)",
        "LPAREN": r"\(",
        "RPAREN": r"\)",
        "SPACE": r"\s+",
        "INVALID_CHAR": r".",
        "END": r"$",
    }

    @cached_property
    def tokens_re(self) -> re.Pattern:
        return re.compile(
            "|".join(f"(?P<{k}>{v})" for k, v in self.TOKEN_PATTERNS.items()),
            re.IGNORECASE,
        )

    def tokenize(self, s: str) -> typing.Iterable[Token]:
        for m in self.tokens_re.finditer(s):
            for k, v in m.groupdict().items():
                if v is not None:
                    if k != "SPACE":
                        yield self.Token(k, v, m.start())
                    break

    def token(self) -> Token:
        return next(self.tokens_it, None)

    def advance(self) -> None:
        self.cur_tok, self.next_tok = self.next_tok, self.token()

    def match(self, tok: str) -> bool:
        if self.next_tok.name == tok:
            self.advance()
            return True
        return False

    def unexpected_next_token(self, expected: str | None = None) -> None:
        message = (
            f"unexpected token {self.next_tok.value!r} at position {self.next_tok.pos}"
            + ("; expected: " + expected if expected else "")
        )
        raise ValueError(message)

    def expect(self, tok: str) -> None:
        if not self.match(tok):
            self.unexpected_next_token(tok)

    # https://docs.python.org/3/reference/expressions.html#operator-precedence
    def expr(self) -> typing.Any:
        rv = self.and_()
        while self.match("OR"):
            # выражение справа от or не выполнится, если левое ИСТИНА!!!
            rhv = self.and_()
            rv = rv or rhv
        return rv

    def and_(self) -> typing.Any:
        rv = self.compare()
        while self.match("AND"):
            rhv = self.compare()
            rv = rv and rhv
        return rv

    def compare(self) -> typing.Any:
        # приоритет должен быть больше чем у операторов сравнения
        if self.match("NOT"):
            return not self.compare()

        # операции типа сложения/вычитания не под-ся
        rv = self.primary()
        while self.match("COMPARE"):
            # try:
            match self.cur_tok.value:
                case ">":
                    rv = rv > self.primary()
                case "<":
                    rv = rv < self.primary()
                case ">=":
                    rv = rv >= self.primary()
                case "<=":
                    rv = rv <= self.primary()
                case "==":
                    rv = rv == self.primary()
                case "!=":
                    rv = rv != self.primary()
                case _:
                    raise ValueError(self.cur_tok.value)
            # # 1 < '2'
            # except TypeError:
            #     rv = False
        return rv

    def primary(self) -> typing.Any:
        if self.match("LPAREN"):
            rv = self.expr()
            self.expect("RPAREN")
            return rv

        if self.match("NULL"):
            return None

        if self.match("BOOLEAN"):
            return self.cur_tok.value.lower() == "true"

        if self.match("ID"):
            return self.vars.get(self.cur_tok.value)

        if self.match("NUMBER"):
            return (int, float)["." in self.cur_tok.value](self.cur_tok.value)

        if self.match("STRING"):
            return ast.literal_eval(self.cur_tok.value)

        self.unexpected_next_token()

    def stmt(self) -> typing.Any:
        rv = self.expr()
        self.expect("END")
        return rv


def execute(s: str, vars: dict[str, typing.Any]) -> typing.Any:
    return ExpressionExecutor(s).execute(vars)


# def parse_header(h: str) -> tuple[str, dict]:
#     from email.message import Message

#     message = Message()
#     message["content-type"] = h
#     params = message.get_params()
#     return params[0][0], dict(params[1:])


class ProbeConfig(typing.TypedDict):
    name: str
    path: str
    method: typing.NotRequired[
        typing.Literal["GET", "HEAD", "POST", "PUT", "DELETE"]
    ]
    params: typing.NotRequired[dict]
    headers: typing.NotRequired[dict]
    cookies: typing.NotRequired[dict]
    data: typing.NotRequired[dict]
    json: typing.NotRequired[dict]
    match: typing.NotRequired[str]
    not_match: typing.NotRequired[str]
    extract: typing.NotRequired[str]
    extract_all: typing.NotRequired[str]
    # status_code >= 200 && status_code < 300
    condition: typing.NotRequired[str]
    save_to: typing.NotRequired[os.PathLike]


class Config(typing.TypedDict):
    workers_num: typing.NotRequired[int]
    timeout: typing.NotRequired[int | float]
    max_host_error: typing.NotRequired[int]
    probes: typing.NotRequired[list[ProbeConfig]]
    ignore_hosts: typing.NotRequired[list[str]]
    proxy_url: typing.NotRequired[str]


class CloudflareChallenge(typing.NamedTuple):
    action: str
    method: str
    param_name: str
    var_east: str
    var_west: str


@dataclasses.dataclass
class Scanner:
    probes: list[ProbeConfig]
    _: dataclasses.KW_ONLY
    output: typing.TextIO = sys.stdout
    workers_num: int = 10
    timeout: float = 10.0
    delay: float = 0.150
    max_host_error: int = 10
    proxy_url: str | None = None
    ignore_hosts: set[str] = dataclasses.field(default_factory=set)

    def __post_init__(self) -> None:
        self.lock = asyncio.Lock()

    async def scan(self, urls: typing.Iterable[str]) -> None:
        if self.proxy_url and not (await self.check_proxy()):
            raise ValueError("ip leak detected!")

        self.scan_queue = asyncio.Queue(maxsize=self.workers_num)
        self.error_counter = collections.Counter()
        self.next_request = 0

        # Если `asyncio.TaskGroup()` первым идет, то падает `RuntimeError: Session is closed`
        async with (
            self.get_session(
                proxy_url=self.proxy_url,
                timeout=self.timeout,
            ) as self.session,
            asyncio.TaskGroup() as tg,
        ):
            user_agent = await self.rand_user_agent()
            log.debug("random user agent: %r", user_agent)
            self.session.headers["User-Agent"] = user_agent

            tg.create_task(self.produce(urls))

            for _ in range(self.workers_num):
                tg.create_task(self.worker_scan())

            tg.create_task(self.stop_scanning())

    # TODO: переименовать
    async def produce(self, urls: typing.Iterable[str]) -> None:
        for url in urls:
            # .hostname переводится в нижний регистр
            if self.is_ignored_host(urllib.parse.urlsplit(url).hostname):
                log.debug(f"skip ignored host url: {url}")
                continue

            log.debug(f"substitute paths to {url}")

            for probe_conf in self.probes:
                for path in expand(probe_conf["path"]):
                    await self.scan_queue.put(
                        (
                            urllib.parse.urljoin(url, path),
                            probe_conf,
                        )
                    )

    async def stop_scanning(self) -> None:
        await self.scan_queue.join()

        log.debug("stop all scanning workers")

        for _ in range(self.workers_num):
            self.scan_queue.put_nowait((None, None))

    async def worker_scan(self) -> None:
        while True:
            url, conf = await self.scan_queue.get()

            if url is None:
                break

            try:
                netloc = urllib.parse.urlsplit(url).netloc

                if self.error_counter[netloc] >= self.max_host_error:
                    log.warning(f"max host error: {netloc}")
                    continue

                headers = conf.get("headers", {}).copy()
                response = await self.probe_request(url, headers, conf)

                # TODO: Я не смог добраться до ip сервера
                # try:
                #     server_addr, _ = response.connection.transport.get_extra_info(
                #         "peername"
                #     )
                # except:  # noqa: E722
                #     server_addr = None

                # Сделать удаление кук для доменов отличных от текущего?
                # список доменов:
                # > set(v['domain'] for v in self.session.cookie_jar)
                # удалить все куки для домена:
                # > self.session.cookie_jar.clear_domain(domain)

                if challenge := await self.detect_cloudflare_challenge(
                    response
                ):
                    log.debug(f"cloudflare challenge detected: {url}")

                    # разгадываем скобки и возвращаем запрашиваемую страницу
                    response = await self.bypass_cloudflare_challenge(
                        challenge,
                        response,
                        headers,
                    )

                if (result := await self.do_probe(response, conf)) is FAIL:
                    continue

                self.output_json(
                    remove_empty_from_dict(
                        {
                            "url": url,
                            "host": netloc,
                            "http_version": f"{response.version.major}.{response.version.minor}",
                            "status_code": response.status,
                            "status_reason": response.reason,
                            "content_length": response.content_length,
                            "content_type": response.content_type,
                            "content_charset": response.charset,
                            "response_headers": dict(response.headers),
                            "probe_name": conf["name"],
                            **result,
                        }
                    )
                )
            except Exception as ex:
                log.error(ex, exc_info=DEBUGGER_ON)
                self.error_counter[netloc] += 1
            finally:
                self.scan_queue.task_done()

    def is_ignored_host(self, hostname: str) -> bool:
        hostname_parts = hostname.split(".")

        # www.linux.org.ru => {'*.linux.org.ru', '*.org.ru', '*.ru', 'www.linux.org.ru'}
        hostname_wildcards = set(
            [hostname]
            + [
                ".".join(["*"] + hostname_parts[i:])
                for i in range(1, len(hostname_parts))
            ]
        )

        return bool(hostname_wildcards & self.ignore_hosts)

    async def sleep_delay(self) -> None:
        if self.delay > 0:
            async with (
                self.lock
            ):  # блокируем асинхронное выполнение остальных заданий
                if (dt := self.next_request - time.monotonic()) > 0:
                    # log.debug(f"sleep: {dt:.3f}")
                    await asyncio.sleep(dt)

                self.next_request = time.monotonic() + self.delay

    async def solve_cloudflare_challenge(
        self, challenge: CloudflareChallenge
    ) -> int:
        try:
            return int(
                await check_output(
                    "node",
                    "-e",
                    f"console.log({challenge.var_east} + {challenge.var_west})",
                )
            )
        except FileNotFoundError as ex:
            raise RuntimeError(
                "Node.js must be installed to solve cloudflare challenge!"
            ) from ex

    async def bypass_cloudflare_challenge(
        self,
        challenge: CloudflareChallenge,
        origin_response: aiohttp.ClientResponse,
        additional_headers: dict[str, str],
    ) -> aiohttp.ClientResponse:
        assert challenge.method == "get", f"unexptected {challenge.method =}"
        solution = await self.solve_cloudflare_challenge(challenge)

        log.debug(f"{solution =}")

        payload = {challenge.param_name: solution}
        challenge_endpoint = urllib.parse.urljoin(
            str(origin_response.url), challenge.action
        )

        challenge_response = await self.session.get(
            url=challenge_endpoint,
            params=payload,
            headers=additional_headers | {"Referer": str(origin_response.url)},
        )

        assert (
            len(challenge_response.history) == 1
            and challenge_response.history[0].status == 301
            # 'Set-Cookie': 'wschkid=ae96fb3bf715d463c7f3328d2e4377cb9aa6b155.1720572164.1; Expires=Thu, 08-Aug-24 00:42:44 GMT; Domain=<censored>; Path=/; HttpOnly; SameSite=Lax'
            and challenge.param_name
            in challenge_response.history[0].headers.get("Set-Cookie", "")
            and challenge_response.url == origin_response.url
        ), f"can't bypass challenge: {origin_response.url}"

        return challenge_response

    async def detect_cloudflare_challenge(
        self, response: aiohttp.ClientResponse
    ) -> typing.Optional[CloudflareChallenge]:
        # {"Cache-Control": "private, no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0, s-maxage=0", "Connection": "close", "Content-Type": "application/zip", "Date": "Mon, 08 Jul 2024 02:01:26 GMT", "Last-Modified": "Monday, 08-Jul-2024 02:01:26 GMT", "Server": "imunify360-webshield/1.21", "Transfer-Encoding": "chunked", "cf-edge-cache": "no-cache"}

        # if not (
        #     response.headers.get("Cache-Control")
        #     == "private, no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0, s-maxage=0"
        #     and response.headers.get("cf-edge-cache") == "no-cache"
        # ):
        #     return

        # {"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0", "Connection": "keep-alive", "Content-Length": "1570", "Date": "Mon, 08 Jul 2024 16:10:57 GMT", "Server": "imunify360-webshield/1.21"}

        # no-cache в заголовке Cache-Control содержится как правило на страницах, формирующихся динамически,
        # а там всегда какой-то текст

        # Так вроде же имеется заголовок `Transfer-Encoding: chunked`...
        if "no-cache" not in response.headers.get("Cache-Control", "").split(
            ", "
        ):
            return

        text = await response.text()

        # TODO: когда-нибудь сделать обход и капчи
        # <title>Captcha</title>

        """
        window.check = function (value) {
            $.ajax({
                type: 'POST',
                url: '/captchacheck',
                data: {
                    'captcha_value': value
                }
            }).done(function (data) {
                if (data['success']) {
                    captchaIsPassed = true;
                    $("#text").next("p").text("");
                    $("#text").text(
                        "IP XXX.XXX.XX.XXX has been unblocked"
                    );
                    setTimeout(do_reload, 1000);
                } else {
                    grecaptcha.reset();
                }
            }).fail(function () {
                grecaptcha.reset();
            });
        };
        """

        """
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="robots" content="noindex, nofollow">
            <title>One moment, please...</title>
            <style>
            body {
                background: #F6F7F8;
                color: #303131;
                font-family: sans-serif;
                margin-top: 45vh;
                text-align: center;
            }
            </style>
            </head>
        <body>
            <h1>Please wait while your request is being verified...</h1>
            <form id="wsidchk-form" style="display:none;" action="/z0f76a1d14fd21a8fb5fd0d03e0fdc3d3cedae52f" method="GET">
            <input type="hidden" id="wsidchk" name="wsidchk"/>
            </form>
            <script>
            (function(){
                var west=+((+!+[]+!![]+!![]+!![]+!![]+!![]+!![])+(+!+[]+!![]+!![]+!![]+!![]+[])+(+!+[]+!![]+!![]+!![]+!![]+!![]+!![])+(+!+[]+!![]+!![]+!![]+!![]+!![]+!![]+!![]+!![]+[])+(+!+[]+!![]+!![]+!![]+!![]+!![])+(+!+[]+!![]+[])+(+!+[]+!![]+!![]+!![]+!![]+!![])),
                    east=+((+!+[])+(+!+[]+!![]+!![]+!![]+!![]+[])+(+!+[]+!![]+!![]+!![]+!![]+!![])+(+!+[]+[])+(+!+[]+!![]+!![]+!![]+!![]+!![]+!![])+(+!+[]+!![]+!![]+!![]+!![]+!![]+!![]+[])+(+!+[]+!![]+!![]+!![]+!![]+!![]+!![]+!![])+(+![]+[])),
                    x=function(){try{return !!window.addEventListener;}catch(e){return !!0;} },
                    y=function(y,z){x() ? document.addEventListener('DOMContentLoaded',y,z) : document.attachEvent('onreadystatechange',y);};
                y(function(){
                    document.getElementById('wsidchk').value = west + east;
                    document.getElementById('wsidchk-form').submit();
                }, false);
            })();
            </script>
        </body>
        </html>
        """

        if "<title>One moment, please...</title>" not in text:
            return

        assert "west + east" in text

        js_vars = dict(re.findall(r"(west|east)=([^,]+)", text))

        return CloudflareChallenge(
            action=re.search(r'action="([^"]+)', text).group(1),
            method=re.search(r'method="([^"]+)"', text).group(1).lower(),
            param_name=re.search(
                r'<input type="hidden".+?name="([^"]+)',
                text,
            ).group(1),
            var_east=js_vars["east"],
            var_west=js_vars["west"],
        )

    async def probe_request(
        self,
        url: str,
        headers: dict[str, str],
        probe_conf: ProbeConfig,
    ) -> aiohttp.ClientResponse:
        await self.sleep_delay()

        method = probe_conf.get("method", "GET").upper()

        response = await self.session.request(
            method,
            url,
            headers=headers,
            params=probe_conf.get("params"),
            data=probe_conf.get("data"),
            json=probe_conf.get("json"),
            cookies=probe_conf.get("cookies"),
            allow_redirects=False,  # игнорируем редиректы
        )

        log.debug(f"{response.status} - {response.method} - {response.url}")

        return response

    async def rand_user_agent(self) -> str:
        return random.choice(await self.get_user_agents())

    async def do_probe(
        self,
        response: aiohttp.ClientResponse,
        conf: ProbeConfig,
    ) -> dict[str, typing.Any] | Fail:
        rv = {}

        if "condition" in conf:
            # уже распарсенный
            # mime_type, _ = parse_header(response.content_type)

            vars_dict = {
                "status_code": response.status,
                "content_length": response.content_length,
                "content_type": response.content_type,
            }

            if not execute(conf["condition"], vars_dict):
                return FAIL

        if "match" in conf:
            text = await response.text()
            if not re.search(conf["match"], text):
                return FAIL

        if "not_match" in conf:
            text = await response.text()
            if re.search(conf["not_match"], text):
                return FAIL

        if "extract" in conf:
            text = await response.text()
            if match := re.search(conf["extract"], text):
                rv |= {"match": match.group()}
            else:
                return FAIL

        if "extract_all" in conf:
            text = await response.text()
            if items := re.findall(conf["extract_all"], text):
                rv |= {"matches": items}
            else:
                return FAIL

        if "save_to" in conf:
            save_path = (
                pathlib.Path(conf["save_to"])
                / response.host
                / (
                    response.url.path[1:]
                    + ("", "index.html")[response.url.path.endswith("/")]
                )
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)

            with save_path.open("wb") as f:
                if response._body is None:
                    async for data in response.content.iter_chunked(2**16):
                        f.write(data)
                else:
                    f.write(response._body)

                log.debug(f"saved: {save_path}")

            stat = save_path.stat()

            if stat.st_size == 0:
                log.warning(f"unlink empty file: {save_path}")
                save_path.unlink()
                return FAIL

            rv |= {
                "saved_bytes": stat.st_size,
                "saved_as": str(save_path.resolve()),
            }

        return rv

    def output_json(self, obj: typing.Any, **kw: typing.Any) -> None:
        js = json.dumps(obj, ensure_ascii=False, sort_keys=True, **kw)
        print(js, file=self.output, flush=True)

    async def get_user_agents(self) -> list[str]:
        if not hasattr(self, "_user_agents"):
            log.debug("get user agents")
            async with self.get_session() as session:
                r = await session.get(
                    "https://useragents.io/random/__data.json?limit=100"
                )
                json_data = await r.json()
            self._user_agents = json_data["nodes"][1]["data"][2:-1][1::4]
            assert self._user_agents
        return list(self._user_agents)

    async def check_proxy(self) -> bool:
        client_ip = await self.get_ip()
        log.debug(f"client ip: {mask_ip(client_ip)}")
        proxy_ip = await self.get_ip(proxy_url=self.proxy_url)
        log.debug(f"proxy ip: {mask_ip(proxy_ip)}")
        return client_ip != proxy_ip

    async def get_ip(self, **kw: typing.Any) -> str:
        async with self.get_session(**kw) as session:
            async with session.get("https://api.ipify.org") as response:
                return await response.text()

    @contextlib.asynccontextmanager
    async def get_session(
        self,
        *,
        timeout: float | aiohttp.ClientTimeout | None = None,
        proxy_url: str | None = None,
        cookie_jar: aiohttp.abc.AbstractCookieJar | None = None,
    ) -> typing.AsyncIterator[aiohttp.ClientSession]:
        if isinstance(timeout, (int, float)):
            timeout = aiohttp.ClientTimeout(
                total=None,
                sock_connect=timeout,
                sock_read=timeout,
            )

        connector = None

        if proxy_url:
            connector = ProxyConnector.from_url(proxy_url)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            cookie_jar=cookie_jar,
        ) as session:
            session.headers.update(self.get_default_headers())
            yield session

    def get_default_headers(self) -> dict[str, str]:
        return {
            "Accept": HEADER_ACCEPT,
            "Accept-Language": HEADER_ACCEPT_LANGUAGE,
            "Referer": GOOGLE_REFERER,
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": DEFAULT_USER_AGENT,
        }


def mask_ip(addr: str, ch: str = "*") -> str:
    """маскирует все сегменты адреса за исключением последнего

    >>> mask_ip('192.168.0.104')
    '***.***.*.104'"""
    return re.sub(r"[^.](?![^.]*$)", ch, addr)


def remove_empty_from_dict(d: dict) -> dict:
    return {k: v for k, v in d.items() if v}


async def check_output(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    p = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs,
    )
    stdout_data, stderr_data = await p.communicate()
    if p.returncode == 0:
        return stdout_data


def normalize_url(u: str) -> str:
    return u if "://" in u else f"https://{u}"


def find_config() -> None | typing.TextIO:
    config_name = pathlib.Path(__file__).stem
    for config_directory in [
        pathlib.Path.cwd(),
        pathlib.Path(os.environ["XDG_CONFIG_HOME"])
        if "XDG_CONFIG_HOME" in os.environ
        else pathlib.Path.home() / ".config",
    ]:
        for ext in ("yml", "yaml"):
            path = config_directory / f"{config_name}.{ext}"
            if path.exists():
                return path.open()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="configurable http scanner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-u",
        "--url",
        dest="urls",
        help="target url to probe(s)",
        default=[],
        nargs="+",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="input file with urls",
        type=argparse.FileType(),
        default="-",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output file to results in JSONL",
        type=argparse.FileType("w"),
        default="-",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="config file in YAML format",
        type=argparse.FileType(),
    )
    parser.add_argument(
        "-igh",
        "--ignore-hosts",
        "--ignore",
        help="ignore hosts file",
        type=argparse.FileType(),
    )
    parser.add_argument(
        "-w",
        "--workers-num",
        "--workers",
        help="number of workers",
        type=int,
        default=30,
    )
    parser.add_argument(
        "-t",
        "--timeout",
        help="request timeout",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "-d",
        "--delay",
        help="delay before each request in seconds",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "-maxhe",
        "--max-host-error",
        help="maximum number of errors for a host after which other paths will be skipped",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--proxy-url",
        "--proxy",
        help="proxy url, eg `socks5://localhost:1080`",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        help="be more verbosity",
        action="count",
        default=0,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: typing.Sequence[str] | None = None) -> None | int:
    parser = create_parser()
    args = parser.parse_args(argv)

    lvl = max(logging.DEBUG, logging.WARNING - logging.DEBUG * args.verbosity)
    log.setLevel(lvl)
    log.addHandler(ColorHandler())

    log.debug("debugger: %s", ["off", "on"][DEBUGGER_ON])

    if (config_file := args.config or find_config()) is None:
        log.error("config not found")
        return 1

    conf: Config = yaml.safe_load(config_file)
    config_file.close()
    log.debug(f"config loaded: {config_file.name}")
    probes = conf["probes"]
    # > {'baz', 'foo', 'bar', 'quix'} > {'bar', 'foo'}
    # True
    if not all(set(item) > ProbeConfig.__required_keys__ for item in probes):
        log.error(
            f"invalid config: probe required keys: {', '.join(ProbeConfig.__required_keys__)}"
        )
        return 1

    urls: list[str] = args.urls

    if not (args.input.isatty() and urls):
        urls: itertools.chain[str] = itertools.chain(
            urls, map(str.strip, args.input)
        )

    urls: map[str] = map(normalize_url, filter(None, urls))

    # in set значительно быстрее
    ignore_hosts: set[str] = set(
        map(str.lower, filter(None, map(str.strip, args.ignore_hosts)))
        if args.ignore_hosts
        else []
    )

    log.debug("ignored hosts: %d", len(ignore_hosts))

    scanner = Scanner(
        probes=probes,
        output=args.output,
        ignore_hosts=conf.get("ignore_hosts", ignore_hosts),
        workers_num=conf.get("workers_num", args.workers_num),
        timeout=conf.get("timeout", args.timeout),
        delay=conf.get("delay", args.delay),
        max_host_error=conf.get("max_host_error", args.max_host_error),
        proxy_url=conf.get("proxy_url", args.proxy_url),
    )

    try:
        asyncio.run(scanner.scan(urls))
    except KeyboardInterrupt:
        log.warning("execution interrupted by user")
        return 2
    except Exception as ex:
        log.critical(ex, exc_info=True)
        return 1
    else:
        log.info("finished!")


if __name__ == "__main__":
    sys.exit(main())
