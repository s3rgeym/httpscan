#!/usr/bin/env python3
from __future__ import annotations

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

__version__ = "0.3.2"
__author__ = "Sergey M"

# При запуске отладчика VS Code устанавливает переменную PYDEVD_USE_FRAME_EVAL=NO
DEBUGGER_ON = any(name.startswith("PYDEVD_") for name in os.environ)

HEADER_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
HEADER_ACCEPT_LANGUAGE = "en-US,en;q=0.9"
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
GOOGLE_REFERER = "https://www.google.com/"

USER_AGENTS_ENDPOINT = "https://useragents.io/random/__data.json?limit=1000"


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


logger = logging.getLogger(__name__)
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


class ProbeDict(typing.TypedDict):
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


class ConfigDict(typing.TypedDict):
    workers_num: typing.NotRequired[int]
    timeout: typing.NotRequired[int | float]
    connect_timeout: typing.NotRequired[int | float]
    read_timeout: typing.NotRequired[int | float]
    # proxy_timeout: typing.NotRequired[int | float]
    max_host_error: typing.NotRequired[int]
    probes: typing.NotRequired[list[ProbeDict]]
    ignore_hosts: typing.NotRequired[list[str]]
    proxy_url: typing.NotRequired[str]


@dataclasses.dataclass
class CloudflareChallenge:
    """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="robots" content="noindex, nofollow">
        <title>One moment, please...</title>
        <!-- ... -->
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

    action: str
    method: str
    param: str
    east: str
    west: str

    @classmethod
    def from_text(
        cls: typing.Type[CloudflareChallenge], text: str
    ) -> CloudflareChallenge:
        assert "west + east" in text

        js_vars = dict(re.findall(r"(west|east)=([^,]+)", text))

        return cls(
            action=re.search(r'action="([^"]+)', text).group(1),
            method=re.search(r'method="([^"]+)"', text).group(1).lower(),
            param=re.search(
                r'<input type="hidden".+?name="([^"]+)',
                text,
            ).group(1),
            east=js_vars["east"],
            west=js_vars["west"],
        )


@dataclasses.dataclass
class Scanner:
    probes: list[ProbeDict]
    _: dataclasses.KW_ONLY
    output: typing.TextIO = sys.stdout
    workers_num: int = 10
    timeout: int | float | None = None
    connect_timeout: int | float = 10.0
    read_timeout: int | float = 5.0
    delay: int = 120
    max_host_error: int = 10
    proxy_url: str | None = None
    ignore_hosts: typing.Iterable[str] | None = None
    follow_redirects: bool = False
    # читаем и проверяем только первые 256kb
    probe_read_length: int = 1 << 18

    def __post_init__(self) -> None:
        self.lock = asyncio.Lock()
        # переводим в нижний регистр
        self.ignore_hosts: set[str] = set(
            map(str.lower, self.ignore_hosts or [])
        )
        if self.proxy_url is None:
            self.proxy_url = os.getenv("PROXY_URL")

    async def scan(self, urls: typing.Iterable[str]) -> None:
        if self.proxy_url and not (await self.check_proxy()):
            raise ValueError("ip leak detected!")

        self.queue = asyncio.Queue(maxsize=self.workers_num)
        self.sem = asyncio.Semaphore(self.workers_num)
        self.host_error = collections.Counter()
        self.next_request = 0

        # Генерирует сранные GroupException, которые сложно перехватывать выше
        # по стеку вызова

        # async with asyncio.TaskGroup() as tg:
        #     for _ in range(self.workers_num):
        #         tg.create_task(self.worker())

        #     await self.produce(urls)
        #     await self.queue.join()

        #     logger.debug("stop workers")

        #     for _ in range(self.workers_num):
        #         self.queue.put_nowait(None)

        workers = [
            asyncio.create_task(self.worker(), name=f"worker_{i}")
            for i in range(self.workers_num)
        ]

        await self.produce(urls)
        await self.queue.join()

        for worker in workers:
            worker.cancel()

    async def produce(self, urls: typing.Iterable[str]) -> None:
        for url in map(normalize_url, urls):
            try:
                # hostname всегда в нижнем регистре
                if self.is_ignored_host(urllib.parse.urlsplit(url).hostname):
                    logger.debug(f"skip ignored host url: {url}")
                    continue

                await self.queue.put(url)
            except Exception as ex:
                logger.exception(ex)

    async def worker(self) -> None:
        task_name = asyncio.current_task().get_name()
        logger.debug("started: %s", task_name)

        # без этой строки отработает как надо, но не выведет сообщения о
        # завершении
        with contextlib.suppress(asyncio.CancelledError):
            while True:
                url = await self.queue.get()

                try:
                    user_agent = await self.rand_user_agent()
                    logger.debug(f"user agent for {url}: {user_agent}")

                    # Для каждого url используем новую сессию из-за того, что сессии
                    # со временем начинают тормозить
                    async with self.get_session(
                        user_agent=user_agent
                    ) as session:
                        probe_tasks = []

                        for probe in self.probes:
                            for path in expand(probe["path"]):
                                probe_tasks.append(
                                    self.make_probe(session, url, path, probe)
                                )

                        logger.debug(
                            f"probe tasks for {url}: {len(probe_tasks)}"
                        )

                        await asyncio.gather(
                            *probe_tasks, return_exceptions=True
                        )
                # ! asyncio.exceptions.CancelledError наследуется напрямую от
                # BaseException, поэтому нужно перехватывать Exception
                except Exception as ex:
                    logger.exception(ex)
                finally:
                    self.queue.task_done()

        logger.debug("finished: %s", task_name)

    async def make_probe(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        path: str,
        probe: ProbeDict,
    ) -> None:
        # параллельно будет совершаться не более workers_num запросов
        async with self.sem:
            try:
                host = urllib.parse.urlparse(base_url).netloc

                if self.host_error[host] >= self.max_host_error:
                    logger.warning(f"max host error exceeded: {host}")
                    return

                url = urllib.parse.urljoin(base_url, path)
                headers = probe.get("headers", {}).copy()

                response = await self.send_probe_request(
                    session, url, headers, probe
                )

                content: bytes = await response.content.read(
                    self.probe_read_length
                )

                text_content: str = content.decode(
                    response.charset or "utf-8", errors="replace"
                )

                if "<title>One moment, please...</title>" in text_content:
                    logger.debug(f"cloudflare challenge detected: {url}")

                    challenge = CloudflareChallenge.from_text(text_content)

                    # разгадываем скобки и возвращаем запрашиваемую страницу
                    response = await self.bypass_cloudflare_challenge(
                        session,
                        challenge,
                        response,
                        headers,
                    )

                if (
                    result := await self.get_probe_result(
                        response,
                        content,
                        text_content,
                        probe,
                    )
                ) is FAIL:
                    return

                self.output_json(
                    remove_empty_from_dict(
                        {
                            "url": str(response.url),
                            "input": base_url,
                            "host": response.url.host,
                            "http_version": f"{response.version.major}.{response.version.minor}",
                            "status_code": response.status,
                            "status_reason": response.reason,
                            "content_length": response.content_length,
                            "content_type": response.content_type,
                            "content_charset": response.charset,
                            "response_headers": dict(response.headers),
                            "probe_name": probe["name"],
                            **result,
                        }
                    )
                )
            except Exception as ex:
                logger.warning(ex)
                self.host_error[host] += 1

    def is_ignored_host(self, hostname: str) -> bool:
        hostname_parts = hostname.split(".")

        # www.linux.org.ru => {'*.linux.org.ru', '*.org.ru', '*.ru', 'www.linux.org.ru'}
        hostname_wildcards = set(
            [
                ".".join(["*"] + hostname_parts[i:])
                for i in range(1, len(hostname_parts))
            ]
            + [hostname]
        )

        return bool(hostname_wildcards & self.ignore_hosts)

    async def sleep_delay(self) -> None:
        if self.delay > 0:
            async with (
                self.lock
            ):  # блокируем асинхронное выполнение остальных заданий
                if (dt := self.next_request - time.monotonic()) > 0:
                    # logger.debug(f"sleep: {dt:.3f}")
                    await asyncio.sleep(dt)

                self.next_request = time.monotonic() + self.delay / 1000

    async def solve_cloudflare_challenge(
        self, challenge: CloudflareChallenge
    ) -> int:
        try:
            return int(
                await check_output(
                    "node",
                    "-e",
                    f"console.log({challenge.west} + {challenge.east})",
                )
            )
        except FileNotFoundError as ex:
            raise RuntimeError(
                "Node.js must be installed to solve cloudflare challenge!"
            ) from ex

    async def bypass_cloudflare_challenge(
        self,
        session: aiohttp.ClientSession,
        challenge: CloudflareChallenge,
        origin_response: aiohttp.ClientResponse,
        additional_headers: dict[str, str],
    ) -> aiohttp.ClientResponse:
        assert challenge.method == "get", f"unexptected {challenge.method = !r}"
        solution = await self.solve_cloudflare_challenge(challenge)

        logger.debug(f"{solution = }")

        payload = {challenge.param: solution}
        challenge_endpoint = urllib.parse.urljoin(
            str(origin_response.url), challenge.action
        )

        challenge_response = await session.get(
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

    async def send_probe_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: dict[str, str],
        probe: ProbeDict,
    ) -> aiohttp.ClientResponse:
        await self.sleep_delay()

        method = probe.get("method", "GET").upper()

        logger.debug(f"send request: {method} {url}")

        response = await session.request(
            method,
            url,
            headers=headers,
            params=probe.get("params"),
            data=probe.get("data"),
            json=probe.get("json"),
            cookies=probe.get("cookies"),
            allow_redirects=self.follow_redirects,
        )

        logger.debug(
            f"got response: {response.status} {response.method} {response.url}"
        )

        return response

    async def rand_user_agent(self) -> str:
        return random.choice(await self.get_user_agents())

    async def get_probe_result(
        self,
        response: aiohttp.ClientResponse,
        text_content: str,
        content: bytes,
        conf: ProbeDict,
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
            if not re.search(conf["match"], text_content):
                return FAIL

        if "not_match" in conf:
            if re.search(conf["not_match"], text_content):
                return FAIL

        if "extract" in conf:
            if match := re.search(conf["extract"], text_content):
                rv |= {"match": match.group()}
            else:
                return FAIL

        if "extract_all" in conf:
            if items := re.findall(conf["extract_all"], text_content):
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
                f.write(content)

                # читаем данные блоками по 64кб, если они остались в сокете
                async for data in response.content.iter_chunked(1 << 16):
                    f.write(data)

                logger.debug(f"saved: {save_path}")

            stat = save_path.stat()

            if stat.st_size == 0:
                logger.warning(f"unlink empty file: {save_path}")
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
            logger.debug("get user agents from %s", USER_AGENTS_ENDPOINT)
            async with self.get_session() as session:
                r = await session.get(USER_AGENTS_ENDPOINT)
                json_data = await r.json()
            self._user_agents = json_data["nodes"][1]["data"][2:-1][1::4]
            assert self._user_agents
        return list(self._user_agents)

    async def check_proxy(self) -> bool:
        client_ip = await self.get_ip(use_proxy=False)
        logger.debug(f"client ip: {mask_ip(client_ip)}")
        proxy_ip = await self.get_ip()
        logger.debug(f"proxy ip: {mask_ip(proxy_ip)}")
        return client_ip != proxy_ip

    async def get_ip(self, **kw: typing.Any) -> str:
        async with self.get_session(**kw) as session:
            async with session.get("https://api.ipify.org") as response:
                return await response.text()

    @contextlib.asynccontextmanager
    async def get_session(
        self,
        *,
        user_agent: str | None = None,
        use_proxy: bool = True,
        headers: dict[str, str] = {},
        **kwargs: typing.Any,
    ) -> typing.AsyncIterator[aiohttp.ClientSession]:
        con = (
            ProxyConnector.from_url(self.proxy_url, limit=0)
            if use_proxy and self.proxy_url
            else aiohttp.TCPConnector(limit=0)
        )

        tmt = aiohttp.ClientTimeout(
            total=self.timeout,
            sock_connect=self.connect_timeout,
            sock_read=self.read_timeout,
        )

        async with aiohttp.ClientSession(
            connector=con,
            timeout=tmt,
            **kwargs,
        ) as session:
            session.headers.update(
                {
                    "Accept": HEADER_ACCEPT,
                    "Accept-Language": HEADER_ACCEPT_LANGUAGE,
                    "Referer": GOOGLE_REFERER,
                    # перебрасывает с http:// на https://
                    # "Upgrade-Insecure-Requests": "1",
                    "User-Agent": user_agent or DEFAULT_USER_AGENT,
                    **headers,
                }
            )
            yield session


def normalize_url(u: str) -> str:
    return u if "://" in u else f"https://{u}"


def mask_ip(addr: str, ch: str = "*") -> str:
    """маскирует все сегменты адреса за исключением последнего

    >>> mask_ip("192.168.0.104")
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
    stdout_data, _ = await p.communicate()
    if p.returncode == 0:
        return stdout_data


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


class NameSpace(argparse.Namespace):
    urls: list[str]
    input: typing.TextIO
    output: typing.TextIO
    config: typing.TextIO
    ignore_hosts: typing.TextIO
    workers_num: int
    timeout: int | float
    connect_timeout: int | float
    read_timeout: int | float
    delay: int | float
    max_host_error: int
    proxy_url: str
    follow_redirects: bool
    probe_read_length: int
    verbosity: int


# sequence = list | tuple
def parse_args(
    argv: typing.Sequence[str] | None = None,
) -> tuple[argparse.ArgumentParser, NameSpace]:
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
        default=20,
    )
    parser.add_argument("-t", "--timeout", help="total timeout sec", type=float)
    parser.add_argument(
        "-r",
        "--read-timeout",
        "--socket-read",
        "--read",
        help="socket read timeout sec",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "-C",
        "--connect-timeout",
        "--socket-connect",
        "--connect",
        help="socket read timeout sec",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "-d",
        "--delay",
        help="delay in milliseconds",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-maxhe",
        "--max-host-error",
        help="maximum number of errors for a host after which other paths will be skipped",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--proxy-url",
        "--proxy",
        help="proxy url, e.g. socks5://localhost:1080",
    )
    parser.add_argument(
        "-f",
        "--follow-redirects",
        help="follow redirects",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--probe-read-length",
        help="probe bytes read",
        type=int,
        default=1 << 18,
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
    return parser, parser.parse_args(argv, namespace=NameSpace())


def main(argv: typing.Sequence[str] | None = None) -> None | int:
    _, args = parse_args(argv=argv)

    logger.setLevel(
        max(logging.DEBUG, logging.WARNING - logging.DEBUG * args.verbosity)
    )
    logger.addHandler(ColorHandler())

    logger.debug("debugger: %s", ["off", "on"][DEBUGGER_ON])

    if (config_file := args.config or find_config()) is None:
        logger.error("config not found")
        return 1

    conf: ConfigDict = yaml.safe_load(config_file)
    config_file.close()
    logger.debug(f"config loaded: {config_file.name}")
    probes = conf["probes"]

    urls: list[str] = args.urls

    if not args.input.isatty():
        urls: itertools.chain[str] = itertools.chain(
            urls, filter(None, map(str.rstrip, args.input))
        )

    ignore_hosts: map[str] | None = (
        filter(None, filter(None, map(str.rstrip, args.ignore_hosts)))
        if args.ignore_hosts
        else None
    )

    scanner = Scanner(
        probes=probes,
        output=args.output,
        timeout=conf.get("timeout", args.timeout),
        connect_timeout=conf.get("connect_timeout", args.connect_timeout),
        read_timeout=conf.get("read_timeout", args.read_timeout),
        workers_num=conf.get("workers_num", args.workers_num),
        delay=conf.get("delay", args.delay),
        ignore_hosts=conf.get("ignore_hosts", ignore_hosts),
        max_host_error=conf.get("max_host_error", args.max_host_error),
        proxy_url=conf.get("proxy_url", args.proxy_url),
        follow_redirects=conf.get("follow_redirects", args.follow_redirects),
        probe_read_length=conf.get("probe_read_length", args.probe_read_length),
    )

    try:
        asyncio.run(scanner.scan(urls))
    except KeyboardInterrupt:
        logger.error("execution interrupted by user")
        return 2
    except Exception as ex:
        logger.critical(ex, exc_info=True)
        return 1
    else:
        logger.info("finished!")


if __name__ == "__main__":
    sys.exit(main())
