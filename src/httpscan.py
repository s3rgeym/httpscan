#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import asyncio
import collections
import contextlib
import copy
import dataclasses
import datetime
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

__version__ = "0.3.3"
__author__ = "Sergey M"

# При запуске отладчика VS Code устанавливает переменную PYDEVD_USE_FRAME_EVAL=NO
DEBUGGER_ON = any(name.startswith("PYDEVD_") for name in os.environ)

HEADER_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
HEADER_ACCEPT_LANGUAGE = "en-US,en;q=0.9"
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
GOOGLE_REFERER = "https://www.google.com/"

USER_AGENTS_ENDPOINT = "https://useragents.io/random/__data.json?limit=1000"

TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE)


# FailType = typing.NewType("FailType", None)

# FAIL = FailType(0)


class ANSI:
    CSI = "\x1b["
    RESET = f"{CSI}m"
    BLACK = f"{CSI}30m"
    RED = f"{CSI}31m"
    GREEN = f"{CSI}32m"
    YELLOW = f"{CSI}33m"
    BLUE = f"{CSI}34m"
    PURPLE = f"{CSI}35m"
    CYAN = f"{CSI}36m"
    WHITE = f"{CSI}37m"


class ColorHandler(logging.StreamHandler):
    _level_colors = {
        "DEBUG": ANSI.BLUE,
        "INFO": ANSI.GREEN,
        "WARNING": ANSI.PURPLE,
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
    is_num = all(x.isdigit() for x in range_parts)
    first, last = map((ord, int)[is_num], range_parts)
    # first, last = min(first, last), max(first, last)
    res = range(first, last + 1)
    return list(res if is_num else map(chr, res))


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
        "COMPARE": r"(?:=~|[=!]=|[<>]=?)",
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
                # соответствие шаблону
                case "=~":
                    rv = re.search(self.primary(), rv) is not None
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
    save_file: typing.NotRequired[bool]


class ConfigDict(typing.TypedDict):
    workers: typing.NotRequired[int]
    timeout: typing.NotRequired[int | float]
    connect_timeout: typing.NotRequired[int | float]
    read_timeout: typing.NotRequired[int | float]
    max_host_error: typing.NotRequired[int]
    probes: typing.NotRequired[list[ProbeDict]]
    exclude_hosts: typing.NotRequired[list[str]]
    proxy_url: typing.NotRequired[str]
    user_agent: typing.NotRequired[str]
    force_https: typing.NotRequired[bool]
    match_statuses: typing.NotRequired[list[int | str]]
    exclude_statuses: typing.NotRequired[list[int | str]]
    save_dir: typing.NotRequired[os.PathLike]
    probe_read_length: typing.NotRequired[str]


OUTPUT_DIR = pathlib.Path.cwd() / "output"


@dataclasses.dataclass
class Settings:
    _: dataclasses.KW_ONLY
    workers: int = 10
    parallel_probes: int = 50
    max_host_error: int = 10
    timeout: int | float | None = None
    connect_timeout: int | float = 10.0
    read_timeout: int | float = 5.0
    host_delay: int = 120
    proxy_url: str | None = None
    exclude_hosts: typing.Iterable[str] | None = None
    user_agent: str | None = None
    force_https: bool = False
    save_dir: pathlib.Path = OUTPUT_DIR
    match_statuses: typing.Sequence[int] = dataclasses.field(
        default_factory=list
    )
    exclude_statuses: typing.Sequence[int] = dataclasses.field(
        default_factory=list
    )
    # читаем и проверяем только первые 64kb
    probe_read_length: int = 1 << 16
    # TODO: добавить флаг?
    bypass_cloudflare_tries: int = 1

    def __post_init__(self) -> None:
        # переводим в нижний регистр
        self.exclude_hosts: set[str] = set(
            map(str.lower, self.exclude_hosts or [])
        )
        if self.proxy_url is None:
            self.proxy_url = os.getenv("PROXY_URL")


@dataclasses.dataclass
class Scanner:
    probes: list[ProbeDict]
    settings: Settings = dataclasses.field(default_factory=Settings)

    async def scan(
        self, urls: typing.Iterable[str], output: typing.TextIO = sys.stdout
    ) -> None:
        if self.settings.proxy_url and not (await self.check_proxy()):
            raise ValueError("ip leak detected!")

        queue = asyncio.Queue(maxsize=self.settings.workers)
        sem = asyncio.Semaphore(
            max(self.settings.parallel_probes, self.settings.workers)
        )
        lock = asyncio.Lock()
        host_errors = collections.Counter()

        user_agents = []
        if not self.settings.user_agent:
            try:
                user_agents = await self.get_user_agents()
            except Exception:
                logger.warning("can't fetch user agents")

        workers = [
            Worker(
                self,
                output=output,
                queue=queue,
                sem=sem,
                lock=lock,
                host_errors=host_errors,
                user_agents=user_agents,
            ).run()
            for _ in range(self.settings.workers)
        ]

        await asyncio.gather(
            self.produce(urls, queue), *workers, return_exceptions=True
        )

    async def produce(
        self, urls: typing.Iterable[str], queue: asyncio.Queue
    ) -> None:
        for url in map(normalize_url, urls):
            try:
                url_sp = urllib.parse.urlsplit(url)
                # hostname всегда в нижнем регистре
                if self.excluded_host(url_sp.hostname):
                    logger.debug(f"skip ignored host: {url_sp.hostname}")
                    continue

                # scheme тоже в нижний переводится
                if url_sp.scheme == "http" and self.settings.force_https:
                    url = "https" + url[4:]

                await queue.put(url)
            except Exception as ex:
                logger.exception(ex)

        for _ in range(self.settings.workers):
            await queue.put(None)

    def excluded_host(self, hostname: str) -> bool:
        hostname_parts = hostname.split(".")

        # www.linux.org.ru => {'*.linux.org.ru', '*.org.ru', '*.ru', 'www.linux.org.ru'}
        hostname_wildcards = set(
            [
                ".".join(["*"] + hostname_parts[i:])
                for i in range(1, len(hostname_parts))
            ]
            + [hostname]
        )

        return bool(hostname_wildcards & self.settings.exclude_hosts)

    async def get_user_agents(self) -> list[str]:
        logger.debug("get user agents from %s", USER_AGENTS_ENDPOINT)
        async with self.get_session() as session:
            r = await session.get(USER_AGENTS_ENDPOINT)
            json_data = await r.json()
        return json_data["nodes"][1]["data"][2:-1][1::4]

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
            ProxyConnector.from_url(self.settings.proxy_url, limit=0)
            if use_proxy and self.settings.proxy_url
            else aiohttp.TCPConnector(limit=0)
        )

        tmt = aiohttp.ClientTimeout(
            total=self.settings.timeout,
            sock_connect=self.settings.connect_timeout,
            sock_read=self.settings.read_timeout,
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
                    "User-Agent": user_agent
                    or self.settings.user_agent
                    or DEFAULT_USER_AGENT,
                    **headers,
                }
            )
            yield session


LANGUAGES_RE = re.compile(
    "|".join(
        f"(?P<{k}>{v})"
        for k, v in {
            # https://stackoverflow.com/questions/23491793/how-to-using-regex-to-check-if-a-string-contains-german-characters-in-javascript
            "de": r"[a-z]*(?:[ßüöä][a-z]*)+",
            # https://stackoverflow.com/questions/1922097/regular-expression-for-french-characters
            "fr": r"[a-z]*(?:[àâçéèêëîïôûùüÿñæœ]+[a-z]*)+",
            "es": r"[a-z]*(?:[ñáéíóú]+[a-z]*)+",
            # английский содержит много общих символов с языками выше
            "en": r"[a-z]+",
            # > re.findall(r'[ЁёА-я]+', 'Воркута - город на севере России')
            # ['Воркута', 'город', 'на', 'севере', 'России']
            "ru": r"[ёа-я]+",
            # https://stackoverflow.com/questions/6787716/regular-expression-for-japanese-characters
            "jp": r"[一-龠ぁ-ゔァ-ヴーａ-ｚＡ-Ｚ０-９々〆〤]+",
            # https://stackoverflow.com/a/71633418/2240578
            "kr": r"[\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uAC00-\uD7AF\uD7B0-\uD7FF]+",
            "cn": r"[\u4E00-\u9FFF]+",
        }.items()
    ),
    re.IGNORECASE,
)


def strip_html_tags(s: str) -> str:
    # > strip_html_tags('<p>Sample text<script>alert("XSS!")</script>')
    # 'Sample text'
    return re.sub(
        "<.*?>", "", re.sub(r"<script.*?</script>", "", s, re.IGNORECASE)
    )


def detect_languages(s: str) -> list[str]:
    """languages are sorted by rate"""
    c = collections.Counter()
    for m in LANGUAGES_RE.finditer(s):
        lang, word = next(filter(lambda x: x[1], m.groupdict().items()))
        c[lang] += len(word)
    return [x[0] for x in c.most_common()]


META_GENERATOR_RE = re.compile(
    r'<meta\s+name="generator"\s+content="([^"]+)', re.IGNORECASE
)


@dataclasses.dataclass
class Worker:
    scanner: Scanner
    output: typing.TextIO
    queue: asyncio.Queue
    lock: asyncio.Lock
    sem: asyncio.Semaphore
    host_errors: collections.Counter[str, int]
    user_agents: typing.Sequence[str]

    async def run(self) -> None:
        task_name = asyncio.current_task().get_name()
        logger.debug("started: %s", task_name)

        # без этой строки отработает как надо, но не выведет сообщения о
        # завершении
        with contextlib.suppress(asyncio.CancelledError):
            while True:
                url = await self.queue.get()

                if url is None:
                    break

                try:
                    # resolver = DNSResolver(
                    #     nameservers=[
                    #         "8.8.8.8",
                    #         "8.8.4.4",
                    #         "1.1.1.1",
                    #         "1.0.0.1",
                    #     ]
                    # )

                    # res = await resolver.getaddrinfo(
                    #     urllib.parse.urlsplit(url).hostname
                    # )

                    user_agent = (
                        self.settings.user_agent
                        if self.settings.user_agent
                        else self.rand_ua()
                        if self.user_agents
                        else DEFAULT_USER_AGENT
                    )

                    logger.debug(f"user agent for {url}: {user_agent}")
                    self.next_request = 0

                    # Для каждого url используем новую сессию из-за того, что сессии
                    # со временем начинают тормозить
                    async with self.scanner.get_session(
                        user_agent=user_agent
                    ) as self.session:
                        async with asyncio.TaskGroup() as tg:
                            for probe in self.scanner.probes:
                                for path in expand(probe["path"]):
                                    await self.sleep()
                                    async with self.sem:
                                        tg.create_task(
                                            self.make_probe(url, path, probe)
                                        )
                # ! asyncio.exceptions.CancelledError наследуется напрямую от
                # BaseException, поэтому нужно перехватывать Exception
                except Exception as ex:
                    logger.exception(ex)
                finally:
                    self.queue.task_done()

        logger.debug("finished: %s", task_name)

    @property
    def settings(self) -> Settings:
        return self.scanner.settings

    def rand_ua(self) -> str:
        return random.choice(self.user_agents)

    async def make_probe(
        self,
        base_url: str,
        path: str,
        probe: ProbeDict,
    ) -> None:
        try:
            netloc = urllib.parse.urlparse(base_url).netloc

            if self.host_errors[netloc] >= self.settings.max_host_error:
                logger.warning(f"max host error exceeded: {netloc}")
                return

            url = urllib.parse.urljoin(base_url, path)
            headers = probe.get("headers", {}).copy()

            response = await self.send_probe_request(url, headers, probe)

            for tries in itertools.count(1):
                # Редиректы с http на https должны срабатывать, а так же на www.
                # assert (
                #     remove_www(response.url._val.netloc) == remove_www(netloc)
                # ), f"response does not have same domain with requested url: {url}"
                if (
                    self.settings.match_statuses
                    and response.status not in self.settings.match_statuses
                ):
                    logger.warning(
                        f"skip status: {response.status} {response.url}"
                    )
                    return

                if response.status in self.settings.exclude_statuses:
                    logger.warning(
                        f"skip status: {response.status} {response.url}"
                    )
                    return

                content: bytes = await response.content.read(
                    self.settings.probe_read_length
                )

                text: str = content.decode(
                    response.charset or "ascii", errors="replace"
                )

                # Всегда содержит заголовки `Cache-Control: *no-cache*` и `Transfer-Encoding: chunked`
                if "<title>One moment, please...</title>" in text:
                    logger.debug(f"cloudflare challenge detected: {url}")

                    assert (
                        tries > self.settings.bypass_cloudflare_tries
                    ), f"maximum tries to bypass cloudflare exceeded: {self.settings.bypass_cloudflare_tries}"

                    challenge = CloudflareChallenge.from_text(text)

                    # разгадываем скобки и возвращаем запрашиваемую страницу
                    response = await self.bypass_cloudflare_challenge(
                        challenge,
                        response,
                        headers,
                    )

                    continue

                break

            if (
                result := await self.get_probe_result(
                    response,
                    text,
                    content,
                    probe,
                )
            ) is None:
                # logger.warning(f"failed probe {probe['name']!r}: {url}")
                return

            logger.info(f"successed probe {probe['name']!r}: {url}")

            report = {
                "input": base_url,
                "response_headers": dict(response.headers),
                "probe": probe,
                "last_visit": datetime.datetime.now().strftime("%F %T"),
                **result,
            }

            # на этот символ python заменяет неверные последовательности
            # if "�" not in text:
            report["content_languages"] = detect_languages(
                strip_html_tags(text)
                if response.content_type == "text/html"
                else text
            )

            js = json.dumps(
                remove_none_from_dict(report),
                ensure_ascii=False,
                sort_keys=True,
            )
            print(js, file=self.output, flush=True)
        except Exception as ex:
            logger.error(ex)
            self.host_errors[netloc] += 1

    async def sleep(self) -> None:
        if self.settings.host_delay > 0:
            async with (
                self.lock
            ):  # блокируем асинхронное выполнение остальных заданий
                if (dt := self.next_request - time.monotonic()) > 0:
                    logger.debug(
                        f"{asyncio.current_task().get_name()} sleep {dt:.3f}s"
                    )
                    await asyncio.sleep(dt)

                self.next_request = (
                    time.monotonic() + self.settings.host_delay / 1000
                )

    async def send_probe_request(
        self,
        url: str,
        headers: dict[str, str],
        probe: ProbeDict,
    ) -> aiohttp.ClientResponse:
        method = probe.get("method", "GET").upper()
        logger.debug(f"send request: {method} {url}")

        response = await self.session.request(
            method,
            url,
            headers=headers,
            params=probe.get("params"),
            data=probe.get("data"),
            json=probe.get("json"),
            cookies=probe.get("cookies"),
            allow_redirects=False,
        )

        logger.debug(
            f"got response: {response.status} {response.method} {response.url}"
        )

        return response

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

        challenge_response = await self.session.get(
            url=challenge_endpoint,
            params=payload,
            headers=additional_headers | {"Referer": str(origin_response.url)},
        )

        assert (
            len(challenge_response.history) == 1
            and challenge_response.history[0].status == 301
            # 'Set-Cookie': 'wschkid=ae96fb3bf715d463c7f3328d2e4377cb9aa6b155.1720572164.1; Expires=Thu, 08-Aug-24 00:42:44 GMT; Domain=<censored>; Path=/; HttpOnly; SameSite=Lax'
            and challenge.param
            in challenge_response.history[0].headers.get("Set-Cookie", "")
            and challenge_response.url == origin_response.url
        ), f"can't bypass challenge: {origin_response.url}"

        return challenge_response

    async def get_probe_result(
        self,
        response: aiohttp.ClientResponse,
        text: str,
        content: bytes,
        conf: ProbeDict,
    ) -> dict[str, typing.Any] | None:
        getheader = response.headers.get

        rv = {
            "url": str(response.url),
            "host": response.url.host,
            "port": response.url.port,
            "path": response.url.path_qs,
            "http_version": f"{response.version.major}.{response.version.minor}",
            "status_code": response.status,
            "status_reason": response.reason,
            "content_length": response.content_length,
            "content_type": response.content_type,
            "content_charset": response.charset,
            # condition не поддерживает массивы, поэтому добавлены эти переменные
            "server": getheader("server"),
            "powered_by": getheader("x-powered-by"),
        }

        if m := TITLE_RE.search(text):
            rv["title"] = m.group(1)

        # содержит название cms
        if m := META_GENERATOR_RE.search(text):
            rv["meta_generator"] = m.group(1)

        if "condition" in conf:
            # уже распарсенный
            # mime_type, _ = parse_header(response.content_type)
            if not execute(conf["condition"], rv):
                return

        if "match" in conf:
            if not re.search(conf["match"], text):
                return

        if "not_match" in conf:
            if re.search(conf["not_match"], text):
                return

        if "extract" in conf:
            if match := re.search(conf["extract"], text):
                rv |= {"match": match.group()}
            else:
                return

        if "extract_all" in conf:
            if items := re.findall(conf["extract_all"], text):
                rv |= {"matches": items}
            else:
                return

        if conf.get("save_file"):
            save_path = (
                self.settings.save_dir
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

            logger.info(f"file saved: {save_path}")

            stat = save_path.stat()

            if stat.st_size == 0:
                logger.warning(f"empty file: {save_path}")
                save_path.unlink()
                return

            rv |= {
                "saved_bytes": stat.st_size,
                "saved_as": str(save_path.resolve()),
            }

        return rv


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


@dataclasses.dataclass
class CloudflareChallenge:
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

        return cls(
            action=re.search(r'action="([^"]+)', text).group(1),
            method=re.search(r'method="([^"]+)"', text).group(1).lower(),
            param=re.search(
                r'<input type="hidden".+?name="([^"]+)',
                text,
            ).group(1),
            **dict(re.findall(r"(west|east)=([^,]+)", text)),
        )


def normalize_url(u: str) -> str:
    return u if "://" in u else f"https://{u}"


def remove_www(s: str) -> str:
    return re.sub(r"^www\.", "", s, re.IGNORECASE)


def mask_ip(addr: str, ch: str = "*") -> str:
    """маскирует все сегменты адреса за исключением последнего

    >>> mask_ip("192.168.0.104")
    '***.***.*.104'"""
    return re.sub(r"[^.](?![^.]*$)", ch, addr)


def remove_none_from_dict(d: typing.Mapping) -> dict:
    return {
        k: remove_none_from_dict(v) if isinstance(v, typing.Mapping) else v
        for k, v in d.items()
        if v is not None
    }


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


def parse_statuses(args: list[int | str]) -> list[int]:
    rv = []
    for arg in args:
        try:
            first, last = sorted(map(int, arg.split("-", 1)))
            rv.extend(range(first, last + 1))
        except (ValueError, AttributeError):
            rv.append(int(arg))
    return rv


def parse_size(s: str) -> int:
    """
    >>> parse_size("512K")
    524288
    """
    s = s.rstrip()
    size, unit = [s[:-1], s[-1]] if s[-1].isalpha() else [s, ""]
    return int(size) * 1024 ** ["", "k", "m", "g"].index(unit.lower())


def filter_empty_lines(fp: typing.TextIO) -> filter[str]:
    return filter(None, map(str.rstrip, fp))


class NameSpace(argparse.Namespace):
    urls: list[str]
    input: typing.TextIO
    output: typing.TextIO
    config: typing.TextIO
    exclude_hosts: typing.TextIO
    workers: int
    parallel_probes: int
    timeout: int | float
    connect_timeout: int | float
    read_timeout: int | float
    host_delay: int | float
    max_host_error: int
    force_https: bool
    match_statuses: list[str]
    exclude_statuses: list[str]
    proxy_url: str
    probe_read_length: int
    save_dir: str
    user_agent: str
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
        type=argparse.FileType("a"),
        default="-",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="custom config file",
        type=argparse.FileType(),
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        help="directory to save files",
        default="./output",
    )
    parser.add_argument(
        "-w",
        "--workers",
        help="number of workers",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-p",
        "--parallel-probes",
        "--parallel",
        help="number of parallel probes",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-maxhe",
        "--max-host-error",
        help="maximum number of errors for a host after which other paths will be skipped",
        type=int,
        default=30,
    )
    parser.add_argument(
        "-t", "--timeout", help="probe total timeout sec", type=float
    )
    parser.add_argument(
        "-r",
        "--read-timeout",
        "--socket-read",
        "--read",
        help="probe socket read timeout sec",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "-C",
        "--connect-timeout",
        "--socket-connect",
        "--connect",
        help="probe socket read timeout sec",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "-d",
        "--host-delay",
        "--delay",
        help="host delay before probe request in milliseconds",
        type=int,
        default=200,
    )
    parser.add_argument(
        "-xh",
        "--exclude-hosts",
        "--ignore-hosts",
        help="exclude hosts file",
        type=argparse.FileType(),
    )
    parser.add_argument(
        "--force-https",
        help="force replace the scheme from http to https",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "-ms",
        "--match-statuses",
        nargs="+",
        default=[],
        help="match status codes",
    )
    parser.add_argument(
        "-xs",
        "--exclude-statuses",
        "--ignore-statuses",
        nargs="+",
        default=[],
        help="exclude status codes",
    )
    parser.add_argument(
        "--proxy-url",
        "--proxy",
        help="proxy url, e.g. socks5://localhost:1080. Also you can set PROXY_URL environmemt variable",
    )
    parser.add_argument(
        "--probe-read-length",
        "--probe-read",
        help="probe read length; supported units: K, M",
        default="64k",
    )
    parser.add_argument(
        "-ua",
        "--user-agent",
        help="use specified user-agent instead random",
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
        max(logging.DEBUG, logging.ERROR - logging.DEBUG * args.verbosity)
    )

    logger.addHandler(ColorHandler())

    logger.debug("debugger: %s", ["off", "on"][DEBUGGER_ON])

    if (config_file := args.config or find_config()) is None:
        logger.error("config not found")
        return 1

    with config_file:
        conf: ConfigDict = yaml.safe_load(config_file)

    logger.debug(f"config loaded: {config_file.name}")

    urls = args.urls

    if not args.input.isatty():
        urls = itertools.chain(urls, filter_empty_lines(args.input))

    # TODO: сделать что-то с настройками. Очень громоздко выглядит
    settings = Settings(
        timeout=conf.get("timeout", args.timeout),
        connect_timeout=conf.get("connect_timeout", args.connect_timeout),
        read_timeout=conf.get("read_timeout", args.read_timeout),
        workers=conf.get("workers", args.workers),
        parallel_probes=conf.get("parallel_probes", args.parallel_probes),
        host_delay=conf.get("host_delay", args.host_delay),
        save_dir=pathlib.Path(conf.get("save_dir", args.save_dir)),
        exclude_hosts=conf.get(
            "exclude_hosts",
            filter_empty_lines(args.exclude_hosts)
            if args.exclude_hosts
            else [],
        ),
        max_host_error=conf.get("max_host_error", args.max_host_error),
        proxy_url=conf.get("proxy_url", args.proxy_url),
        force_https=conf.get("force_https", args.force_https),
        exclude_statuses=parse_statuses(
            conf.get("exclude_statuses", args.exclude_statuses)
        ),
        match_statuses=parse_statuses(
            conf.get("match_statuses", args.match_statuses)
        ),
        probe_read_length=parse_size(
            conf.get("probe_read_length", args.probe_read_length)
        ),
        user_agent=conf.get("user_agent", args.user_agent),
    )

    scanner = Scanner(
        probes=conf["probes"],
        settings=settings,
    )

    try:
        asyncio.run(scanner.scan(urls, output=args.output))
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
