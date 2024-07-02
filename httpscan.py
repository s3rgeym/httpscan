#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import copy
import dataclasses
import functools
import itertools
import json
import logging
import os
import pathlib
import random
import re
import sys
import typing
import urllib.parse
from collections import Counter
from email.message import Message as EmailMessage

import aiohttp
import aiohttp.abc
import yaml
from aiohttp_socks import ProxyConnector

__version__ = "0.1.0"
__author__ = "Sergey M"

# При запуске отладчика VS Code устанавливает переменную PYDEVD_USE_FRAME_EVAL=NO
DEBUGGER_ON = any(name.startswith("PYDEVD_") for name in os.environ)

CSI = "\x1b["
RESET = f"{CSI}m"
RED = f"{CSI}31m"
GREEN = f"{CSI}32m"
YELLOW = f"{CSI}33m"
BLUE = f"{CSI}34m"
PURPLE = f"{CSI}35m"
CYAN = f"{CSI}36m"

log = logging.getLogger(__name__)


class ColorHandler(logging.StreamHandler):
    LEVEL_COLORS = {
        "DEBUG": GREEN,
        "INFO": YELLOW,
        "WARNING": PURPLE,
        "ERROR": RED,
        "CRITICAL": RED,
    }

    _fmt = logging.Formatter("[ %(asctime)s ] %(levelname)8s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        message = self._fmt.format(record)
        return f"{self.LEVEL_COLORS[record.levelname]}{message}{RESET}"


# stderr = functools.partial(print, file=sys.stderr, flush=True)


@functools.lru_cache
def expand(s: str) -> set[str]:
    # Модифицированная версия отсюда https://algo.monster/liteproblems/1096
    rv = set()

    def dfs(exp: str) -> None:
        try:
            inner, after = exp.split("}", 1)
        except ValueError:
            rv.add(exp)
            return

        before, inner = inner.rsplit("{", 1)

        for item in inner.split(","):
            dfs(before + item + after)

    dfs(s)

    return rv


TOKENS_RE = re.compile(
    r"""(?P<NULL>(?:null|nil))|(?P<BOOLEAN>(?:true|false))|(?P<ID>[a-z_][a-z0-9_]*)|(?P<NUMBER>\d+(\.\d+)?)|(?P<STRING>(?:"[^"]*"|'[^']*'))|(?P<COMPARE>(?:[=!]=|[<>]=?))|(?P<NOT>(?:not|!))|(?P<AND>(?:and|&&))|(?P<OR>(?:or|\|\|))|(?P<LPAREN>\()|(?P<RPAREN>\))|(?P<SPACE>\s+)|(?P<INVALID_CHAR>.)|(?P<END>$)""",
    re.IGNORECASE,
)


class Token(typing.NamedTuple):
    name: str
    value: str
    pos: int


@dataclasses.dataclass
class ExpressionExecutor:
    source: str

    def execute(self, vars: dict[str, typing.Any] = {}) -> typing.Any:
        self.vars = copy.deepcopy(vars)
        self.tokens_it = self.tokenize(self.source)
        self.next_tok = None
        self.advance()
        return self.stmt()

    def tokenize(self, s: str) -> typing.Iterable[Token]:
        for m in TOKENS_RE.finditer(s):
            for k, v in m.groupdict().items():
                if v is not None:
                    if k != "SPACE":
                        yield Token(k, v, m.start())
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
            return [int, float]["." in self.cur_tok.value](self.cur_tok.value)

        if self.match("STRING"):
            return self.cur_tok.value[1:-1]

        self.unexpected_next_token()

    def stmt(self) -> typing.Any:
        rv = self.expr()
        self.expect("END")
        return rv


def execute(s: str, vars: dict[str, typing.Any]) -> typing.Any:
    return ExpressionExecutor(s).execute(vars)


def parse_header(h: str) -> tuple[str, dict]:
    message = EmailMessage()
    message["content-type"] = h
    params = message.get_params()
    return params[0][0], dict(params[1:])


class ProbeConfig(typing.TypedDict):
    name: str
    path: str
    method: typing.NotRequired[typing.Literal["GET", "HEAD", "POST", "PUT", "DELETE"]]
    params: typing.NotRequired[dict]
    headers: typing.NotRequired[dict]
    cookies: typing.NotRequired[dict]
    data: typing.NotRequired[dict]
    json: typing.NotRequired[dict]
    match: typing.NotRequired[str]
    not_match: typing.NotRequired[str]
    extract: typing.NotRequired[str]
    # status_code >= 200 && status_code < 300
    condition: typing.NotRequired[str]
    save_to: typing.NotRequired[os.PathLike]


DEFAULT_PROBES: list[ProbeConfig] = [
    {
        "name": "server directory listing",
        "path": "/{wp-content,{backup,dump}{,s}}",
        "match": "<title>Index of /",
        "condition": "status_code == 200 && mime_type == 'text/html'",
    },
]


class Config(typing.TypedDict):
    workers: typing.NotRequired[int]
    timeout: typing.NotRequired[int | float]
    max_host_error: typing.NotRequired[int]
    probes: typing.NotRequired[list[ProbeConfig]]
    proxy_url: typing.NotRequired[str]


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
        "-w",
        "--workers-num",
        "--workers",
        help="number of workers",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-t",
        "--timeout",
        help="request timeout",
        type=float,
        default=10.0,
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
        help="proxy url, eg `socks5://127.0.0.1:9050`",
    )
    parser.add_argument(
        "-v", "--verbosity", help="be more verbosity", action="count", default=0
    )
    return parser


FAIL = object()


@dataclasses.dataclass
class Scanner:
    _: dataclasses.KW_ONLY
    probes: list[ProbeConfig]
    output: typing.TextIO = sys.stdout
    workers_num: int = 10
    timeout: float = 10.0
    max_host_error: int = 10
    proxy_url: str | None = None
    host_error_counter: Counter[str] = dataclasses.field(
        init=False, repr=False, default_factory=Counter
    )

    async def scan(self, urls: typing.Sequence[str]) -> None:
        log.info("scanning started")

        if self.proxy_url and not (await self.check_proxy()):
            raise ValueError("ip leak detected!")

        self.user_agents = await self.fetch_user_agents()

        self.queue = asyncio.Queue(maxsize=self.workers_num)

        # Если `asyncio.TaskGroup()` первым идет, то падает `RuntimeError: Session is closed`
        async with self.get_session(
            proxy_url=self.proxy_url,
            timeout=self.timeout,
            # Игнориуем куки чтобы сканер не зависал после посещения N сайтов
            cookie_jar=aiohttp.DummyCookieJar(),
        ) as session, asyncio.TaskGroup() as tg:
            tg.create_task(self.produce(urls))

            for _ in range(self.workers_num):
                tg.create_task(self.worker(session))

            tg.create_task(self.stop_workers())

        log.info("scanning finished!")

    async def produce(self, urls: typing.Sequence[str]) -> None:
        for url in urls:
            for probe_conf in self.probes:
                for path in expand(probe_conf["path"]):
                    target_url = urllib.parse.urljoin(url, path)
                    await self.queue.put((target_url, probe_conf))

    async def stop_workers(self) -> None:
        await self.queue.join()

        log.debug("stop workers")

        for _ in range(self.workers_num):
            self.queue.put_nowait((None, None))

    async def worker(
        self,
        session: aiohttp.ClientSession,
    ) -> None:
        while True:
            url, conf = await self.queue.get()

            if url is None:
                break

            try:
                hostname = urllib.parse.urlsplit(url).netloc

                if self.host_error_counter[hostname] > self.max_host_error:
                    log.warning(f"maximum host error exceeded: {hostname}")
                    continue

                headers = conf.get("headers", {})

                headers |= {
                    "User-Agent": random.choice(self.user_agents),
                    "Referer": "https://www.google.com/",
                }

                method = conf.get("method", "GET").upper()

                response = await session.request(
                    method,
                    url,
                    headers=headers,
                    params=conf.get("params"),
                    data=conf.get("data"),
                    json=conf.get("json"),
                    cookies=conf.get("cookies"),
                    allow_redirects=False,
                )

                log.debug(f"{response.status} - {response.method} - {response.url}")

                result = await self.do_probe(response, conf)

                if result is FAIL:
                    continue

                self.output_json(
                    {
                        k: v
                        for k, v in {
                            "url": url,
                            "status_code": response.status,
                            "content_type": response.headers.get("Content-Type"),
                            "content_length": int(
                                response.headers.get("Content-Length", 0)
                            ),
                            "description": conf["name"],
                            "result": result,
                        }.items()
                        if v
                    },
                    sort_keys=True,
                )
            except Exception as ex:
                self.host_error_counter[hostname] += 1
                if DEBUGGER_ON:
                    log.exception(ex)
                else:
                    log.error(ex)
            finally:
                self.queue.task_done()

    async def do_probe(
        self,
        response: aiohttp.ClientResponse,
        conf: ProbeConfig,
    ) -> typing.Any:
        rv = {}

        if "condition" in conf:
            mime_type, _ = parse_header(response.content_type)

            vars_dict = {
                "status_code": response.status,
                "content_length": response.content_length,
                "content_type": response.content_type,
                "mime_type": mime_type,
            }

            if not execute(conf["condition"], vars_dict):
                return FAIL

        if "extract" in conf:
            text = await response.text()
            if match := re.search(conf["extract"], text):
                rv |= {"extracted": match.group()}
            else:
                return FAIL

        if "match" in conf:
            text = await response.text()
            if not re.search(conf["match"], text):
                return FAIL

        if "not_match" in conf:
            text = await response.text()
            if re.search(conf["not_match"], text):
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
                async for data in response.content.iter_chunked(2**16):
                    f.write(data)

                log.debug(f"saved: {save_path}")

        return rv

    def output_json(self, obj: typing.Any, **kw: typing.Any) -> None:
        js = json.dumps(obj, ensure_ascii=False, **kw)
        print(js, file=self.output, flush=True)

    async def fetch_user_agents(self) -> list[str]:
        async with self.get_session() as session:
            r = await session.get("https://www.useragents.me/")
            content = await r.text()
            data = re.findall(
                r'<textarea class="form-control" rows="8">(\[.*?\])</textarea>',
                content,
                re.DOTALL,
            )
            assert data, "can't fetch user agents"
            return [item["ua"] for item in itertools.chain(*map(json.loads, data))]

    async def check_proxy(self) -> bool:
        real_ip = await self.get_public_ip()
        proxy_ip = await self.get_public_ip(proxy_url=self.proxy_url)
        return real_ip != proxy_ip

    async def get_public_ip(self, **kw: typing.Any) -> str:
        async with self.get_session(**kw) as session:
            response = await session.get("https://api.ipify.org?format=json")
            return (await response.json())["ip"]

    @contextlib.asynccontextmanager
    async def get_session(
        self,
        *,
        timeout: float | aiohttp.ClientTimeout | None = None,
        user_agent: str | None = None,
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
            log.debug(f"using proxy {self.proxy_url!r} for all connections")
            connector = ProxyConnector.from_url(self.proxy_url)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            cookie_jar=cookie_jar,
        ) as session:
            if user_agent:
                session.headers.update({"User-Agent": user_agent})
            yield session


def normalize_url(u: str) -> str:
    return u if "://" in u else f"https://{u}"


def main(argv: typing.Sequence | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)

    lvl = max(logging.DEBUG, logging.WARNING - logging.DEBUG * args.verbosity)
    log.setLevel(lvl)
    log.addHandler(ColorHandler())

    conf: Config

    if config_file := args.config or find_config():
        conf = yaml.safe_load(config_file)
        log.debug(f"config loaded: {config_file.name}")
    else:
        log.warning("config not found")
        conf = {}

    urls = list(args.urls)
    # log.debug(f"{urls=}")

    if not (args.input.isatty() and urls):
        urls = itertools.chain(urls, filter(None, map(str.strip, args.input)))

    urls = map(normalize_url, urls)

    probes: list[ProbeConfig] = conf.get("probes", DEFAULT_PROBES)

    scanner = Scanner(
        probes=probes,
        output=args.output,
        workers_num=conf.get("workers", args.workers_num),
        timeout=conf.get("timeout", args.timeout),
        max_host_error=conf.get("max_host_error", args.max_host_error),
        proxy_url=conf.get("prixy_url", args.proxy_url),
    )

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(scanner.scan(urls))


def find_config() -> None | typing.TextIO:
    config_name = pathlib.Path(__file__).stem
    for config_directory in [pathlib.Path.cwd(), pathlib.Path.home() / ".config"]:
        for ext in ("yml", "yaml"):
            path = config_directory / f"{config_name}.{ext}"
            if path.exists():
                return path.open()


if __name__ == "__main__":
    sys.exit(main())
