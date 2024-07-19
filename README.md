# httpscan 🌐🔍

A configurable HTTP scanner for finding publicly available dumps, backups, configs, and more. It can send requests using other HTTP methods besides `GET`. It can bypass **Cloudflare** checks, but **Node.js** must be installed on your system.

I've been using the httpx tool for a long time. That tool is great except for its configurability (and it doesn't know how to bypass CloudFlare). I needed a declarative way to describe scanning rules, plus the tool had to download the found backups, dumps, etc., on its own. The same developers have **nuclei**, but it solves slightly different problems, and mainly, I can't just rewrite tons of code in Go, where doing many things is a pain.

## Installation 🛠️

```bash
pipx install httpscan
```

> Use pipx instead of pip to install packages containing executables.

Install the latest version from GitHub:

```bash
pipx install git+https://github.com/s3rgeym/httpscan.git
```

## Usage 🚀

```bash
$ httpscan -i urls.txt -o results.json -vv --proxy 'socks5://localhost:1080'
```

Help:

```bash
$ httpscan -h
```

- `--workers` specifies the total number of workers.
- Each worker processes one link from the queue.
- There is a `--delay` between attempts on the same link (site) because Nginx often limits the number of requests from one IP per second.
- `--parallel` is the maximum number of parallel attempts for ALL PROBES run by different workers.

If the config path is not specified, files named `httpscan.yml` or `httpscan.yaml` will be searched in the current working directory or `~/.config`.

> Fields contained in the config: see `Config`.

The config has a `probes` field, which contains a list of `ProbeDict` objects.

### Example probe for finding DB dumps:

```yaml
probes:
# ...
- condition: status_code == 200
  match: INSERT INTO
  name: database dump
  path: /{db,dump,database,backup}.sql
  save_file: true
```

The repository contains a [sample.httpscan.yml](./sample.httpscan.yml) (you can move it to `~/.config/httpscan.yml`).

- Each element in the `probes` array contains a required `name` field with the name of the probe.
- `path` is the path to be appended to each URL. The path supports brace expansion like in BASH, for example, `/{foo,ba{r,z}}.biz` (the paths `/foo.biz`, `/bar.biz`, and `/baz.biz` will be checked).
- `method` specifies the HTTP method; `params` is for passing **QUERY STRING** parameters, `data` for parameters sent via `application/x-www-form-urlencoded`, `json`..., `cookies`..., `headers`...
- `condition` allows filtering results. A built-in expression engine supports operators `==`, `!=`, `<`, `<=`, `>`, `>=`, `!` or `NOT`, `AND` or `&&`, `OR` or `||`. `=~` checks if a string matches a pattern, similar to calling `bool(re.search(right, left))`. All operators are case-insensitive. They can be grouped using parentheses. Available variables: `status_code`, `content_length`, `content_type`, `title`... For example, `status_code == 200 && content_type == 'application/json'`. Note that strings must be in quotes...
- `match`, `not_match` check for matching a regular expression pattern in the server's response. `extract` and `extract_all` allow extracting content that matches a pattern. Since the response body can be huge, the probe reads the first 64 KB of data from the socket by default. For an HTML page, this is enough (according to [this site](https://almanac.httparchive.org/en/2022/page-weight), the average HTML size in 2022 was 31 KB), and archives can be checked for the absence of HTML tags.
- `save_file: true` saves the file in case of success by default in the `./output/%hostname%` directory.

Scanning results are output in **JSONL** format (JSON Lines, where each object is on a new line). Use `jq` to work with them.

```json
{
  "content_languages": ["en"],
  "content_length": 303,
  "content_type": "application/octet-stream",
  "host": "domain.tld",
  "http_version": "1.1",
  "input": "https://domain.tld",
  "port": 443,
  "probe": {
    "name": "docker config file",
    "not_match": "^\\s*<[a-zA-Z]+",
    "path": "/{{prod,dev,}.env,Dockerfile{,.prod,.dev},docker-compose{,.prod,.dev}.yml}",
    "save_file": true
  },
  "response_headers": {
    "Accept-Ranges": "bytes",
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Content-Length": 303,
    "Date": "Tue, 16 Jul 2024 17:38:08 GMT",
    "Etag": "\"12f-60425670233e7\"",
    "Expires": "0",
    "Last-Modified": "Wed, 30 Aug 2023 15:15:48 GMT",
    "Pragma": "no-cache",
    "Server": "Apache",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Vary": "User-Agent",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "sameorigin",
    "X-XSS-Protection": "1; mode=block"
  },
  "response_url": "https://domain.tld/.env",
  "saved_as": "/tmp/x/domain.tld/.env",
  "saved_bytes": 303,
  "server": "Apache",
  "status_code": 200,
  "status_reason": "OK"
}
```

## Other Features 🧩

- Each link to be scanned uses a random `User-Agent`.
- Proxy support, for example, `socks5://localhost:1080`.
- `--exclude-hosts` can be used to pass a list of ignored hosts, and patterns with asterisks like `*.shopify.com` can be used to filter out subdomains. Domains can be written in any case.
- Certain response codes can be skipped, for example, `--exclude-statuses 401 403`.

## Development 🖥️

```bash
git clone ... && cd ...
python -m venv .venv
. .venv/bin/activate
# install all dependencies from pyproject.toml
pip install .
```

### TODO

- `set_state: next_state`
- `on_state: state_name`
- `run_python: script_name`

```python
def run(...) -> ...:
    ...
```
