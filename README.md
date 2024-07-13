# httpscan

Конфигуриемый HTTP сканер для поиска дампов, бекапов, конфигов в открытом доступе, а так же может отправлять запросы другими HTTP-методами, а не только `GET`. Может обходить проверку от **Cloudflare**, но для этого в системе должна быть установлена **Node.js**.

Я долгое время пользовался утилитой httpx. Та утилита всем хороша за исключением настраивамости (а еще она CloudFlare обходить не умеет). Мне нужен был декларативный способ описания правил для сканирования + утилита должна была сама выкачивать найденные бекапы, дампы и тп. От тех же разработчиков есть **nuclei**, но он решает немного другие вопросы, ну а главное, я не могу взять и просто переписать тонны кода на Go, где дельть многие вещи — боль.

Установка:

```bash
pipx install httpscan
```

> Используйте pipx вместо pip для установки пакетов, содержащих исполняемые файлы

Установка самой последней версии с github:

```bash
pipx install git+https://github.com/s3rgeym/httpscan.git
```

Использование:

```bash
$ httpscan -i urls.txt -o results.json -vv --proxy 'socks5://localhost:1080'

$ httpscan -h
usage: httpscan.py [-h] [-u URLS [URLS ...]] [-i INPUT] [-o OUTPUT] [-c CONFIG]
                   [-s SAVE_DIR] [-w WORKERS_NUM] [-t TIMEOUT] [-r READ_TIMEOUT]
                   [-C CONNECT_TIMEOUT] [-d DELAY] [-igh IGNORE_HOSTS]
                   [-maxhe MAX_HOST_ERROR] [-f | --follow-redirects | --no-follow-redirects]
                   [-xs SKIP_STATUSES [SKIP_STATUSES ...]] [--proxy-url PROXY_URL]
                   [-pl PROBE_READ_LENGTH] [-v] [--version]

configurable http scanner

options:
  -h, --help            show this help message and exit
  -u URLS [URLS ...], --url URLS [URLS ...]
                        target url to probe(s) (default: [])
  -i INPUT, --input INPUT
                        input file with urls (default: -)
  -o OUTPUT, --output OUTPUT
                        output file to results in JSONL (default: -)
  -c CONFIG, --config CONFIG
                        custom config file (default: None)
  -s SAVE_DIR, --save-dir SAVE_DIR
                        directory to save files (default: ./output)
  -w WORKERS_NUM, --workers-num WORKERS_NUM, --workers WORKERS_NUM
                        number of workers (default: 20)
  -t TIMEOUT, --timeout TIMEOUT
                        total timeout sec (default: None)
  -r READ_TIMEOUT, --read-timeout READ_TIMEOUT, --socket-read READ_TIMEOUT, --read READ_TIMEOUT
                        socket read timeout sec (default: 5.0)
  -C CONNECT_TIMEOUT, --connect-timeout CONNECT_TIMEOUT, --socket-connect CONNECT_TIMEOUT, --connect CONNECT_TIMEOUT
                        socket read timeout sec (default: 10.0)
  -d DELAY, --delay DELAY
                        delay in milliseconds (default: 50)
  -igh IGNORE_HOSTS, --ignore-hosts IGNORE_HOSTS, --ignore IGNORE_HOSTS
                        ignore hosts file (default: None)
  -maxhe MAX_HOST_ERROR, --max-host-error MAX_HOST_ERROR
                        maximum number of errors for a host after which other paths will be
                        skipped (default: 10)
  -f, --follow-redirects, --no-follow-redirects
                        follow redirects (default: False)
  -xs SKIP_STATUSES [SKIP_STATUSES ...], --skip-statuses SKIP_STATUSES [SKIP_STATUSES ...]
                        always skip status codes (default: [])
  --proxy-url PROXY_URL, --proxy PROXY_URL
                        proxy url, e.g. socks5://localhost:1080. Also you can set PROXY_URL
                        environmemt variable (default: None)
  -pl PROBE_READ_LENGTH, --probe-read-length PROBE_READ_LENGTH
                        probe read length; supported units: K, M (default: 128k)
  -v, --verbosity       be more verbosity (default: 0)
  --version             show program's version number and exit
```

Если путь до конфига не задан, то в текущей рабочей директории либо в `~/.config` ищутся файлы с именами `httpscan.yml` или `httpscan.yaml`.

> Поля которые содержит конфиг: см. `Config`.

В конфиге есть поле `probes`. Оно содержит список объектов `ProbeConfig`.

Пример пробы для поиска дампов БД:

```yaml
probes:
# ...
- condition: status_code == 200
  match: INSERT INTO
  name: database dump
  path: /{db,dump,database,backup}.sql
  save_file: true
```

В репозитории имеется [sample.httpscan.yml](./sample.httpscan.yml) (можно его переместить в `~/.config/httpscan.yml`).

* Каждый элемент массива `probes` содержит обязательное поле `name` с именем пробы.
* `path` — это путь для подстановки к каждому URL. Путь поддерживает brace expansion как в BASH, например, `/{foo,ba{r,z}}.biz` (будут проверены пути `/foo.biz`, `/bar.biz` и `/baz.biz`).
* `method` задает HTTP-метод; `params` служит для передачи параметров **QUERY STRING**, `data` — параметры передаваемые с помощью `application/x-www-form-urlencoded`, `json`..., `cookies`..., `headers`...
* `condition` позволяет отфильтровать результаты. Имеется встроенный движок выражений. Поддерживаются операторы `==`, `!=`, `<`, `<=`, `>`, `>=`, `!` или `NOT`, `AND` или `&&`, `OR` или `||`. Все регистронезависимы. Их можно группировать с помощью круглых скобок. Доступны переменные: `status_code`, `content_length`, `content_type`... Например, `status_code == 200 && content_type == 'application/json'`. Заметьте, что строки должны быть в кавычках...
* `match`, `not_match` проверяют на соответсвие шаблону регулярного выражения совместимого с Python ответ сервера. `extract` и `extract_all` позволяют извлечь содержимое, если оно соответствует шаблону. Так как тело ответа может быть гигантским, то из сокета для пробы по умолчанию читаются первые 128 килобайт данных. Для html-страницы этого достаточно, а всякие архивы можно проверять на отсутствие html-тегов.
* `save_file: true` ­— сохраняет файл в случае успеха по умолчанию в каталог `./output/%hostname%`.

Результаты сканирования выводятся в формате **JSONL** (JSON Lines, где каждый объект с новой строки). Для работы с ними используйте `jq`.

```json
{"content_charset": "UTF-8", "content_length": 667, "content_type": "text/html", "host": "<censored>", "http_version": "1.1", "input": "http://<censored>/", "probe_name": "server directory listing", "response_headers": {"Content-Encoding": "gzip", "Content-Length": "667", "Content-Type": "text/html;charset=UTF-8", "Date": "Fri, 12 Jul 2024 16:14:55 GMT", "Server": "Apache/2.4.25 (Debian)", "Vary": "Accept-Encoding"}, "status_code": 200, "status_reason": "OK", "url": "http://<censored>/includes/"}
```

Другие особенности:

* Для каждой ссылки для сканирования используется рандомный `User-Agent`.
* Поддерживаются прокси, например, `socks5://localhost:1080`.
* С помощью `--ignore-hosts` можно передать список игнорируемых хостов, причем можно использовать шаблоны со звездочкой типа `*.shopify.com`, чтобы отсеивать поддомены. Домены можно писать в любом регистре.
* Определенные коды ответов можно пропустить, например, `--skip-statuses 401 403`.

Для разработки:

```bash
git clone ... && cd ...
python -m venv .venv
. .venv/bin/activate
# установит все зависимости из pyproject.toml
pip install .
```
