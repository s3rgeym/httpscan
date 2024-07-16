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
```

Справка:

```bash
$ httpscan -h
```

* `--workers` указывает общее количество воркеров (?).
*  Каждый воркер обрабатывает одну ссылку из очереди.
*  Между пробами одной ссылки (сайта) есть задержка `--delay`, тк Nginx часто ограничивает количество запросов с одного айпи в секунду.
*  `--parallel` — это максимальное количество параллельных проб для ВСЕХ ПРОБ, запущенными разными воркерами.

Если путь до конфига не задан, то в текущей рабочей директории либо в `~/.config` ищутся файлы с именами `httpscan.yml` или `httpscan.yaml`.

> Поля которые содержит конфиг: см. `Config`.

В конфиге есть поле `probes`. Оно содержит список объектов `ProbeDict`.

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
* `condition` позволяет отфильтровать результаты. Имеется встроенный движок выражений. Поддерживаются операторы `==`, `!=`, `<`, `<=`, `>`, `>=`, `!` или `NOT`, `AND` или `&&`, `OR` или `||`. `=~` служит для проверки строки на соответствие шаблону, он аналогичен вызову `bool(re.search(right, left))`. Все операторы регистронезависимы. Их можно группировать с помощью круглых скобок. Доступны переменные: `status_code`, `content_length`, `content_type`, `title`... Например, `status_code == 200 && content_type == 'application/json'`. Заметьте, что строки должны быть в кавычках...
* `match`, `not_match` проверяют на соответсвие шаблону регулярного выражения совместимого с Python ответ сервера. `extract` и `extract_all` позволяют извлечь содержимое, если оно соответствует шаблону. Так как тело ответа может быть гигантским, то из сокета для пробы по умолчанию читаются первые 64 килобайт данных. Для html-страницы этого достаточно (если верить [этому сайту](https://almanac.httparchive.org/en/2022/page-weight), то в 2022 году средний размер html был 31 килобайт), а всякие архивы можно проверять на отсутствие html-тегов.
* `save_file: true` ­— сохраняет файл в случае успеха по умолчанию в каталог `./output/%hostname%`.

Результаты сканирования выводятся в формате **JSONL** (JSON Lines, где каждый объект с новой строки). Для работы с ними используйте `jq`.

```json
{"content_languages": ["en"], "content_length": 303, "content_type": "application/octet-stream", "host": "domain.tld", "http_version": "1.1", "input": "https://domain.tld", "port": 443, "probe": {"name": "docker config file", "not_match": "^\\s*<[a-zA-Z]+", "path": "/{{prod,dev,}.env,Dockerfile{,.prod,.dev},docker-compose{,.prod,.dev}.yml}", "save_file": true}, "response_headers": {"Accept-Ranges": "bytes", "Cache-Control": "no-cache, no-store, must-revalidate", "Content-Length": "303", "Date": "Tue, 16 Jul 2024 17:38:08 GMT", "Etag": "\"12f-60425670233e7\"", "Expires": "0", "Last-Modified": "Wed, 30 Aug 2023 15:15:48 GMT", "Pragma": "no-cache", "Server": "Apache", "Strict-Transport-Security": "max-age=31536000; includeSubDomains", "Vary": "User-Agent", "X-Content-Type-Options": "nosniff", "X-Frame-Options": "sameorigin", "X-XSS-Protection": "1;  mode=block"}, "response_url": "https://domain.tld/.env", "saved_as": "/tmp/x/domain.tld/.env", "saved_bytes": 303, "server": "Apache", "status_code": 200, "status_reason": "OK"}
```

Другие особенности:

* Для каждой ссылки для сканирования используется рандомный `User-Agent`.
* Поддерживаются прокси, например, `socks5://localhost:1080`.
* С помощью `--exclude-hosts` можно передать список игнорируемых хостов, причем можно использовать шаблоны со звездочкой типа `*.shopify.com`, чтобы отсеивать поддомены. Домены можно писать в любом регистре.
* Определенные коды ответов можно пропустить, например, `--exclude-statuses 401 403`.

Для разработки:

```bash
git clone ... && cd ...
python -m venv .venv
. .venv/bin/activate
# установит все зависимости из pyproject.toml
pip install .
```

### TODO

* `set_state: next_state`
* `on_state: state_name`
* `run_python: script_name`

```python
def run(...) -> ...:
    ...
```
