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
# -i urls.txt                       - загрузить список ссылок для сканирования
#                                     из файла, где каждая ссылка с новой строки
# -o results.json                   - файл с результатами сканирования, где
#                                     результаты в виде JSON-объектов - каждый с
#                                     новой строки
# -vv                               - вывод отладочных сообщений
# --proxy 'socks5://localhost:1080' - использовать прокси для сканирования
httpscan -i urls.txt -o results.json -vv --proxy 'socks5://localhost:1080'

httpscan -h
```

Если путь до конфига не задан, то в текущей рабочей директории либо в `~/.config` ищутся файлы с именами `httpscan.yml` или `httpscan.yaml`.

> Поля которые содержит конфиг: см. `Config`.

В конфиге есть поле `probes`. Оно содержит список объектов `ProbeConfig`.

Пример пробы:

```yaml
probes:
# ...
- condition: status_code == 200
  match: INSERT INTO
  name: database dump
  path: /{db,dump,database,backup}.sql
  save_to: output
```

В репозитории имеется [sample.httpscan.yml](./sample.httpscan.yml) (можно его переместить в `~/.config/httpscan.yml`).

* Каждый элемент массива `probes` содержит обязательное поле `name` с именем пробы.
* `path` — это путь для подстановки к каждому URL. Путь поддерживает brace expansion как в BASH, например, `/{foo,ba{r,z}}.biz` (будут проверены пути `/foo.biz`, `/bar.biz` и `/baz.biz`).
* `method` задает HTTP-метод; `params` служит для передачи параметров **QUERY STRING**, `data` — параметры передаваемые с помощью `application/x-www-form-urlencoded`, `json`..., `cookies`..., `headers`...
* `condition` позволяет отфильтровать результаты. Имеется встроенный движок выражений. Поддерживаются операторы `==`, `!=`, `<`, `<=`, `>`, `>=`, `!` или `NOT`, `AND` или `&&`, `OR` или `||`. Все регистронезависимы. Их можно группировать с помощью круглых скобок. Доступны переменные: `status_code`, `content_length`, `content_type`... Например, `status_code == 200 and content_type == 'application/json'`. Заметьте, что строки должны быть в кавычках...
* `match`, `not_match` проверяют на соответсвие шаблону регулярного выражения ответ сервера. `extract` и `extract_all` позволяют извлечь содержимое, если оно соответствует шаблону. Так как тело ответа может быть гигантским, то из сокета читаются только первые 256 килобайт данных.
* `save_to` сохраняет файл...

Результаты сканирования выводятся в формате **JSONL** (JSON Lines, где каждый объект с новой строки). Для работы с ними используйте `jq`.

```json
{"content_charset": "UTF-8", "content_length": 667, "content_type": "text/html", "host": "<censored>", "http_version": "1.1", "input": "http://<censored>/", "probe_name": "server directory listing", "response_headers": {"Content-Encoding": "gzip", "Content-Length": "667", "Content-Type": "text/html;charset=UTF-8", "Date": "Fri, 12 Jul 2024 16:14:55 GMT", "Server": "Apache/2.4.25 (Debian)", "Vary": "Accept-Encoding"}, "status_code": 200, "status_reason": "OK", "url": "http://<censored>/includes/"}
```

Другие особенности:

* При каждом запросе используется рандомный заголовок `User-Agent`.
* Поддерживаются прокси, например, `socks5://localhost:1080`.
* С помощью `--ignore-hosts` можно передать список игнорируемых хостов, причем можно использовать шаблоны со звездочкой типа `*.shopify.com`, чтобы отсеивать поддомены. Домены можно писать в любом регистре.

Для разработки:

```bash
git clone ... && cd ...
python -m venv .venv
. .venv/bin/activate
# установит все зависимости из pyproject.toml
pip install .
```
