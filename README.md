# httpscan

Конфигуриемый http-сканер. Позволяет просканировать список URL'ов...

Я долгое время пользовался утилитой httpx. Она позволяет, например, искать папки `/.git`, доступные из браузера. Та утилита всем хороша за исключением настраивамости. Мне нужен декларативный способ описания правил для сканирования. От тех же разработчиков есть nuclei, но он решает немного другие вопросы, ну а главное - я не могу его взять и переписать, потому как там тонны кода на Go.

В PyPi пакет называется `httpscan`.

Установка самой последней версии:

```bash
pipx install git+https://github.com/s3rgeym/httpscan.git
```
> Используйте pipx вместо pip для установки пакетов, содержащих исполняемые файлы

```bash
httpscan -h

httpscan -i URLs.txt -c /path/to/config.yml > results.json
```

Если путь до конфига не задан, то в текущей рабочей директории либо в `~/.config` ищутся файлы с именами `httpscan.yml` или `httpscan.yaml`.

В конфиге есть поле `probes`. Оно содержит список объектов `ProbeConfig`.

> Поля которые содержит конфиг: см. `Config`.

В репозитории имеется [httpscan.yml](./httpscan.yml) для примера.

* Каждый элемент массива `probes` содержит обязательное поле `name` с именем пробы.
* `path` — это путь для подстановки к каждому URL. Путь поддерживает brace expansion как в BASH, например, `/{foo,ba{r,z}}.biz` (будут проверены пути `/foo.biz`, `/bar.biz` и `/baz.biz`).
* `method` задает HTTP-метод; `params` служит для передачи параметров **QUERY STRING**, `data` — параметры передаваемые с помощью `application/x-www-form-urlencoded`, `json`..., `cookies`..., `headers`...
* `condition` позволяет отфильтровать результаты. Имеется встроенный движок выражений. Поддерживаются операторы `==`, `!=`, `<`, `<=`, `>`, `>=`, `!` или `NOT`, `AND` или `&&`, `OR` или `||`. Все регистронезависимы. Их можно группировать с помощью скобок. Доступны переменные: `status_code`, `content_length`, `content_type`, `mime_type`... Например, `status_code == 200 and mime_type == 'application/json'`. Заметьте, что строки должны быть в кавычках...
* `match`, `not_match` проверяют на соответсвие шаблону регулярного выражения ответ сервера. `extract` позволяет извлечь содержимое, если оно соответствует шаблону.
* `save_to` сохраняет файл...

Результаты сканирования выводятся в формате **JSONL** (JSON Lines, где каждый объект с новой строки). Для работы с ними используйте `jq`.

Другие особенности:

* При каждом запросе используется рандомный заголовок `User-Agent`.
* Поддерживаются прокси, например, `socks5://127.0.0.1:9050`.
