# app.py
import asyncio
import aiohttp
from flask import Flask, render_template, request, jsonify
import logging
from typing import List, Dict, Any, Optional

# Настройка логирования для отладки
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Конфигурация
CROSSREF_API_URL = "https://api.crossref.org/works"
DEFAULT_LIMIT = 20
TIMEOUT_SECONDS = 10

# -------------------------------------------------------------------
# Асинхронные функции для работы с Crossref API
# -------------------------------------------------------------------

async def fetch_publications(session: aiohttp.ClientSession,
                              author: Optional[str] = None,
                              title: Optional[str] = None,
                              limit: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    """
    Выполняет один запрос к Crossref API с заданными параметрами.
    Возвращает список публикаций (каждая публикация — словарь с нужными полями).
    """
    params = {
        "rows": limit,
        "sort": "relevance",
        "order": "desc"
    }
    if author:
        params["query.author"] = author
    if title:
        params["query.title"] = title

    try:
        # Асинхронный GET-запрос с таймаутом
        async with session.get(CROSSREF_API_URL, params=params,
                               timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)) as response:
            if response.status != 200:
                logger.error(f"API вернул код {response.status}")
                return []
            data = await response.json()
            items = data.get("message", {}).get("items", [])
            return parse_publications(items)
    except asyncio.TimeoutError:
        logger.error("Превышено время ожидания ответа от API")
        return []
    except aiohttp.ClientError as e:
        logger.error(f"Ошибка HTTP-клиента: {e}")
        return []
    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}")
        return []

def parse_publications(items: List[Dict]) -> List[Dict[str, Any]]:
    """Извлекает из сырого ответа API необходимые поля для каждой публикации."""
    results = []
    for item in items:
        # Название статьи (берём первый элемент массива)
        title_list = item.get("title", [])
        title_text = title_list[0] if title_list else "Без названия"

        # Информация об авторах
        authors = item.get("author", [])
        first_author = authors[0] if authors else None
        author_name = ""
        affiliation = "Не указано"
        if first_author:
            given = first_author.get("given", "")
            family = first_author.get("family", "")
            author_name = f"{given} {family}".strip()
            # Аффилиация первого автора
            affiliations = first_author.get("affiliation", [])
            if affiliations and isinstance(affiliations, list):
                affiliation = affiliations[0].get("name", "Не указано")

        # Название журнала/сборника
        container = item.get("container-title", [])
        journal = container[0] if container else "Неизвестное издание"

        # Год публикации
        pub_date = item.get("published", {}) or item.get("published-print", {}) or item.get("published-online", {})
        date_parts = pub_date.get("date-parts", [[]])
        year = date_parts[0][0] if date_parts and date_parts[0] else "Нет даты"

        results.append({
            "title": title_text,
            "author": author_name,
            "journal": journal,
            "year": str(year),
            "affiliation": affiliation,
            "doi": item.get("DOI", "")
        })
    return results

async def search_by_author_parallel(authors_list: List[str],
                                     limit_per_author: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    """
    Выполняет параллельные запросы для нескольких авторов (поиск по каждой фамилии).
    Возвращает объединённый список публикаций без дубликатов (по DOI).
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for author in authors_list:
            if author.strip():
                tasks.append(fetch_publications(session, author=author.strip(), limit=limit_per_author))
        # Запускаем все запросы параллельно
        results_per_author = await asyncio.gather(*tasks, return_exceptions=True)

    # Объединяем результаты, фильтруя ошибки и дубликаты по DOI
    seen_dois = set()
    combined = []
    for res in results_per_author:
        if isinstance(res, list):
            for pub in res:
                doi = pub.get("doi")
                if doi and doi not in seen_dois:
                    seen_dois.add(doi)
                    combined.append(pub)
    return combined

async def search_crossref_async(author: Optional[str] = None,
                                 title: Optional[str] = None,
                                 limit: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    """
    Основная асинхронная функция поиска.
    Поддерживает поиск по одному автору, по названию, либо по обоим критериям.
    Если в поле author передана строка с запятыми, запускается параллельный поиск по каждому автору.
    """
    # Обработка множественных авторов (через запятую)
    if author and "," in author:
        authors_list = [a.strip() for a in author.split(",") if a.strip()]
        if authors_list:
            # Параллельный поиск по всем указанным авторам
            return await search_by_author_parallel(authors_list, limit)

    # Обычный поиск (один автор или только название)
    async with aiohttp.ClientSession() as session:
        return await fetch_publications(session, author=author, title=title, limit=limit)

# -------------------------------------------------------------------
# Flask маршруты (синхронные обёртки для асинхронных вызовов)
# -------------------------------------------------------------------

def run_async(coro):
    """Запускает корутину в новом цикле событий (для использования в синхронном Flask)."""
    try:
        # Создаём новый цикл событий в текущем потоке
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@app.route("/", methods=["GET", "POST"])
def index():
    """Главная страница с формой поиска и отображением результатов."""
    results = []
    error_message = None
    if request.method == "POST":
        author = request.form.get("author", "").strip()
        title = request.form.get("title", "").strip()

        # Валидация: хотя бы одно поле должно быть заполнено
        if not author and not title:
            error_message = "Пожалуйста, укажите хотя бы автора или название публикации."
        else:
            # Асинхронный вызов поиска (запуск через отдельный цикл)
            try:
                results = run_async(search_crossref_async(author=author or None,
                                                           title=title or None,
                                                           limit=DEFAULT_LIMIT))
                if not results:
                    error_message = "По вашему запросу ничего не найдено. Попробуйте изменить критерии."
            except Exception as e:
                logger.exception("Ошибка при выполнении асинхронного поиска")
                error_message = "Произошла ошибка при обращении к API Crossref. Попробуйте позже."

    return render_template("index.html", results=results, error=error_message)

# -------------------------------------------------------------------
# Запуск приложения
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Для production используйте WSGI-сервер (gunicorn), но для локальной отладки подойдёт
    app.run(debug=True, threaded=True)