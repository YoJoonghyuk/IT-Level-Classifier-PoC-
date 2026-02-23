def is_it_developer(position_text: str) -> bool:
    """
    Определяет, относится ли текст должности к IT-разработке.

    Используется для первичной фильтрации датасета HeadHunter, чтобы выделить
    целевую группу специалистов (разработчики, инженеры, QA и т.д.).

    Args:
        position_text (str): Текстовое описание должности (искомой или текущей).

    Returns:
        bool: True, если в тексте найдены ключевые слова, характерные для IT-разработки,
              иначе False.
    """
    if not isinstance(position_text, str):
        return False

    s = position_text.lower()
    it_keywords = [
        'разработчик', 'developer', 'программист', 'engineer', 'инженер',
        'backend', 'frontend', 'fullstack', 'qa', 'тестировщик', 'devops',
        'python', 'java', 'c++', 'javascript', 'php', 'go', 'data scientist'
    ]
    return any(kw in s for kw in it_keywords)