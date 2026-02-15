from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional

class Handler(ABC):
    """
    Абстрактный класс, определяющий интерфейс для обработчика в паттерне "Цепочка ответственности".
    Каждый конкретный обработчик должен реализовывать метод `handle`.
    """
    _next: Optional[Handler] = None
    def set_next(self, handler: Handler) -> Handler:
        """
        Устанавливает следующий обработчик в цепочке.

        Args:
            handler: Следующий обработчик (экземпляр класса, наследующего Handler).

        Returns:
            Экземпляр следующего обработчика, что позволяет связывать их в цепь.
        """
        self._next = handler
        return handler
    @abstractmethod
    def handle(self, data: Any) -> Any:
        """
        Обрабатывает данные или передает их следующему обработчику в цепочке.

        Args:
            data: Входные данные, которые могут быть любого типа.

        Returns:
            Результат обработки данных, который может быть любого типа.
            Если есть следующий обработчик, возвращается его результат.
        """
        if self._next:
            return self._next.handle(data)
        return data