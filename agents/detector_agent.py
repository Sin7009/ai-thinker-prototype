import json

class DetectorAgent:
    """
    Агент для диагностики когнитивных искажений (Контур Б).
    В этой версии реализована имитация с помощью простого поиска по ключевым словам.
    """
    def __init__(self):
        # Слова-маркеры для определения "черно-белого мышления"
        self.black_white_markers = [
            "всегда", "никогда", "все", "никто", "абсолютно", "полностью", "совершенно"
        ]
        print("DetectorAgent (Mock) инициализирован.")

    def analyze(self, text: str) -> str:
        """
        Анализирует текст на наличие когнитивных искажений.
        Возвращает JSON-строку с результатами.
        """
        detected_patterns = []

        # Поиск маркеров черно-белого мышления
        found_markers = [marker for marker in self.black_white_markers if marker in text.lower()]

        if found_markers:
            detected_patterns.append({
                "bias": "black_and_white_thinking",
                "confidence": 85,
                "context": f"Обнаружены маркеры: {', '.join(found_markers)}"
            })

        # Если ничего не найдено, возвращаем пустой список
        if not detected_patterns:
            return json.dumps([])

        return json.dumps(detected_patterns)

if __name__ == '__main__':
    # Пример использования
    detector = DetectorAgent()

    text1 = "Это абсолютно лучший проект, который я когда-либо видел."
    analysis1 = detector.analyze(text1)
    print(f"Текст: \"{text1}\"")
    print(f"Результат анализа: {analysis1}\n")

    text2 = "Мне кажется, это хорошая идея."
    analysis2 = detector.analyze(text2)
    print(f"Текст: \"{text2}\"")
    print(f"Результат анализа: {analysis2}\n")
