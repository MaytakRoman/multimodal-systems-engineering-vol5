# Карта программных ссылок для книги

Пути относительные: после публикации добавляется фактический адрес репозитория и `/blob/main/`.
Для цитирования закреплённой редакции используйте SHA коммита вместо `main`.
Файл `make_book_links.py` формирует полные адреса по переданному URL.

| Раздел | Описание и запуск | Исходный код |
|---|---|---|
| 1.1 Раннее, позднее и промежуточное объединение модальностей | [docs/chapter01.md](docs/chapter01.md) | [mmbook/fusion.py](mmbook/fusion.py) |
| 1.2 Общие пространства представлений: контрастивное обучение (CLIP и аналоги) | [docs/chapter01.md](docs/chapter01.md) | [mmbook/fusion.py](mmbook/fusion.py) |
| 2.1 Визуальные трансформеры и самообучение | [docs/chapter02.md](docs/chapter02.md) | [mmbook/vision.py](mmbook/vision.py) |
| 2.2 Детекция и сегментация по текстовому запросу | [docs/chapter02.md](docs/chapter02.md) | [mmbook/vision.py](mmbook/vision.py) |
| 2.3 Дообучение визуальных моделей под домен | [docs/chapter02.md](docs/chapter02.md) | [mmbook/vision.py](mmbook/vision.py) |
| 3.1 Связка визуального кодировщика и языковой модели | [docs/chapter03.md](docs/chapter03.md) | [mmbook/vlm.py](mmbook/vlm.py) |
| 3.2 Данные и этапы обучения | [docs/chapter03.md](docs/chapter03.md) | [mmbook/vlm.py](mmbook/vlm.py) |
| 3.3 Визуальные галлюцинации и их контроль | [docs/chapter03.md](docs/chapter03.md) | [mmbook/vlm.py](mmbook/vlm.py) |
| 4.1 Распознавание текста, таблиц, формул | [docs/chapter04.md](docs/chapter04.md) | [mmbook/documents.py](mmbook/documents.py) |
| 4.2 Чертежи, схемы и техническая документация | [docs/chapter04.md](docs/chapter04.md) | [mmbook/documents.py](mmbook/documents.py) |
| 5.1 Распознавание речи и диаризация | [docs/chapter05.md](docs/chapter05.md) | [mmbook/speech.py](mmbook/speech.py) |
| 5.2 Синтез речи и клонирование голоса: практика и этика | [docs/chapter05.md](docs/chapter05.md) | [mmbook/speech.py](mmbook/speech.py) |
| 5.3 Голосовые ассистенты в реальном времени | [docs/chapter05.md](docs/chapter05.md) | [mmbook/speech.py](mmbook/speech.py) |
| 6.1 Понимание длинного видео | [docs/chapter06.md](docs/chapter06.md) | [mmbook/video.py](mmbook/video.py) |
| 6.2 Видеоаналитика: события, аномалии, подсчёт | [docs/chapter06.md](docs/chapter06.md) | [mmbook/video.py](mmbook/video.py) |
| 7.1 Диффузионные модели и согласование потоков на практике | [docs/chapter07.md](docs/chapter07.md) | [mmbook/generation.py](mmbook/generation.py) |
| 7.2 Управляемая генерация и дообучение под стиль | [docs/chapter07.md](docs/chapter07.md) | [mmbook/generation.py](mmbook/generation.py) |
| 7.3 Ускорение генерации: дистилляция шагов | [docs/chapter07.md](docs/chapter07.md) | [mmbook/generation.py](mmbook/generation.py) |
| 8.1 Видеомодели и модели мира | [docs/chapter08.md](docs/chapter08.md) | [mmbook/media.py](mmbook/media.py) |
| 8.2 Генерация звука и музыки, 3D-сцен | [docs/chapter08.md](docs/chapter08.md) | [mmbook/media.py](mmbook/media.py) |
| 9.1 Модели «зрение — язык — действие» | [docs/chapter09.md](docs/chapter09.md) | [mmbook/robotics.py](mmbook/robotics.py) |
| 9.2 Перенос из симуляции в реальный мир | [docs/chapter09.md](docs/chapter09.md) | [mmbook/robotics.py](mmbook/robotics.py) |
| 10.1 Сжатие мультимодальных моделей | [docs/chapter10.md](docs/chapter10.md) | [mmbook/deployment.py](mmbook/deployment.py) |
| 10.2 Вывод на периферийных устройствах и промышленных камерах | [docs/chapter10.md](docs/chapter10.md) | [mmbook/deployment.py](mmbook/deployment.py) |
| 11.1 Дефектоскопия и контроль безопасности персонала | [docs/chapter11.md](docs/chapter11.md) | [mmbook/industrial.py](mmbook/industrial.py) |
| 11.2 Сбор данных, обучение, внедрение, эффект | [docs/chapter11.md](docs/chapter11.md) | [mmbook/industrial.py](mmbook/industrial.py) |

## Дополнение 1.1

- 1.2 — CLIP: три фотографии: [real_models.py](real_models.py), [review_results/open_models/clip_retrieval.json](review_results/open_models/clip_retrieval.json).
- 5.1 — Whisper tiny: [real_models.py](real_models.py), [review_results/open_models/whisper_asr.json](review_results/open_models/whisper_asr.json).
- 9.1 — Обязательный проход и неоднозначная команда: [review_robotics.py](review_robotics.py), [review_results/robotics/summary.json](review_results/robotics/summary.json).
- 10.1 — Матрица полного ранга: [review_deployment.py](review_deployment.py), [review_results/full_rank_svd.json](review_results/full_rank_svd.json).
- 10.2 — Профиль CPU CLIP: [real_models.py](real_models.py), [review_results/open_models/cpu_profile.json](review_results/open_models/cpu_profile.json).
- 11.2 — Партии и групповой бутстреп: [review_industrial.py](review_industrial.py), [review_results/industrial/summary.json](review_results/industrial/summary.json).
- 11.3 — Три канала и погрешность датчика: [review_industrial.py](review_industrial.py), [review_results/industrial/summary.json](review_results/industrial/summary.json).
