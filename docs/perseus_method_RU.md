# Методика и сохранённые измерения Perseus

Это сведения из этапа выбора модели до OOT, а не новый запуск. Позднейшие OOT-результаты находятся в `artifacts/compare_oot_eval.json`.

Основной выбор R3/R4: **256 событий, dim 128, 2 слоя, 2 эпохи; val-A GAUC 0.54857372**. Авторы хуже прежнего выбора: 0.54685030 против 0.54857372; основные R3/R4 сохранены.

Выбор только по половине A дней 287–293. GAUC: AUC внутри (uid, day), вес — число показов, canonical src/tbrec/metrics.py:summary. Val-B и OOT метрики не вычислялись.

| Конфиг | Окно меток | История / dim / слои | Batch × накопление | GAUC val-A | Лучшая / выполнено эпох | Fit+save, с | VRAM allocated / reserved, MiB |
|---|---|---|---|---:|---:|---:|---:|
| R3 small | 273–286 | 256 / 128 / 2 | 16 × 1 | 0.54857372 | 2 / 3 | 498.7 | 1007 / 1120 |
| R3 large | 273–286 | 512 / 256 / 2 | 8 × 1 | 0.54672667 | 1 / 2 | 990.9 | 2053 / 2236 |
| H6: лимиты типов и горизонты | 273–286 | 256 / 128 / 2 | 16 × 1 | 0.55019249 | 2 / 2 | 312.5 | 1003 / 1100 |
| R3 авторы | 273–286 | 512 / 256 / 4 | 8 × 8 | 0.54685030 | 2 / 3 | 1864.9 | 2307 / 2726 |

Архитектура авторов сверена в обоих `external/perseus-exps/configs/{baseline_comparison,extra_features}/perseus/` yambda-50m-конфигах: ModernBERT 4 слоя, dim256, 4 головы, dropout0.1, 512 событий, sum, AdamW lr3e-4. Исходный batch64 уменьшен до физического 8 с накоплением8 (effective64); поддержка Accelerator.accumulate и фактическое число шагов накопления проверены. Архитектура не уменьшалась. Исходная retrieval-задача адаптирована в ranking Listen+; данные, cutoff, признаки, словари и PLE такие же, как в остальных R3. В отличие от исходных 30 эпох/patience5 — максимум3, early stopping val-A/patience1 по заданию.
bf16; num_workers=0; унаследованы CPU 11–15 и below-normal. Torch VRAM показывает память нашего аллокатора; на GPU также была фоновая нагрузка.
| Выход | Модель / метки | Строк / null | Сек / 1000 строк |
|---|---|---:|---:|
| `artifacts\runs\r3_perseus\pred_val.parquet` | r3_small; [273, 286] | 725488 / 0 (0.0000%) | 0.052 |
| `artifacts\features\perseus_val.parquet` | r4_val; [259, 272] | 2124374 / 5864 (0.2760%) | 0.160 |
| `artifacts\features\perseus_oot.parquet` | r4_oot; [266, 279] | 2167133 / 3375 (0.1557%) | 0.045 |
| r3_oot | r3_oot; checkpoint готов | скоринга пока нет | — |

Для каждой строки история строго t < day*86400−6395. Все listen (включая skips), like/dislike; played_ratio, organic, artist/album, длина. Дневные повторы train: mean(y), при инференсе одинаковый скор каждому row. Listen+ = pr>=50, до273 метки из алгоритмических listen в events; с273 из rows. Метки294–300 не читались и не выводились из pr. На инференсе rows_no_labels. Пустые истории возвращаются с null; для GAUC только null заменяется фиксированным0.5. allow_unknown_items=true; словари/PLE fit только на train, item cap250k, artist/album100k. Score=sigmoid FP32 нативного ranking logit, единственная метка listen_plus с весом1. Priority0, независимый last-N для каждого cutoff; порядок t/исходный индекс, ловушки объединения срезов нет.
Длина выбранной основной истории: `artifacts/perseus_len.txt`. Конфиг: `artifacts/runs/r3_perseus/selected_config.yaml`; выбор и model mapping: `external/perseus_stage2/selection.json`. Полные затраты (подготовка/модель, fit, чекпойнт, инференс, VRAM), включая прежний технический перезапуск: `artifacts/runs/r3_perseus/cost.json`. H6 остаётся отдельным экспериментом и не подменяет основной выбор. Длинное окно также сохраняется отдельно, без перезаписи R3/R4.


## Предварительные проверки адаптера

Следующие наблюдения относятся к предварительному smoke test на закреплённой версии Perseus, а не к OOT основного эксперимента.

## Данные и конфиг прогона

Основа — `external/perseus-exps/configs/extra_features/perseus/yambda-50m-all-w-features/config.yaml`, задача **retrieval**. Локальные Yambda Parquet и оба mapping прочитаны без изменений и скачивания. `external/prepare_probe.py` выбирает первые 1000 uid с ≥12 likes до общей отсечки X = 80-й перцентиль времени likes; до четырёх последних likes до X становятся обучающими срезами, первые доступные warm-item likes после X — тестовыми. Это короткий smoke test, не воспроизведение метрик статьи.

`external/probe/config.yaml`: sequence 256, dim 128, ModernBERT 2 слоя / 4 головы, batch 8, num_workers 0, bf16 через `ACCELERATE_MIXED_PRECISION=bf16`, 2 эпохи, AdamW lr 3e-4, sampled negatives 128, priority=0 для всех типов. Хранятся like/dislike/listen; listen отфильтрован по played_ratio_pct > 50, как в upstream. Unlikes/undislikes в этом прогоне не использованы. Истории и наблюдаемая длина треков для artifacts ограничены строго `<X`; тест использует замороженную на X историю. Временные метки Yambda при переводе секунд в datetime попадают в 1970 год: это шкала датасета, не доказанные календарные даты реальных событий.

Найдено и исправлено **в подготовке входа**: upstream-пример приводит track_length_seconds кандидатов к Int64, тогда как локальный listen содержит UInt32. Общий PLE Observer падает с `SchemaError: type Int64 is incompatible with expected type UInt32`. В `prepare_probe.py` dtype кандидатов приведён к dtype этого признака в событиях. Файлы кода Perseus не менялись. Для любых общих энкодеров необходимо согласовывать dtype между событиями/context/artifacts.

## API для эксперимента

Все пути ниже относительно `external/perseus/perseus/`.

| Вопрос | Реальный API / контракт |
|---|---|
| Собственный лог | `api/add_events.py:add_events(events, name, source)`: отдельная таблица на тип события; обязательны `client_id: String` без null и `timestamp: Datetime('ns')` без timezone/null. uid → строка, item_id лучше строка; секунды → `pl.from_epoch(..., time_unit='s').cast(pl.Datetime('ns'))`. Источник задаёт process-env `INTERNAL_STORAGE_EVENT_HUB`; CLI `python -m perseus event-hub add-events events.pq --name my-click`. Повторный импорт одного дня/partition **перезаписывает** файл, не append: заранее объединять порции. |
| Свои типы / признаки | `core/config.py:Event, Feature, Encoder`: ключи `events.<имя>.attributes`, `source`, `max_duration_per_sequence`, `max_events_per_sequence`, `priority`; `features.<признак>.located_in.event/context/artifacts`, `encoder`. Тип события автоматически кодируется признаком `event`. Для нового типа нужен отдельный импорт и запись в events, затем переобучение. `played_ratio` либо переименовать в `played_ratio_pct` с корректным масштабом 0–100, либо объявить свой ple-признак; `is_organic` — id. artist/album можно объявить в событиях и/или artifacts; автоматического join mapping нет. Массивы артистов: `attributes.artist_id.multi: true` + `max_tokens`. Числовые признаки — `ple`, категории — `id`. |
| Basis / labels | `api/prepare_dataset.py:prepare_dataset(config, train_samples, test_samples, train_artifacts=..., test_artifacts=...)`. Basis — client_id, timestamp, target; context/groups опциональны. Для retrieval train target = List[item]; test target = List[{item,relevance}]. Artifacts — таблица `item` и признаков кандидата. Для ranking train target = List[{item, labels:{like:0/1,dislike:0/1,...}}], test аналогично; test/inference дополнительно `items: List[item]`. Отсутствие события не автоматически отрицательная метка: формирование кандидатов/негативов и горизонта labels — наша задача. |
| Обучение только до дня X | Отфильтровать **все обучающие labels/их горизонт**, basis и вычисляемые artifacts до X; передать собственные train/test. Удобнее отдельный EventHub snapshot `<X`, как в пробе. Единого config-ключа train_end_date нет. `core/event_hub.py:Reader._filter_by_time` берёт события строго `< basis.timestamp`, нижняя граница duration включительна; duration отдельно у типа заменяет глобальную duration для этого типа. `api/prepare_dataset.py:_observe_events` обучает препроцессоры на истории до последнего train timestamp пользователя. Нельзя брать глобальные агрегации/словарь из будущего. |
| Произвольные (user, cutoff, item) | Basis: `{client_id, timestamp=cutoff_time, items:[candidate_item]}`. `api.distribute_samples` → для каждой partition `api.make_backbone_embeddings` → `api.embed_artifacts` → `api.make_head_predictions`; `_index` связывает embeddings с basis внутри partition. История строго `<cutoff`. **Ranking штатно поддерживает кандидатов:** `core/tasks/ranking/target.py:Head.predict`; Layer.forward = linear(user_embedding * aggregated_item_embedding), sigmoid отдельно для каждой label. |
| Веса скоринга | Ranking `task.head.label_to_weight: {like:1.0, dislike:-0.5, ...}`; нужно перечислить **все** labels. Score = сумма `weight[label] * sigmoid(logit[label])`; по умолчанию равные 1/число labels. Это веса целевых меток, не вес входного типа события. Само добавление dislike в history не создаёт dislike-label. В retrieval `core/tasks/retrieval/target.py:Head.predict` score = dot(user,item), без label_to_weight. |
| Retrieval-кандидаты | Штатный `extras['items']` добавляет запрошенные кандидаты к top-k; `k=0, only_new=False` возвращает их без top-k. Однако Head всё равно считает матрицу user × переданные artifacts. Для экономии передавать artifacts только кандидатов текущего пакета. `task.head.allow_unknown_items` по умолчанию false: кандидаты вне train known_items пропускаются. true разрешает их при наличии artifacts, но качество OOV не гарантируется. |
| Эмбеддинги | `api/make_backbone_embeddings.py:make_backbone_embeddings` → DataFrame `_index, embeddings: Array[float32,dim]`, это выход backbone readout (`core/backbone/__init__.py:Backbone.forward`, последний токен). `api/embed_artifacts.py:embed_artifacts` возвращает `(original_item_ids, tensor[N,num_features,dim])`, ещё **не** итоговый item-vector. Для retrieval `checkpoint.head.layer(None, embedded_artifacts)`; для ranking `checkpoint.head.layer.item_aggregator(embedded_artifacts[1])`. Суммировать вручную корректно только при item_aggregator=sum. |

При изменении весов ranking **после загрузки** обновлять значения `checkpoint.head.label_to_weight[label] = new_weights[label]`, сохраняя исходные ключи и их порядок: Head.predict сопоставляет выходы linear порядку этого dict. Одних overrides `task.head` при `Trained.load` недостаточно: Head.load восстанавливает собственные сохранённые kwargs. Новые labels требуют переобучения головы.

`priority`: `core/event_hub.py:Reader._sort_by_priority` сортирует по priority ASC, timestamp ASC; `_filter_by_count` оставляет хвост. Следовательно **большее число приоритетнее**, внутри равного priority сохраняются последние события. Сначала применяются лимиты каждого типа, затем общий max_events_per_sequence; итоговый порядок — снова хронологический. Это отбор истории, не вес loss/score.

Обучить модель до X и заморозить историю на X — разные решения. В пробе заморожены оба. Для обновляемой истории после X при инференсе можно направить `INTERNAL_STORAGE_EVENT_HUB` на полный импортированный лог: checkpoint/энкодеры остаются обученными до X, Reader выбирает события строго до конкретного cutoff. При этом учесть OOV и сохранённые в config типы событий. Сведения artist/album из mapping трактуются как статические; у mapping нет времени доступности для проверки исторической доступности этих сведений.

**Подтверждённый нюанс:** `_flatten_unique_events_by_client` объединяет отборы нескольких срезов одного пользователя, а `core/event_store/base.py:EventsReader.read` заново берёт последние max_events по хронологии. Это может выбросить старый high-priority event, который отдельно для этого cutoff сохранялся. Минимальный воспроизводимый тест — `external/api_contract_checks.py` / `external/probe/api_contract_checks.json`. Поэтому в многосрезовом обучении оставить priority одинаковым; отдельные per-type лимиты/duration тоже требуют проверки при объединении. Для инференса helper группирует разные cutoff в отдельные вызовы. Пустая история не получает штатный fallback: такие samples фильтруются, нельзя терять строки при обратном join.


## Измерения и рекомендация для 8 ГБ

Данные: **1000 пользователей, 3646 train-срезов, 710 test-срезов, 3379 кандидатов**; совпавшие uid/timestamp объединены в multi-target. Общая отсечка X = 22447175 секунд на шкале Yambda (`1970-09-17 19:19:35`). Импортировано 126646 like, 15735 dislike и 3222888 listen строго до X. Повторная успешная подготовка заняла 310.7 с без старта Python, первичного импорта EventHub и скачивания пакетов.

| Замер | Эпоха 1 | Эпоха 2 |
|---|---:|---:|
| Обучение, 456 batches | 76.72 с | 92.70 с |
| Проверка | 13.63 с | 16.64 с |
| Recall@100 (только smoke) | 0.0225 | 0.0408 |

Модель — 11297152 параметра. Полный `fit_model` + сохранение: **233.4 с (~3 мин 53 с)**, без импорта Python/загрузки датасета. Пик **torch allocated = 235.24 MiB, reserved = 282 MiB**; это аллокатор данного процесса, не общая VRAM с CUDA/драйвером и другими программами. Во время обучения `nvidia-smi` показывал около 3296 MiB занятой VRAM суммарно. Весь probe, включая установку и исправленные попытки, укладывается в двухчасовой лимит.

**Оценка для 9 тыс. пользователей:** при том же числе срезов на пользователя (~32.8 тыс. train-срезов), sequence256/batch8/negatives128 и сходном размере словарей линейный пересчёт даёт **~12–14 мин обучения + 2–2.5 мин проверки на эпоху**, то есть ~14–17 мин/эпоха. Это экстраполяция, реальный прогон 9 тыс. не запускался. Если обучаться на каждом like, а не на ≤4 последних на пользователя, считать по числу basis-строк: измерено ~39–48 train-срезов/с; одних 9 тыс. uid для оценки недостаточно. Расширение словарей и фоновая нагрузка могут увеличить время и VRAM.

Артефакты проверки: `external/probe/gpu_check.json`, `data_stats.json`, `timings.json`, `train_result.json`, `inference_result.json`, `api_contract_checks.json`, `freeze.txt`, каталог `checkpoint/`; реальные результаты — `external/probe/inference/{scores,user_embeddings,candidate_embeddings}.pq`. Итоговый лог — **`external/inference_audited.log`**; `inference.log` сохраняет первую попытку с чрезмерно строгим assert побитового совпадения. Другие логи — `external/prepare*.log`, `sync*.log`, `train.log`. Smoke-метрики не оценивают качество нашего будущего эксперимента.

Рекомендованный стартовый конфиг — проверяемый `external/probe/config.yaml`: 256 событий, dim128, 2 слоя, batch8, bf16, workers0, sampled negatives128, одинаковые priority. Для полного эксперимента эпохи/early stopping подобрать на собственной временной validation; не переносить smoke-метрики как оценку качества. Увеличивать до sequence512 или dim256 только после замера свободной VRAM и размера словарей: память ID-эмбеддингов и AdamW растёт с числом item/artist/album, а не только пользователей. При проверке чужие процессы занимали около 3 ГБ из 8188 MiB; свободной RAM было 1.43 из 15.78 GiB, фоновая загрузка GPU доходила до 37%. Чужие процессы не останавливали; замеры скорости относятся к этим условиям.
