"""Render the short final report from completed artifacts; never reads labels."""
import json
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[2]
(ROOT/'logs/experiment').mkdir(parents=True, exist_ok=True)
BASE=ROOT/'external/perseus_stage2';RUN=ROOT/'artifacts/runs/r3_perseus'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    cost=read(RUN/'cost.json');selection=read(BASE/'selection.json');verification=read(BASE/'verification.json')
    assert verification['success']
    validation=read(RUN/'validation.json')
    outputs=cost['inference']
    lines=['# Perseus: этап 2 — результат',
           f"Обновлено {time.strftime('%Y-%m-%d %H:%M:%S')} (локальное время). **Пункты 1–4 выполнены**, R3 OOT скорится только при наличии `artifacts/freeze.json`.",
           f"**R3 val-A GAUC = {validation['gauc']:.8f}** на {validation['rows']} исходных показах / {validation['groups']} группах с двумя классами. "
           'GAUC внутри (uid,day), вес — число показов; канонический `src/tbrec/metrics.py:summary`. '
           'Readback файла pred_val совпал с метрикой выбранной эпохи (delta=0). Метрики val-B и OOT не считались.',
           '| Модель | Метки train | Скоры | Строк / null |', '|---|---|---|---|']
    for kind,train_days,output_path in [
        ('r3_val','273–286','artifacts/runs/r3_perseus/pred_val.parquet'),
        ('r4_val','259–272','artifacts/features/perseus_val.parquet'),
        ('r4_oot','266–279','artifacts/features/perseus_oot.parquet')]:
        x=outputs[kind]
        lines.append(f"| {kind} | {train_days} | `{output_path}`; дни {x['days'][0]}–{x['days'][1]} | {x['rows']} / {x['missing']} ({x['missing_fraction']:.4%}) |")
    if 'r3_oot' in outputs:
        x=outputs['r3_oot'];lines.append(f"| r3_oot | 280–293 | `artifacts/runs/r3_perseus/pred_oot.parquet`; 294–300 | {x['rows']} / {x['missing']} ({x['missing_fraction']:.4%}) |")
    else:lines.append('| r3_oot | 280–293 | checkpoint готов; **294–300 не проскорены: freeze.json отсутствует** | — |')
    lines += ['',f"Заморожено: **{selection['sequence']} событий, dim {selection['dim']}, ModernBERT 2 слоя / 4 головы, {selection['chosen_epochs']} эпохи**, bf16, batch16, AdamW lr=3e-4, dropout=0.1. "
              'Голова ranking, единственная метка listen_plus, score=sigmoid(logit) в FP32, вес 1.0; allow_unknown_items=true. '
              'PLE ≤32 bins, item vocabulary ≤250000, artist/album ≤100000 с UNK. Показов не сэмплировали: повторы схлопнуты средним y; случаи без истории исключены только из train, при инференсе возвращены с null.',
              'Выбор только по val-A 287–293: 256/128 — GAUC 0.548574, лучшая эпоха 2 (проверены 3); '
              '512/256 — GAUC 0.546727, лучшая эпоха 1 (остановлен после 2). После выбора конфиг/число эпох не менялись. '
              'Снимок: `artifacts/runs/r3_perseus/selected_config.yaml`, SHA256 и выбор: `external/perseus_stage2/selection.json`. '
              'В снимке остаются параметры поиска (max_epochs=3, early stopping); для повторного fixed-fit использовать --epochs 2 без --select.',
              '', 'Данные/контроль:',
              '- Для всех строк история строго t < day*86400−6395; события текущего дня не видны. Listen включает skips, played_ratio и organic; используются artist/album, like/dislike. Unlike/undislike не включались.',
              '- Дни <273: алгоритмические listen из events, y=(pr>=50). Дубликаты (uid,day,item) в train — mean(y), при инференсе один скор возвращён всем row. Null не удалены: это пустая история; при метрике используется только фиксированный 0.5 вместо null.',
              '- Стабильный порядок (t, исходный индекс события), отдельный last-N для каждого среза: ловушка priority не возникает. Препроцессоры/словари fit только на train; rolling-история при инференсе обновляется до cutoff, веса фиксированы. Длина трека взята как статический атрибут из processed/items.parquet, artist/album — готовые статические метаданные.',
              '- Скоры вычисляются нативными backbone и ranking Layer; загрузчик батчей и callback GAUC адаптированы для дневного кэша. Код фреймворка не менялся. Сверка с Head.predict: расхождение ≤0.001954 из-за округления sigmoid в bf16 в штатном Head; экспорт использует FP32 sigmoid. Вся математика головы/веса те же.',
              '- Метки 294–300 не читались и не вычислялись из pr. Инференс читает отдельный rows_no_labels. Проверки ключей, повторов, диапазона вероятностей, конфигов и checkpoint headers прошли: `external/perseus_stage2/verification.json`; аудит окон — `label_access.jsonl` рядом. Raw/cache contract — `cache/contract_checks.json`.',
              '', '| Модель | Подготовка, с | Fit + save, с | Peak torch allocated / reserved, MiB | Checkpoint, MiB |',
              '|---|---:|---:|---:|---:|']
    for model in [selection['model'],'r4_val','r4_oot','r3_oot']:
        x=cost['models'][model]
        lines.append(f"| {model} | {x['prepare_seconds']:.1f} | {x['fit_save_seconds']:.1f} | {x['peak_allocated_bytes']/2**20:.1f} / {x['peak_reserved_bytes']/2**20:.1f} | {x['checkpoint_bytes']/2**20:.1f} |")
    lines += ['',f"Общий дневной кэш: {cost['global_prepare']['seconds']:.1f} с. У R3 val Fit включает поиск по 3 эпохам/проверки; остальные модели — ровно 2 эпохи без eval. "
              f"Дополнительный trial 512/256: {cost['models']['r3_large']['fit_save_seconds']:.1f} с. Один технический перезапуск из-за float y в NDCG-части canonical summary: "
              '+184.8 с обучения; исправлен входной dtype, источник метрики не менялся. Полные затраты/попытки — `artifacts/runs/r3_perseus/cost.json`.',
              'Инференс, с / 1000 исходных строк (без загрузки модели, с записью результата): '+', '.join(f"{kind}={x['inference_seconds_per_1000_rows']:.3f}" for kind,x in outputs.items())+'.',
              'Torch peak — аллокатор нашего процесса, без драйвера/чужих программ. GPU общий (чужие процессы ≈3 ГБ); CPU affinity 11–15 и below-normal унаследованы, workers=0, библиотеки ≤4 потоков. Приоритет/affinity и чужие процессы не менялись. Замеры относятся к общей нагрузке ПК.',
              'Кандидаты вне train-target dictionary / UNK item-encoder, доля строк: '+', '.join(f"{kind}: {x['outside_train_item_dictionary_fraction']:.2%} / {x['item_encoder_unk_row_fraction']:.2%}" for kind,x in outputs.items())+'.',
              '', 'R3 OOT checkpoint: `external/perseus_stage2/r3_oot/checkpoint/`. Команда ниже повторно проверяет freeze.json, без него возвращается до чтения данных инференса:',
              '```powershell', '. external/probe_env.ps1', '& $ProbePython external/perseus_adapter/stage2.py score --model r3_oot --start 294 --end 300 --kind r3_oot', '```',
              'Скрипты — `external/perseus_adapter/`. Повтор pipeline: prepare_stage2.py → stage2.py train --model r3_small --start 273 --end 286 --seq 256 --dim 128 --epochs 3 --select → run_stage2.py. '
              'Для инференса нужны checkpoint, общий `external/perseus_stage2/cache/` и event_codes.npy соответствующей модели. Логи — `logs/experiment/perseus_stage2_*.log`.']
    if 'h6' in cost:
        h=cost['h6'];m=h['val_A'];i=h['inference'];t=h['typed_history']
        lines += ['',f"**H6 готов:** val-A GAUC={m['gauc']:.8f}, delta к основной ветке={m['delivered_minus_selection_gauc']:+.8f}. "
                  f"`artifacts/runs/r3_perseus_typed/pred_val.parquet`, {i['rows']} строк, null={i['missing']} ({i['missing_fraction']:.4%}). "
                  'Квоты listen/like/dislike=176/64/16, горизонты=30/365/90 дней. Те же 256/128 и 2 эпохи, без нового подбора; основная ветка не переобучалась/не менялась. '
                  f"Fit+save={h['model']['fit_save_seconds']:.1f} с, typed-cache={t['seconds']:.1f} с; `artifacts/runs/r3_perseus_typed/cost.json`. "
                  'Это сравнение квот вместе с горизонтами, не изолированный эффект одного priority.']
    document='\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n'
    (ROOT/'logs/experiment/perseus_stage2_report.md').write_text(document,encoding='utf-8')

if __name__=='__main__':main()
