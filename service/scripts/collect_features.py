"""Прогон восприятия по всем видео датасета и сбор числовых признаков для подбора порогов правил.

Запускается на машине с датасетом (open/videos, open/open_labels.xlsx). Пишет в service/results/:
  perception/<видео>.json — детекции людей/объектов на 3 к/с и треки (без кадров)
  states/<видео>.json     — якоря сцены и признаки состояний на 2 к/с (дверь, сейф, экран, стол, предметы)
  verdicts.json           — результат текущего пайплайна (analyze_video) по каждому видео + разметка
  summary.md              — таблица по видео, время обработки, ошибки
  env.json                — платформа, версии пакетов, процессор
Изображения не сохраняются (кадры датасета публиковать нельзя).

Использование (из папки service/):
  python scripts/collect_features.py                    # все видео из ../open/videos
  python scripts/collect_features.py --names Х-11 Х-12  # только указанные
  python scripts/collect_features.py --skip-items       # без детекции предметов на столе (быстрее)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from pipeline import PIPELINE_VERSION, analyze_video  # noqa: E402
from pipeline.perception import SAMPLE_FPS, cached_perception, perception_to_dict  # noqa: E402
from pipeline.states import STATE_FPS, extract_states  # noqa: E402
from pipeline.video import probe  # noqa: E402


def norm_name(name: str) -> str:
    return name.strip().replace("X", "Х").replace(".mov", "")


def load_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        import openpyxl
    except ImportError:
        print("openpyxl не установлен — разметка не прочитана")
        return {}
    ws = openpyxl.load_workbook(path, read_only=True).active
    out = {}
    for row in list(ws.iter_rows(values_only=True))[1:]:
        if not row or not row[0]:
            continue
        m = re.findall(r"\d+", str(row[1] or ""))
        tc = None
        if m:
            tc = float(m[0]) * 60 + float(m[1]) if "мин" in str(row[1]) and len(m) >= 2 else float(m[0])
        out[norm_name(str(row[0]))] = {"violation": "без" not in str(row[2]).lower(), "timecode": tc}
    return out


def env_info() -> dict:
    info = {"platform": platform.platform(), "python": sys.version, "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "pipeline_version": PIPELINE_VERSION, "sample_fps": SAMPLE_FPS, "state_fps": STATE_FPS}
    for mod in ("cv2", "numpy", "onnxruntime"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    return info


def ascii_path(path: Path, tmpdir: Path) -> Path:
    """На Windows OpenCV может не открыть путь с кириллицей — копируем во временный ASCII-файл."""
    if os.name != "nt" or path.name.isascii():
        return path
    dst = tmpdir / (re.sub(r"[^A-Za-z0-9_.-]", "_", path.stem.encode("ascii", "replace").decode()) + path.suffix)
    if not dst.exists():
        shutil.copy2(path, dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(SERVICE_DIR.parent / "open" / "videos"))
    ap.add_argument("--labels", default=str(SERVICE_DIR.parent / "open" / "open_labels.xlsx"))
    ap.add_argument("--out", default=str(SERVICE_DIR / "results"))
    ap.add_argument("--names", nargs="*", help="обработать только эти видео (без .mov)")
    ap.add_argument("--skip-items", action="store_true", help="не запускать детектор предметов на кропе стола")
    ap.add_argument("--skip-pipeline", action="store_true", help="не запускать analyze_video")
    args = ap.parse_args()

    videos_dir = Path(args.videos)
    out = Path(args.out)
    if not videos_dir.exists():
        raise SystemExit(f"нет папки с видео: {videos_dir} (скопируйте датасет в open/videos)")
    paths = sorted(videos_dir.glob("*.mov"))
    if args.names:
        want = {norm_name(n) for n in args.names}
        paths = [p for p in paths if norm_name(p.stem) in want]
    if not paths:
        raise SystemExit("видео не найдены")
    (out / "perception").mkdir(parents=True, exist_ok=True)
    (out / "states").mkdir(parents=True, exist_ok=True)
    labels = load_labels(Path(args.labels))
    (out / "env.json").write_text(json.dumps(env_info(), ensure_ascii=False, indent=2), encoding="utf-8")
    log_f = open(out / "run.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"=== прогон {time.strftime('%Y-%m-%d %H:%M:%S')} на {platform.node()} ({platform.platform()}), {len(paths)} видео")
    durations: dict[str, float] = {}
    for p in paths:
        try:
            durations[p.name] = probe(str(p)).duration_sec
        except Exception as exc:  # noqa: BLE001
            durations[p.name] = 0.0
            log(f"   !! {p.name}: не читается ({exc})")
    total_video = sum(durations.values())
    log(f"    видео: {len(paths)} файлов, суммарно {total_video / 60:.1f} мин записи; разметка: {len(labels)} строк; "
        f"детектор: YOLOX-m (ONNX Runtime, CPU), {SAMPLE_FPS:.0f} к/с; состояния: {STATE_FPS:.0f} к/с"
        + ("" if not args.skip_items else "; детекция предметов на столе выключена"))
    log(f"    результаты пишутся в {out}")
    verdicts: dict[str, dict] = {}
    vpath = out / "verdicts.json"
    if vpath.exists():
        try:
            verdicts = json.loads(vpath.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            verdicts = {}
    tmpdir = Path(tempfile.mkdtemp(prefix="cashdesk_"))
    t_all = time.time()
    done_video = 0.0

    def make_progress(prefix: str):
        last = {"pct": -100}

        def cb(stage: str, pct: int | None = None) -> None:
            if pct is not None and pct - last["pct"] >= 25:
                last["pct"] = pct
                print(f"      {prefix}: {pct:3d} %", flush=True)
        return cb

    try:
        for i, p in enumerate(paths, 1):
            name = norm_name(p.stem)
            rec: dict = {"file": p.name, "label": labels.get(name)}
            t0 = time.time()
            lbl = labels.get(name) or {}
            lbl_s = ("нарушение" + (f" на {lbl['timecode']:.0f} с" if lbl.get("timecode") is not None else "")) if lbl.get("violation") else ("без нарушения" if lbl else "разметки нет")
            elapsed = time.time() - t_all
            eta = "" if done_video <= 0 else f", осталось ~{(total_video - done_video) * elapsed / done_video / 60:.0f} мин"
            log(f"[{i:2d}/{len(paths)}] {name}  ({durations.get(p.name, 0):.0f} с записи; по разметке: {lbl_s}{eta})")
            try:
                src = ascii_path(p, tmpdir)
                if src != p:
                    log(f"      путь с кириллицей скопирован во временный ASCII-файл {src.name}")
                info = probe(str(src))
                rec["duration_sec"] = info.duration_sec
                log(f"      видео: {info.width}x{info.height}, {info.fps:.2f} к/с, {info.frame_count} кадров, {info.duration_sec:.1f} с")
                # 1) восприятие (люди, объекты, треки)
                tp = time.time()
                log("      шаг 1/3: детекция людей и объектов (YOLOX-m), треки, признак «внутри кассы» ...")
                perc = cached_perception(str(src), info, cache_dir=None, progress=make_progress("детекция"))
                (out / "perception" / f"{name}.json").write_text(json.dumps(perception_to_dict(perc)), encoding="utf-8")
                rec["perception_sec"] = round(time.time() - tp, 1)
                rec["tracks"] = len([t for t in perc.tracks if t.t1 - t.t0 >= 0.5])
                rec["max_inside"] = int(perc.count_inside().max()) if perc.times else 0
                log(f"      готово за {rec['perception_sec']} с: кадров {len(perc.times)}, треков людей {rec['tracks']}, "
                    f"максимум людей внутри одновременно {rec['max_inside']}")
                # 2) состояния сцены
                ts = time.time()
                log("      шаг 2/3: медианный кадр, якоря сцены (стол, сейф, ноутбук, дверь), состояния на 2 к/с"
                    + ("" if args.skip_items else ", предметы на столе") + " ...")
                states = extract_states(str(src), info, perc, progress=make_progress("состояния"), detect_items=not args.skip_items)
                (out / "states" / f"{name}.json").write_text(json.dumps(states, ensure_ascii=False), encoding="utf-8")
                rec["states_sec"] = round(time.time() - ts, 1)
                sc = states["scene"]
                log(f"      готово за {rec['states_sec']} с: стол {'найден ' + str(sc['desk']['bbox']) if sc['desk'] else 'НЕ найден'}, "
                    f"сейф {'найден ' + str(sc['safe']['bbox']) if sc['safe'] else 'не найден'}, "
                    f"ноутбук {'найден' if sc['laptop'] else 'не найден'}, кадров состояний {len(states['frames'])}")
                rec["scene"] = {k: (v if k in ("door_band", "median_brightness") else (None if v is None else v.get("bbox")))
                                for k, v in states["scene"].items() if k in ("desk", "safe", "laptop", "door_band", "median_brightness")}
                # 3) текущий пайплайн
                if not args.skip_pipeline:
                    ta = time.time()
                    log("      шаг 3/3: текущий пайплайн analyze_video (правила T9-контекст и T3) ...")
                    res = analyze_video(str(src))
                    rec["verdict"] = res["verdict"]
                    rec["violations"] = [{k: v[k] for k in ("type_code", "timecode_sec", "end_sec", "confidence") if k in v} for v in res["violations"]]
                    rec["context_events"] = [{k: e[k] for k in ("type_code", "timecode_sec", "end_sec")} for e in res["meta"]["context_events"]]
                    rec["pipeline_sec"] = round(time.time() - ta, 1)
                    ev = ", ".join(f"{v['type_code']} @ {v['timecode_sec']} с" for v in rec["violations"]) or "нет"
                    cx = ", ".join(f"{e['type_code']} @ {e['timecode_sec']} с" for e in rec["context_events"]) or "нет"
                    verdict_ru = "НАРУШЕНИЕ" if rec["verdict"] == "violation" else "без нарушения"
                    match = ""
                    if lbl:
                        match = "  → совпало с разметкой" if (rec["verdict"] == "violation") == bool(lbl.get("violation")) else "  → НЕ совпало с разметкой"
                    log(f"      готово за {rec['pipeline_sec']} с: вердикт {verdict_ru}; события: {ev}; контекст: {cx}{match}")
                rec["ok"] = True
            except Exception as exc:  # noqa: BLE001
                rec["ok"] = False
                rec["error"] = f"{type(exc).__name__}: {exc}"
                log("      !! ОШИБКА на этом видео, перехожу к следующему:")
                log(traceback.format_exc())
            rec["total_sec"] = round(time.time() - t0, 1)
            done_video += durations.get(p.name, 0.0)
            verdicts[name] = rec
            vpath.write_text(json.dumps(verdicts, ensure_ascii=False, indent=1), encoding="utf-8")
            log(f"      итого по {name}: {rec['total_sec']} с ({'OK' if rec['ok'] else 'ОШИБКА'}); обработано {done_video / 60:.1f} из {total_video / 60:.1f} мин записи")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    total = time.time() - t_all
    # --- сводка
    lines = [f"# Прогон {time.strftime('%Y-%m-%d %H:%M')} — {platform.node()} ({platform.platform()})", "",
             f"Видео: {len(paths)}, всего {total / 60:.1f} мин, pipeline {PIPELINE_VERSION}", "",
             "| Видео | Разметка | t разметки | Вердикт пайплайна | События | Треки | Макс. людей внутри | Стол | Сейф | Ноутбук | Время, с | Ошибка |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tp = fp = fn = tn = 0
    for name in sorted(verdicts, key=lambda n: int(re.sub(r"\D", "", n) or 0)):
        r = verdicts[name]
        lbl = r.get("label") or {}
        lv = lbl.get("violation")
        v = r.get("verdict")
        if v is not None and lv is not None:
            if lv and v == "violation":
                tp += 1
            elif lv and v != "violation":
                fn += 1
            elif not lv and v == "violation":
                fp += 1
            else:
                tn += 1
        ev = ", ".join(f"{x['type_code']}@{x['timecode_sec']}" for x in r.get("violations", [])) + \
            ("" if not r.get("context_events") else " ctx:" + ",".join(f"{x['type_code']}@{x['timecode_sec']}" for x in r["context_events"]))
        sc = r.get("scene") or {}
        lines.append(f"| {name} | {'нарушение' if lv else ('чистое' if lv is False else '?')} | {lbl.get('timecode', '')} | {v or '-'} | {ev} | "
                     f"{r.get('tracks', '')} | {r.get('max_inside', '')} | {'да' if sc.get('desk') else 'нет'} | {'да' if sc.get('safe') else 'нет'} | "
                     f"{'да' if sc.get('laptop') else 'нет'} | {r.get('total_sec', '')} | {r.get('error', '')} |")
    lines += ["", f"Текущий пайплайн: TP={tp} FN={fn} FP={fp} TN={tn}", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    errors = [n for n, r in verdicts.items() if not r.get("ok")]
    log("")
    log(f"=== ГОТОВО за {total / 60:.1f} мин. Видео обработано: {len(verdicts) - len(errors)}, с ошибками: {len(errors)}"
        + (f" ({', '.join(errors)})" if errors else ""))
    log(f"    текущий пайплайн против разметки: верно найдено нарушений {tp}, пропущено {fn}, ложных на чистых {fp}, чистых верно {tn}")
    log(f"    файлы: {out / 'summary.md'} (сводная таблица), {out / 'verdicts.json'}, {out / 'perception'}, {out / 'states'}, {out / 'run.log'}")
    log_f.close()


if __name__ == "__main__":
    main()
