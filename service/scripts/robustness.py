"""Проверка устойчивости к другому ракурсу камеры (критерий 5 «Robustness», финальное видео снято с другого угла).

Для набора видео строятся преобразованные варианты (отражение, поворот, перспектива, приближение со сдвигом,
недоэкспонированная картинка, альбомный кадр другого разрешения), по каждому запускается ЖИВОЙ пайплайн
(analyze_video без FEATURES_CACHE: детектор, треки, состояния — всё по самому варианту) и сравнивается с
результатом на оригинале: вердикт, типы событий, сдвиг таймкода, новые (ложные) типы.

Варианты и их признаки пишутся в рабочую папку ВНЕ репозитория (по умолчанию $TMPDIR/cashdesk_robustness):
кадры и видео датасета публиковать нельзя. В репозиторий попадает только таблица с числами (--report).

Признаки, посчитанные живым прогоном (perception/states в формате results/), сохраняются в рабочей папке;
--reuse пересчитывает только сцену и правила по ним (минуты вместо часа) — для проверки правок правил.
Правки слоя восприятия/состояний требуют прогона без --reuse.

  python scripts/robustness.py                              # набор по умолчанию, все преобразования
  python scripts/robustness.py --names Х-17 Х-29 --transforms hflip rot20
  python scripts/robustness.py --reuse --report results/robustness.md
  python scripts/robustness.py --timing Х-14                # время живого прогона самого длинного видео
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
sys.path.insert(0, str(SERVICE_DIR / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# Набор по умолчанию: нарушения, которые пайплайн находит на оригинале (по одному на тип правила),
# и чистые ролики с теми же действиями, выполненными правильно (уход с заблокированным экраном, приём сумки, клиент).
DEFAULT_NAMES = ["Х-17", "Х-30", "Х-28", "Х-15", "Х-18", "Х-46", "Х-11", "Х-29", "Х-16", "Х-3", "Х-21"]
DEFAULT_TRANSFORMS = ["hflip", "rot20", "rotm15", "persp", "zoom", "dark", "land"]


# ------------------------------------------------------------------ преобразования кадра

def _rotate(f: np.ndarray, deg: float) -> np.ndarray:
    h, w = f.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    # постоянная (чёрная) заливка: отражённая граница порождает «фантомных» людей у края
    return cv2.warpAffine(f, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _persp(f: np.ndarray) -> np.ndarray:
    """Трапеция: камера смещена вбок/повёрнута по рысканью (левая сторона сцены дальше и уже)."""
    h, w = f.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0.14 * w, 0.07 * h], [w, 0], [w, h], [0.08 * w, 0.93 * h]])
    return cv2.warpPerspective(f, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _zoom(f: np.ndarray) -> np.ndarray:
    """Приближение ×1.25 со смещением: часть двери слева и низ кадра обрезаны, пропорции сохранены."""
    h, w = f.shape[:2]
    cw, ch = int(0.8 * w), int(0.8 * h)
    x0, y0 = int(0.14 * w), int(0.04 * h)
    return cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)


def _dark(f: np.ndarray) -> np.ndarray:
    """Недоэкспонированная камера с пониженным контрастом (другая автоэкспозиция/освещение)."""
    return cv2.convertScaleAbs(f, alpha=0.72, beta=-12)


def _bright(f: np.ndarray) -> np.ndarray:
    """Переэкспонированная камера: гамма 0.65."""
    lut = (np.clip(((np.arange(256) / 255.0) ** 0.65) * 255.0, 0, 255)).astype(np.uint8)
    return cv2.LUT(f, lut)


def _land(f: np.ndarray) -> np.ndarray:
    """Альбомный кадр 1280×720 другого разрешения: помещение (без переднего стола) крупнее, по бокам поля."""
    h, w = f.shape[:2]
    crop = f[int(0.04 * h):int(0.68 * h)]
    ch, cw = crop.shape[:2]
    nh = 720
    nw = int(round(cw * nh / ch))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((720, 1280, 3), np.uint8)
    x0 = (1280 - nw) // 2
    out[:, x0:x0 + nw] = crop
    return out


TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "hflip": lambda f: cv2.flip(f, 1),
    "rot20": lambda f: _rotate(f, 20.0),
    "rotm15": lambda f: _rotate(f, -15.0),
    "persp": _persp,
    "zoom": _zoom,
    "dark": _dark,
    "bright": _bright,
    "land": _land,
}


def make_variant(src: Path, kind: str, dst: Path) -> Path:
    """Записывает преобразованный вариант (mp4v, тот же fps). Существующий файл переиспользуется."""
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, f = cap.read()
    if not ok:
        raise RuntimeError(f"не читается {src}")
    g = TRANSFORMS[kind](f)
    tmp = dst.with_suffix(".part.mp4")
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (g.shape[1], g.shape[0]))
    while ok:
        wr.write(TRANSFORMS[kind](f))
        ok, f = cap.read()
    wr.release()
    cap.release()
    tmp.replace(dst)
    return dst


# ------------------------------------------------------------------ воркер

_WORK: Optional[Path] = None
_REUSE = False


def _init_worker(work: str, reuse: bool, threads: int) -> None:
    global _WORK, _REUSE
    _WORK, _REUSE = Path(work), reuse
    os.environ["EVIDENCE_FRAMES"] = "0"
    os.environ.pop("FEATURES_CACHE", None)
    import pipeline.detect as det

    det._detector = det.YoloxDetector(threads=threads)   # ограничиваем потоки ONNX на процесс-воркер


def _feature_stem(path: Path) -> str:
    return path.stem.replace("X", "Х")


def _run_one(name: str, kind: str, src: str) -> dict:
    import pipeline
    from pipeline.perception import perception_to_dict

    assert _WORK is not None
    src_p = Path(src)
    path = src_p if kind == "orig" else make_variant(src_p, kind, _WORK / "videos" / f"{name}__{kind}.mp4")
    feats = _WORK / "features"
    stem = _feature_stem(path)
    have = (feats / "perception" / f"{stem}.json").exists() and (feats / "states" / f"{stem}.json").exists()
    t0 = time.time()
    live = not (_REUSE and have)
    if live:
        # живой прогон: FEATURES_CACHE не задан, признаки считаются по видео; побочно сохраняем их для --reuse
        os.environ.pop("FEATURES_CACHE", None)
        orig = pipeline._perceive

        def dump(p, info, progress):
            perc, states = orig(p, info, progress)
            (feats / "perception").mkdir(parents=True, exist_ok=True)
            (feats / "states").mkdir(parents=True, exist_ok=True)
            (feats / "perception" / f"{stem}.json").write_text(json.dumps(perception_to_dict(perc)), encoding="utf-8")
            (feats / "states" / f"{stem}.json").write_text(json.dumps(states), encoding="utf-8")
            return perc, states

        pipeline._perceive = dump
        try:
            res = pipeline.analyze_video(str(path))
        finally:
            pipeline._perceive = orig
    else:
        os.environ["FEATURES_CACHE"] = str(feats)
        try:
            res = pipeline.analyze_video(str(path))
        finally:
            os.environ.pop("FEATURES_CACHE", None)
    sc = res["meta"].get("scene") or {}
    return {
        "name": name, "kind": kind, "live": live, "elapsed": round(time.time() - t0, 1),
        "verdict": res["verdict"],
        "events": [(v["type_code"], float(v["timecode_sec"])) for v in res["violations"]],
        "context": [(e["type_code"], float(e["timecode_sec"])) for e in res["meta"].get("context_events") or []],
        "resolution": res["meta"].get("resolution"),
        "perception": res["meta"].get("perception"),
        "camera": sc.get("camera"),
        "cashier_tracks": sc.get("cashier_tracks"), "visitor_tracks": sc.get("visitor_tracks"),
        "notes": sc.get("notes"),
    }


# ------------------------------------------------------------------ сравнение и отчёт

def _cell(r: Optional[dict]) -> str:
    if r is None:
        return "—"
    if r.get("error"):
        return "ошибка"
    if not r["events"]:
        return "чисто"
    return ", ".join(f"{c}@{t:.1f}" for c, t in r["events"])


def compare(rows: list[dict], labels: dict) -> dict:
    by = {(r["name"], r["kind"]): r for r in rows}
    names = sorted({r["name"] for r in rows}, key=lambda s: int(re.sub(r"\D", "", s) or 0))
    kinds = [k for k in ["orig"] + list(TRANSFORMS) if any(r["kind"] == k for r in rows)]
    summ = {}
    for k in kinds:
        if k == "orig":
            continue
        n = same_v = same_t = fp_clean = miss = new_types = lost = 0
        dts: list[float] = []
        for nm in names:
            o, v = by.get((nm, "orig")), by.get((nm, k))
            if o is None or v is None or o.get("error") or v.get("error"):
                continue
            n += 1
            lv = (labels.get(nm) or {}).get("violation")
            same_v += o["verdict"] == v["verdict"]
            to = {c for c, _ in o["events"]}
            tv = {c for c, _ in v["events"]}
            same_t += to == tv
            new_types += len(tv - to)
            lost += len(to - tv)
            if lv is False and v["verdict"] == "violation":
                fp_clean += 1
            if lv and v["verdict"] != "violation":
                miss += 1
            for c, t in o["events"]:
                tt = [x for cc, x in v["events"] if cc == c]
                if tt:
                    dts.append(min(abs(x - t) for x in tt))
        summ[k] = {
            "n": n, "verdict_same": same_v, "types_same": same_t, "new_types": new_types, "lost_types": lost,
            "fp_clean": fp_clean, "missed_viol": miss,
            "dt_median": round(statistics.median(dts), 2) if dts else None,
            "dt_max": round(max(dts), 2) if dts else None,
        }
    return {"names": names, "kinds": kinds, "by": by, "summary": summ}


def _summary_table(summ: dict) -> list[str]:
    L = [
        "| преобразование | видео | вердикт как на оригинале | типы как на оригинале | новые типы | потерянные типы | ложные на чистых | пропуски нарушений | Δt медиана, с | Δt макс, с |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, s in summ.items():
        L.append(
            f"| {k} | {s['n']} | {s['verdict_same']}/{s['n']} | {s['types_same']}/{s['n']} | {s['new_types']} | {s['lost_types']} | "
            f"{s['fp_clean']} | {s['missed_viol']} | {'' if s['dt_median'] is None else s['dt_median']} | {'' if s['dt_max'] is None else s['dt_max']} |"
        )
    tot = {key: sum(s[key] for s in summ.values()) for key in ("n", "verdict_same", "types_same", "fp_clean", "missed_viol", "new_types", "lost_types")}
    L.append(f"| **всего** | {tot['n']} | {tot['verdict_same']}/{tot['n']} | {tot['types_same']}/{tot['n']} | {tot['new_types']} | {tot['lost_types']} | {tot['fp_clean']} | {tot['missed_viol']} | | |")
    return L


def _matrix(cmp: dict, labels: dict) -> list[str]:
    names, kinds, by = cmp["names"], cmp["kinds"], cmp["by"]
    L = ["| видео | разметка | " + " | ".join(kinds) + " |", "|---|---|" + "---|" * len(kinds)]
    for nm in names:
        lb = labels.get(nm) or {}
        lab = "чисто" if lb.get("violation") is False else (f"нарушение @{lb.get('timecode')}" if lb.get("violation") else "?")
        L.append(f"| {nm} | {lab} | " + " | ".join(_cell(by.get((nm, k))) for k in kinds) + " |")
    return L


def write_report(cmp: dict, labels: dict, path: Path, timing: Optional[dict], header: str,
                 baseline: Optional[dict] = None, live_times: Optional[list[float]] = None) -> None:
    names, kinds, by, summ = cmp["names"], cmp["kinds"], cmp["by"], cmp["summary"]
    n_clean = sum(1 for nm in names if (labels.get(nm) or {}).get("violation") is False)
    n_viol = sum(1 for nm in names if (labels.get(nm) or {}).get("violation"))
    L = [
        "# Устойчивость к ракурсу камеры: живой прогон на преобразованных видео",
        "",
        header,
        "",
        f"Видео: {len(names)} (нарушений {n_viol}, чистых {n_clean}); преобразований: {len(kinds) - 1}. "
        "Каждый вариант прогнан живым пайплайном (`analyze_video` без `FEATURES_CACHE`): детекции, треки и состояния "
        "считаются по самому варианту. Кадры и видео не сохраняются в репозитории, только числа.",
        "",
        "Преобразования: `hflip` — зеркальное отражение; `rot20`/`rotm15` — поворот на +20°/−15° (чёрные углы); "
        "`persp` — трапеция (камера смещена вбок); `zoom` — приближение ×1.25 со смещением (обрезаны часть двери и низ); "
        "`dark` — недоэкспонированная картинка (×0.72 −12); `bright` — переэкспонированная (гамма 0.65); "
        "`land` — альбомный кадр 1280×720, помещение крупнее и с полями.",
        "",
        "## Сводка по преобразованиям (текущий код)",
        "",
    ]
    L += _summary_table(summ)
    if baseline:
        L += ["", "## Сводка до исправлений (тот же прогон на коде без самокалибровки камеры)", ""]
        L += _summary_table(baseline["summary"])
    L += ["", "## По видео (тип@таймкод, с), текущий код", ""] + _matrix(cmp, labels)
    if baseline:
        L += ["", "## По видео до исправлений", ""] + _matrix(baseline, labels)
    cams = [(r["name"], r["kind"], r.get("camera")) for r in by.values() if r.get("camera")]
    if cams:
        L += ["", "## Самокалибровка камеры по ролику (meta.scene.camera)", "",
              "Режим теста «внутри кассы» (`fixed` — геометрия эталонной камеры согласуется с роликом, `relative` — "
              "относительный тест по столу и росту людей), масштаб сцены к кадру 464×848, коэффициент экспозиции и доля "
              "уверенных наблюдений, где геометрия эталонной камеры противоречит относительному тесту.", "",
              "| видео | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
        for nm in names:
            cells = []
            for k in kinds:
                c = (by.get((nm, k)) or {}).get("camera") or {}
                cells.append(f"{c.get('mode', '—')} ×{c.get('scale', '')} g{c.get('gain', '')} c{c.get('conflict', '')}" if c else "—")
            L.append(f"| {nm} | " + " | ".join(cells) + " |")
    if timing:
        L += [
            "",
            "## Время живого прогона",
            "",
            f"Видео {timing['name']} ({timing['duration_sec']} с, самое длинное в датасете): {timing['elapsed_sec']} с полного "
            f"`analyze_video` без кеша признаков (без VLM, с кадрами-доказательствами), процессорное время "
            f"{timing.get('cpu_sec')} с. Машина: {timing['machine']}; средняя загрузка (load average) во время замера: "
            f"{timing['load']}. Начало этапов, с от старта: "
            + ", ".join(f"{k} {v}" for k, v in (timing.get("stages") or {}).items()) + ".",
        ]
        if timing.get("note"):
            L += ["", timing["note"]]
    el = live_times or [r["elapsed"] for r in by.values() if r.get("live") and r.get("elapsed")]
    if el:
        L += ["", f"Живые прогоны вариантов: {len(el)}, время на вариант: медиана {statistics.median(el):.0f} с, "
              f"максимум {max(el):.0f} с (параллельно несколько воркеров, машина под нагрузкой — для оценки лимита 5 мин "
              f"см. замер выше)."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def measure_timing(name: str, videos: Path) -> dict:
    """Время полного живого analyze_video (без кеша, с кадрами-доказательствами) на одном видео."""
    import platform

    os.environ.pop("FEATURES_CACHE", None)
    os.environ.pop("EVIDENCE_FRAMES", None)
    from collect_features import norm_name

    from pipeline import analyze_video

    src = next(p for p in videos.glob("*.mov") if norm_name(p.stem) == norm_name(name))
    load0 = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    stages: dict[str, float] = {}
    t0 = time.time()
    c0 = time.process_time()

    def progress(stage: str, pct=None) -> None:
        stages.setdefault(stage, round(time.time() - t0, 1))   # момент начала этапа, с от старта

    res = analyze_video(str(src), progress=progress)
    el = time.time() - t0
    cpu = time.process_time() - c0
    load1 = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    return {
        "name": norm_name(name), "duration_sec": res["meta"]["duration_sec"], "elapsed_sec": round(el, 1),
        "cpu_sec": round(cpu, 1), "stages": stages,
        "verdict": res["verdict"], "events": [(v["type_code"], v["timecode_sec"]) for v in res["violations"]],
        "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPU",
        "load": None if load0 is None else f"{load0:.1f}→{load1:.1f}",
    }


def main() -> None:
    from collect_features import load_labels, norm_name

    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(SERVICE_DIR.parent / "open" / "videos"))
    ap.add_argument("--labels", default=str(SERVICE_DIR.parent / "open" / "open_labels.xlsx"))
    ap.add_argument("--names", nargs="*", default=DEFAULT_NAMES)
    ap.add_argument("--transforms", nargs="*", default=DEFAULT_TRANSFORMS, choices=list(TRANSFORMS))
    ap.add_argument("--work-dir", default=str(Path(tempfile.gettempdir()) / "cashdesk_robustness"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2, help="потоков ONNX Runtime на воркер")
    ap.add_argument("--reuse", action="store_true", help="переиспользовать признаки живых прогонов из рабочей папки")
    ap.add_argument("--report", default=None, help="markdown-таблица (например results/robustness.md)")
    ap.add_argument("--json", default=None, help="сырые результаты (по умолчанию <work-dir>/robustness.json)")
    ap.add_argument("--timing", default=None, help="замерить время живого прогона одного видео (например Х-14)")
    ap.add_argument("--timing-json", default=None, help="сохранённый замер времени для отчёта")
    ap.add_argument("--header", default="", help="строка-пояснение в начале отчёта")
    ap.add_argument("--baseline-json", default=None, help="результаты прогона до исправлений (--json прошлого прогона) для сравнения")
    ap.add_argument("--live-times-json", default=None, help="json живого прогона, из которого брать время на вариант (при --reuse)")
    ap.add_argument("--from-json", default=None, help="не запускать прогоны, построить отчёт по сохранённому --json")
    args = ap.parse_args()

    videos = Path(args.videos)
    work = Path(args.work_dir)
    if SERVICE_DIR.parent in work.resolve().parents and "results" not in work.parts:
        print("рабочая папка внутри репозитория — варианты видео нельзя коммитить; укажите --work-dir вне репо")
    work.mkdir(parents=True, exist_ok=True)
    timing = None
    if args.timing:
        timing = measure_timing(args.timing, videos)
        print("timing:", timing, flush=True)
        (work / "timing.json").write_text(json.dumps(timing, ensure_ascii=False), encoding="utf-8")
        if not args.report:
            return
    elif args.timing_json and Path(args.timing_json).exists():
        timing = json.loads(Path(args.timing_json).read_text(encoding="utf-8"))

    labels = load_labels(Path(args.labels))

    def report(rows: list[dict]) -> None:
        cmp = compare(rows, labels)
        base = None
        if args.baseline_json and Path(args.baseline_json).exists():
            base = compare(json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))["rows"], labels)
        lt = None
        if args.live_times_json and Path(args.live_times_json).exists():
            lt = [r["elapsed"] for r in json.loads(Path(args.live_times_json).read_text(encoding="utf-8"))["rows"]
                  if r.get("live") and r.get("elapsed")]
        write_report(cmp, labels, Path(args.report), timing, args.header, base, lt)
        print("отчёт:", args.report)

    if args.from_json:
        rows = json.loads(Path(args.from_json).read_text(encoding="utf-8"))["rows"]
        for r in rows:
            r["events"] = [tuple(e) for e in r.get("events") or []]
        if args.report:
            report(rows)
        return
    want = [norm_name(n) for n in args.names]
    srcs = {norm_name(p.stem): p for p in videos.glob("*.mov")}
    jobs = [(nm, k, str(srcs[nm])) for nm in want if nm in srcs for k in ["orig"] + list(args.transforms)]
    missing = [nm for nm in want if nm not in srcs]
    if missing:
        print("нет видео:", missing)
    # варианты пишем заранее в главном процессе (декодирование/кодирование дешевле детекции)
    for nm, k, src in jobs:
        if k != "orig":
            make_variant(Path(src), k, work / "videos" / f"{nm}__{k}.mp4")
    print(f"{len(jobs)} прогонов, воркеров {args.workers}, reuse={args.reuse}, рабочая папка {work}", flush=True)
    rows: list[dict] = []
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=max(1, args.workers), initializer=_init_worker,
                             initargs=(str(work), args.reuse, args.threads)) as ex:
        futs = {ex.submit(_run_one, nm, k, src): (nm, k) for nm, k, src in jobs}
        for fu in as_completed(futs):
            nm, k = futs[fu]
            try:
                r = fu.result()
            except Exception as exc:  # noqa: BLE001
                r = {"name": nm, "kind": k, "error": f"{type(exc).__name__}: {exc}", "events": [], "verdict": None}
            rows.append(r)
            print(f"{nm:6} {k:7} {_cell(r):40} {r.get('elapsed', '')}s live={r.get('live')}", flush=True)
    cmp = compare(rows, labels)
    for k, s in cmp["summary"].items():
        print(k, s)
    print(f"время: {time.time() - t_all:.0f} с")
    out_json = Path(args.json) if args.json else work / "robustness.json"
    out_json.write_text(json.dumps({"rows": rows, "summary": cmp["summary"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.report:
        report(rows)


if __name__ == "__main__":
    main()
