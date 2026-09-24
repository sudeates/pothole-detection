"""Render a pothole demo with raw detections and optional ByteTrack filtering.

No training or labelled evaluation takes place. All reported metrics describe
drawn boxes in an unlabelled video, not detection accuracy.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
SWEEP_HITS = (1, 3, 5, 8)
SWEEP_OVERLAP = (0.5, 0.6, 0.8)
RAW_CONF = 0.25
TRACK_CONF = 0.10  # ByteTrack needs its low-score pool down to track_low_thresh.
MATCH_IOU = 0.30
IMAGE_SIZE = 640


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    conf: float
    track_id: int | None = None
    held: bool = False


@dataclass
class TrackState:
    hits: int
    first_seen: int
    last_seen: int
    box: tuple[float, float, float, float]
    conf: float
    first_drawn: int | None = None
    last_drawn: int | None = None


class TrackFilter:
    """Confirm, hold and smooth tracks; optionally deduplicate drawn boxes."""

    def __init__(self, min_hits: int, hold: int, alpha: float,
                 overlap_thresh: float | None = None) -> None:
        self.min_hits = min_hits
        self.hold = hold
        self.alpha = alpha
        self.overlap_thresh = overlap_thresh
        self.states: dict[int, TrackState] = {}
        self.previous_drawn: set[int] = set()
        self.frames = 0
        self.frames_with_boxes = 0
        self.boxes_drawn = 0
        self.disappearances = 0
        self.handoffs = 0
        self.suppressed = {'held_active': 0, 'active_active': 0, 'held_held': 0}
        self.last_handoffs = 0
        self.last_suppressed = dict.fromkeys(self.suppressed, 0)

    def _deduplicate(self, candidates: list[Detection]
                     ) -> tuple[list[Detection], dict[int, tuple[str, int]]]:
        """Keep active over held; then prefer established or recently seen IDs."""
        if self.overlap_thresh is None:
            return candidates, {}
        threshold = self.overlap_thresh
        active = [det for det in candidates if not det.held]
        held = [det for det in candidates if det.held]
        suppressed: dict[int, tuple[str, int]] = {}

        # Rule a: identify held boxes covered by an active box. If that active
        # later loses rule b, keep the held box unless another survivor covers it.
        held_near_active = {
            det.track_id for det in held
            if any(intersection_over_smaller(det.box, other.box) >= threshold
                   for other in active)
        }

        # Rule b: higher hit count wins; ties go to the older track.
        active.sort(key=lambda det: (-self.states[det.track_id].hits,
                                     self.states[det.track_id].first_seen,
                                     det.track_id))
        kept_active: list[Detection] = []
        for det in active:
            winner = next((other for other in kept_active
                           if intersection_over_smaller(det.box, other.box) >= threshold), None)
            if winner is None:
                kept_active.append(det)
            else:
                suppressed[det.track_id] = ('active_active', winner.track_id)

        remaining_held: list[Detection] = []
        for det in held:
            if det.track_id in held_near_active:
                winner = next((other for other in kept_active
                               if intersection_over_smaller(det.box, other.box) >= threshold), None)
                if winner is not None:
                    suppressed[det.track_id] = ('held_active', winner.track_id)
                    continue
            remaining_held.append(det)

        # Rule c: the held box observed most recently wins.
        remaining_held.sort(key=lambda det: (-self.states[det.track_id].last_seen,
                                             -self.states[det.track_id].hits,
                                             self.states[det.track_id].first_seen,
                                             det.track_id))
        kept_held: list[Detection] = []
        for det in remaining_held:
            winner = next((other for other in kept_held
                           if intersection_over_smaller(det.box, other.box) >= threshold), None)
            if winner is None:
                kept_held.append(det)
            else:
                suppressed[det.track_id] = ('held_held', winner.track_id)
        return kept_active + kept_held, suppressed

    def update(self, observations: list[Detection], frame: int) -> tuple[list[Detection], int]:
        seen: set[int] = set()
        for det in observations:
            assert det.track_id is not None
            track_id = det.track_id
            if track_id in seen:
                raise ValueError(f'Duplicate ByteTrack id {track_id} in frame {frame}')
            seen.add(track_id)
            state = self.states.get(track_id)
            if state is None:
                self.states[track_id] = TrackState(1, frame, frame, det.box, det.conf)
            else:
                state.hits += 1
                state.last_seen = frame
                state.box = tuple(self.alpha * new + (1 - self.alpha) * old
                                  for new, old in zip(det.box, state.box))
                state.conf = det.conf

        candidates: list[Detection] = []
        for track_id, state in self.states.items():
            if state.hits < self.min_hits or frame - state.last_seen > self.hold:
                continue
            held = frame != state.last_seen
            candidates.append(Detection(state.box, state.conf, track_id, held))

        drawn, suppressed = self._deduplicate(candidates)
        current_drawn = {det.track_id for det in drawn}
        for det in drawn:
            state = self.states[det.track_id]
            if state.first_drawn is None:
                state.first_drawn = frame
            state.last_drawn = frame

        missing = self.previous_drawn - current_drawn
        handoffs = sum(track_id in suppressed and suppressed[track_id][1] in current_drawn
                       for track_id in missing)
        disappeared = len(missing) - handoffs
        self.last_handoffs = handoffs
        self.handoffs += handoffs
        self.last_suppressed = {rule: sum(item[0] == rule for item in suppressed.values())
                                for rule in self.suppressed}
        for rule, count in self.last_suppressed.items():
            self.suppressed[rule] += count
        self.previous_drawn = current_drawn
        self.disappearances += disappeared
        self.frames += 1
        self.frames_with_boxes += bool(drawn)
        self.boxes_drawn += len(drawn)
        return drawn, disappeared

    def summary(self, fps: float) -> dict:
        approved = [s for s in self.states.values() if s.first_drawn is not None]
        lifetimes = [s.last_drawn - s.first_drawn + 1 for s in approved]
        minutes = self.frames / fps / 60
        return {
            'frames_with_boxes_pct': 100 * self.frames_with_boxes / self.frames if self.frames else 0,
            'boxes_per_frame': self.boxes_drawn / self.frames if self.frames else 0,
            'disappearances': self.disappearances,
            'disappearances_per_minute': self.disappearances / minutes if minutes else 0,
            'handoffs': self.handoffs,
            'handoffs_per_minute': self.handoffs / minutes if minutes else 0,
            'suppressed': dict(self.suppressed),
            'suppressed_total': sum(self.suppressed.values()),
            'tracks_total': len(self.states),
            'tracks_never_drawn': len(self.states) - len(approved),
            'tracks_below_min_hits': sum(state.hits < self.min_hits
                                         for state in self.states.values()),
            'tracks_confirmed_never_drawn': sum(state.hits >= self.min_hits and
                                                state.first_drawn is None
                                                for state in self.states.values()),
            'approved_lifetime_median_frames': median(lifetimes) if lifetimes else None,
        }


def intersection_over_union(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union else 0.0


def intersection_over_smaller(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection divided by the smaller box area; catches containment."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    smaller = min(area_a, area_b)
    return overlap / smaller if smaller else 0.0


def raw_disappearances(previous: list[Detection], current: list[Detection]) -> int:
    """Count unmatched boxes from the preceding frame via greedy IoU matching."""
    candidates = sorted(
        ((intersection_over_union(a.box, b.box), i, j)
         for i, a in enumerate(previous) for j, b in enumerate(current)),
        reverse=True,
    )
    used_previous: set[int] = set()
    used_current: set[int] = set()
    for overlap, i, j in candidates:
        if overlap < MATCH_IOU:
            break
        if i not in used_previous and j not in used_current:
            used_previous.add(i)
            used_current.add(j)
    return len(previous) - len(used_previous)


def unpack_boxes(result, *, tracked: bool) -> list[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    if tracked and boxes.id is None:
        # No active tracks are emitted on this frame.
        return []
    coordinates = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    ids = boxes.id.int().cpu().tolist() if tracked else [None] * len(boxes)
    return [Detection(tuple(map(float, box)), float(conf), track_id)
            for box, conf, track_id in zip(coordinates, confidences, ids)]


def draw(frame, detections: list[Detection], title: str, *, tracked: bool):
    image = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in det.box)
        color = (0, 180, 255) if det.held else ((55, 205, 55) if tracked else (0, 0, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f'#{det.track_id} {det.conf:.2f}' if tracked else f'{det.conf:.2f}'
        if det.held:
            label += ' hold'
        y_text = max(17, y1 - 5)
        cv2.putText(image, label, (max(0, x1), y_text), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, color, 2, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 210), 28), (20, 20, 20), -1)
    cv2.putText(image, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def video_writer(path: Path, fps: float, size: tuple[int, int]):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f'Could not open MP4 writer: {path}')
    return writer


def render_report(*, source: Path, weights: Path, fps: float, width: int, height: int,
                  raw: dict, base_track: dict | None, track: dict | None,
                  sweep: dict[int, dict], overlap_sweep: dict[float, dict],
                  min_hits: int, hold: int, alpha: float, overlap_thresh: float,
                  check_frames: tuple[int, ...]) -> str:
    def row(label: str, item: dict, note: str) -> str:
        return (f'| {label} | {item["frames_with_boxes_pct"]:.1f}% | '
                f'{item["boxes_per_frame"]:.2f} | '
                f'{item["disappearances_per_minute"]:.1f} | {note} |')

    lines = [
        '# Video demo: raw ve tracking', '',
        '| Mod | Tespitli kare % | Kutu/kare | Titreme (geçiş/dk) | Not |',
        '| --- | ---: | ---: | ---: | --- |',
        row('raw', raw, 'Bağımsız predict, conf=0.25'),
    ]
    if track:
        lines.append(row('track+filter', base_track,
                         f'ByteTrack, min_hits={min_hits}, hold={hold}, EMA α={alpha:g}'))
        lines.append(row('track+filter+dedup', track,
                         f'Küçük kutuya göre örtüşme ≥{overlap_thresh:g}'))
    lines.extend([
        '', f'Kaynak: `{source}` · ağırlık: `{weights}`.',
        f'Video: {width}×{height}, {fps:g} FPS, {raw["frames"]} kare '
        f'({raw["frames"] / fps / 60:.2f} dk). Model giriş boyutu: {IMAGE_SIZE}.',
        '', 'Titreme, çizilmiş bir kutunun bir sonraki karede kaybolmasıdır. Raw kutular '
        f'ardışık karelerde greedy IoU≥{MATCH_IOU:g} ile eşleştirildi; track kutuları ID ile izlendi. '
        'Dedup nedeniyle görünmez olan ID, yerinde bastıran kutu duruyorsa **devir** sayıldı; '
        'titremeye eklenmedi. Son kareden sonraki kaybolma sayılmadı. '
        'Bunlar **etiketsiz görüntüleme ölçüleri**; '
        'precision/recall veya gerçek yanlış alarm sayısı değildir.',
    ])
    if track:
        lines.extend([
            '', f'Raw tahmin eşiği {RAW_CONF:g}; ByteTrack giriş eşiği {TRACK_CONF:g} '
            '(düşük skorlu eşleştirme havuzu için). Tracker ayarları '
            '`trackers/pothole_bytetrack.yaml` dosyasındadır. Hold sırasında son EMA kutusu '
            'turuncu çizilir.',
            '', '## Track bilgisi', '',
            f'- Toplam track: **{track["tracks_total"]}**.',
            f'- Hiç çizilmeyen track: **{track["tracks_never_drawn"]}** '
            f'(`min_hits` altı: **{track["tracks_below_min_hits"]}**; '
            f'yeterli hit alıp hep bastırılan: **{track["tracks_confirmed_never_drawn"]}**). '
            'Etiket olmadığı için bunların yanlış alarm olduğu doğrulanamaz.',
            f'- Onaylı track ömrü medyanı: **{track["approved_lifetime_median_frames"]} kare** '
            '(ilk çizilen ile son çizilen kare arası, hold dahil).',
            f'- Bastırılan kutu sayısı: held→aktif **{track["suppressed"]["held_active"]}**, '
            f'aktif→aktif **{track["suppressed"]["active_active"]}**, '
            f'held→held **{track["suppressed"]["held_held"]}** '
            f'(toplam **{track["suppressed_total"]}** kare-kutu örneği).',
            f'- Devir: **{track["handoffs"]}**, yani '
            f'**{track["handoffs_per_minute"]:.1f}/dk**.',
            '', '## `min_hits` taraması', '',
            f'Örtüşme eşiği {overlap_thresh:g}; her N için aynı bastırma ve devir kuralları.', '',
            '| N | Tespitli kare % | Kutu/kare | Titreme/dk | Devir/dk | Bastırılan kutu | Hiç çizilmeyen track | Ömür medyanı (kare) |',
            '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ])
        for n, item in sweep.items():
            life = item['approved_lifetime_median_frames']
            lines.append(f'| {n} | {item["frames_with_boxes_pct"]:.1f}% | '
                         f'{item["boxes_per_frame"]:.2f} | '
                         f'{item["disappearances_per_minute"]:.1f} | '
                         f'{item["handoffs_per_minute"]:.1f} | '
                         f'{item["suppressed_total"]} | '
                         f'{item["tracks_never_drawn"]} | '
                         f'{life if life is not None else "—"} |')
        low, high = sweep[1], sweep[8]
        lines.extend([
            '', f'N=1→8 aralığında titreme {low["disappearances_per_minute"]:.1f}→'
            f'{high["disappearances_per_minute"]:.1f} geçiş/dk düşerken hiç çizilmeyen '
            f'track sayısı {low["tracks_never_drawn"]}→{high["tracks_never_drawn"]} çıktı. '
            f'Tespitli kare oranı da %{low["frames_with_boxes_pct"]:.1f}→'
            f'%{high["frames_with_boxes_pct"]:.1f} indi; yüksek N, gerçek çukurların '
            'ilk görüldüğü kareleri saklayabilir. Etiketsiz videoda bu dengeyi '
            'görselleştirebiliriz, fakat titreme azalması tek başına doğruluk artışı değildir.',
        ])
        lines.extend(['', '## Örtüşme eşiği taraması', '',
                      f'`min_hits={min_hits}`, `hold={hold}`, `alpha={alpha:g}`; '
                      'yalnız metrik hesaplandı, ek video üretilmedi.', '',
                      '| Eşik | Kutu/kare | Titreme/dk | Bastırılan kutu | Devir/dk |',
                      '| ---: | ---: | ---: | ---: | ---: |'])
        for threshold, item in overlap_sweep.items():
            lines.append(f'| {threshold:g} | {item["boxes_per_frame"]:.2f} | '
                         f'{item["disappearances_per_minute"]:.1f} | '
                         f'{item["suppressed_total"]} | '
                         f'{item["handoffs_per_minute"]:.1f} |')
        frame_names = ', '.join(map(str, check_frames))
        lines.extend(['', f'Görsel kontrol: `dedup_check.jpg` içinde {frame_names}. '
                      'karelerin eski/yeni tracking çizimleri yan yanadır.'])
    lines.extend(['', 'Çıktılar: `per_frame.csv`; track modunda `track.mp4`, '
                  '`compare.mp4`, `dedup_check.jpg` ve beş saniye aralıklı '
                  '`frames/`; raw modunda `raw.mp4`.', ''])
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--mode', choices=('raw', 'track'), default='track')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--tracker', type=Path, default=ROOT / 'trackers/pothole_bytetrack.yaml')
    parser.add_argument('--min-hits', type=int, default=5)
    parser.add_argument('--hold', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--overlap-thresh', type=float, default=0.6,
                        help='Intersection / smaller box area threshold for drawing deduplication.')
    parser.add_argument('--max-frames', type=int, help='Optional short smoke run.')
    args = parser.parse_args()
    if args.min_hits < 1 or args.hold < 0 or not 0 < args.alpha <= 1:
        parser.error('Require min-hits≥1, hold≥0 and 0<alpha≤1.')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive.')
    if not 0 < args.overlap_thresh <= 1:
        parser.error('--overlap-thresh must be in (0, 1].')
    weights = args.weights.resolve()
    source = args.source.resolve()
    if not weights.is_file() or not source.is_file():
        parser.error('Weights and source must be existing files.')
    tracker_yaml = args.tracker.resolve()
    if args.mode == 'track' and not tracker_yaml.is_file():
        parser.error(f'Tracker config missing: {tracker_yaml}')

    output = (args.out if args.out else ROOT / 'reports' /
              f'video_demo_{args.mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.mode == 'track':
        (output / 'frames').mkdir()
    print(f'OUTPUT={output}', flush=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f'Could not read video: {source}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError('Invalid video FPS or frame dimensions.')

    raw_model = YOLO(str(weights))
    tracked_model = YOLO(str(weights)) if args.mode == 'track' else None
    base_filter = TrackFilter(args.min_hits, args.hold, args.alpha) if tracked_model else None
    filter_keys = ({(n, args.overlap_thresh) for n in (*SWEEP_HITS, args.min_hits)} |
                   {(args.min_hits, threshold) for threshold in SWEEP_OVERLAP}) if tracked_model else set()
    filters = {key: TrackFilter(key[0], args.hold, args.alpha, key[1])
               for key in sorted(filter_keys)}
    selected_key = (args.min_hits, args.overlap_thresh)
    check_frames = ((0,) if args.max_frames is not None and args.max_frames <= 1250
                    else (1250, 2750, 3375))
    check_images = {}
    raw_writer = video_writer(output / 'raw.mp4', fps, (width, height)) if not tracked_model else None
    track_writer = video_writer(output / 'track.mp4', fps, (width, height)) if tracked_model else None
    compare_writer = video_writer(output / 'compare.mp4', fps, (2 * width, height)) if tracked_model else None

    frame_count = raw_frames_with_boxes = raw_box_count = raw_lost = 0
    previous_raw: list[Detection] = []
    fields = ['frame', 'time_s', 'raw_boxes', 'raw_disappearances', 'track_observed',
              'track_base_drawn', 'track_base_disappearances',
              'track_drawn', 'track_held', 'track_disappearances', 'track_handoffs',
              'suppressed_held_active', 'suppressed_active_active', 'suppressed_held_held']
    try:
        with (output / 'per_frame.csv').open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            while True:
                if args.max_frames is not None and frame_count >= args.max_frames:
                    break
                okay, frame = capture.read()
                if not okay:
                    break
                raw_result = raw_model.predict(frame, conf=RAW_CONF, imgsz=IMAGE_SIZE,
                                               device=0, verbose=False)[0]
                raw = unpack_boxes(raw_result, tracked=False)
                lost = raw_disappearances(previous_raw, raw) if frame_count else 0
                previous_raw = raw
                raw_lost += lost
                raw_frames_with_boxes += bool(raw)
                raw_box_count += len(raw)
                raw_image = draw(frame, raw, 'raw', tracked=False)
                row = dict(frame=frame_count, time_s=f'{frame_count / fps:.3f}',
                           raw_boxes=len(raw), raw_disappearances=lost,
                           track_observed='', track_base_drawn='',
                           track_base_disappearances='', track_drawn='', track_held='',
                           track_disappearances='', track_handoffs='',
                           suppressed_held_active='', suppressed_active_active='',
                           suppressed_held_held='')
                if tracked_model:
                    tracked_result = tracked_model.track(
                        frame, persist=True, tracker=str(tracker_yaml),
                        conf=TRACK_CONF, imgsz=IMAGE_SIZE, device=0, verbose=False)[0]
                    observed = unpack_boxes(tracked_result, tracked=True)
                    base_visible, base_lost = base_filter.update(observed, frame_count)
                    filtered = {}
                    for key, flt in filters.items():
                        filtered[key] = flt.update(observed, frame_count)
                    visible, track_lost = filtered[selected_key]
                    selected_filter = filters[selected_key]
                    track_image = draw(frame, visible, 'track+filter+dedup', tracked=True)
                    comparison = cv2.hconcat([raw_image, track_image])
                    track_writer.write(track_image)
                    compare_writer.write(comparison)
                    if frame_count % max(1, round(fps * 5)) == 0:
                        cv2.imwrite(str(output / 'frames' / f'frame_{frame_count:06d}.jpg'), comparison)
                    if frame_count in check_frames:
                        before = draw(frame, base_visible, 'track+filter', tracked=True)
                        check_images[frame_count] = cv2.hconcat([before, track_image])
                    row.update(track_observed=len(observed),
                               track_base_drawn=len(base_visible),
                               track_base_disappearances=base_lost,
                               track_drawn=len(visible),
                               track_held=sum(det.held for det in visible),
                               track_disappearances=track_lost,
                               track_handoffs=selected_filter.last_handoffs,
                               suppressed_held_active=selected_filter.last_suppressed['held_active'],
                               suppressed_active_active=selected_filter.last_suppressed['active_active'],
                               suppressed_held_held=selected_filter.last_suppressed['held_held'])
                else:
                    raw_writer.write(raw_image)
                writer.writerow(row)
                frame_count += 1
                if frame_count % 250 == 0:
                    print(f'Processed {frame_count} frames', flush=True)
    finally:
        capture.release()
        for video in (raw_writer, track_writer, compare_writer):
            if video is not None:
                video.release()

    if frame_count == 0:
        raise RuntimeError('Video contained no readable frames.')
    if tracked_model and check_images:
        ordered_check_frames = tuple(sorted(check_images))
        cv2.imwrite(str(output / 'dedup_check.jpg'),
                    cv2.vconcat([check_images[frame] for frame in ordered_check_frames]))
    else:
        ordered_check_frames = ()
    minutes = frame_count / fps / 60
    raw_stats = dict(frames=frame_count,
                     frames_with_boxes_pct=100 * raw_frames_with_boxes / frame_count,
                     boxes_per_frame=raw_box_count / frame_count,
                     disappearances=raw_lost,
                     disappearances_per_minute=raw_lost / minutes)
    base_stats = base_filter.summary(fps) if base_filter else None
    track_stats = filters[selected_key].summary(fps) if filters else None
    sweep = {n: filters[(n, args.overlap_thresh)].summary(fps)
             for n in SWEEP_HITS} if filters else {}
    overlap_sweep = {threshold: filters[(args.min_hits, threshold)].summary(fps)
                     for threshold in SWEEP_OVERLAP} if filters else {}
    report = render_report(source=source, weights=weights, fps=fps, width=width,
                           height=height, raw=raw_stats, base_track=base_stats,
                           track=track_stats, sweep=sweep, overlap_sweep=overlap_sweep,
                           min_hits=args.min_hits, hold=args.hold, alpha=args.alpha,
                           overlap_thresh=args.overlap_thresh,
                           check_frames=ordered_check_frames)
    (output / 'summary.md').write_text(report, encoding='utf-8')
    print(f'SUMMARY={output / "summary.md"}', flush=True)


if __name__ == '__main__':
    main()
