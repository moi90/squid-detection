from abc import ABC, abstractmethod
import csv
import enum
from typing import List, Optional, Tuple
import uuid
from morphocut.core import (
    Node,
    Output,
    RawOrVariable,
    ReturnOutputs,
    Stream,
    Variable,
    closing_if_closable,
)
from morphocut.utils import stream_groupby
import skimage.measure._regionprops
import numpy as np
import cv2
import scipy.optimize
import shapely.geometry
import skimage.draw
import skimage.color
import skimage.color.rgb_colors
import skimage.util

BBox = Tuple[int, int, int, int]


def iou(bbox1: BBox, bbox2: BBox):
    # min_row, min_col, max_row, max_col

    box1 = shapely.geometry.box(*bbox1)
    box2 = shapely.geometry.box(*bbox2)

    return box1.intersection(box2).area / box1.union(box2).area


class Track(ABC):
    def __init__(
        self,
        frame: np.ndarray,
        region: skimage.measure._regionprops.RegionProperties,
        id: Optional[str] = None,
    ) -> None:
        if id is None:
            id = uuid.uuid4().hex

        self.region = region
        self.id = id

        self.age = 0
        self.staleness = 0

        self.bbox = self.region.bbox

        self._init_tracker(frame, self.bbox)

    def is_stale(self, max_staleness):
        return self.staleness > max_staleness

    def update_frame(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        self.age += 1

        self.bbox = self._update_tracker(frame)

        return self.bbox

    def set_region(
        self,
        frame: np.ndarray,
        region: Optional[skimage.measure._regionprops.RegionProperties],
    ):
        if region is None:
            self.staleness += 1
            return

        self._init_tracker(frame, region.bbox)
        self.bbox = region.bbox
        self.staleness = 0

    def set_stale(self):
        self.staleness += 1

    def to_dict(self):
        y0, x0, y1, x1 = self.bbox
        w, h = x1 - x0, y1 - y0
        return {
            "id": self.id,
            "x": x0,
            "y": y0,
            "w": w,
            "h": h,
            "bbox_area": w * h,
            "staleness": self.staleness,
            "age": self.age,
        }

    @abstractmethod
    def _init_tracker(self, frame: np.ndarray, bbox: BBox):
        pass

    @abstractmethod
    def _update_tracker(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        pass


class CSRTTrack(Track):
    def _init_tracker(self, frame: np.ndarray, bbox: BBox):
        self.tracker = cv2.TrackerCSRT_create()
        y0, x0, y1, x1 = bbox
        self.tracker.init(frame, (x0, y0, x1 - x0, y1 - y0))

    def _update_tracker(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        ok, (x, y, w, h) = self.tracker.update(frame)

        return (y, x, y + h, x + w)


class TrackManager:
    def __init__(self, max_staleness=15, min_iou=0.25) -> None:
        self.max_staleness = max_staleness
        self.min_iou = min_iou

        self.tracks: List[Track] = []

    def update(
        self,
        frame: np.ndarray,
        regions: List[skimage.measure._regionprops.RegionProperties],
    ) -> List[Track]:
        # Advance tracks using frame
        new_locations = [t.update_frame(frame) for t in self.tracks]

        # Match supplied regions to known tracks
        iou_matrix = self._calc_iou_matrix(new_locations, regions)
        ii, jj = scipy.optimize.linear_sum_assignment(iou_matrix, maximize=True)

        # Append regions to matched tracks
        matched_tracks = set()
        matched_regions = set()
        for i, j in zip(ii, jj):
            if iou_matrix[i, j] < self.min_iou:
                continue

            self.tracks[i].set_region(frame, regions[j])
            matched_tracks.add(i)
            matched_regions.add(j)

        # Increase staleness for unmatched tracks
        for i, t in enumerate(self.tracks):
            if i in matched_tracks:
                continue

            t.set_stale()
            # print(f"Lost track {t.id}")

        # Create new tracks for unmatched regions
        for j, r in enumerate(regions):
            if j in matched_regions:
                continue

            self.tracks.append(CSRTTrack(frame, r))
            # print(f"Created track for {r.bbox}")

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if not t.is_stale(self.max_staleness)]

        return self.tracks

    def _calc_iou_matrix(
        self,
        new_locations: List[Tuple[int, int, int, int]],
        regions: List[skimage.measure._regionprops.RegionProperties],
    ):
        iou_matrix = np.zeros((len(self.tracks), len(regions)))
        for i, t_bbox in enumerate(new_locations):
            for j, r in enumerate(regions):
                iou_matrix[i, j] = iou(t_bbox, r.bbox)
        return iou_matrix


@ReturnOutputs
@Output("tracks")
class TrackObjects(Node):
    def __init__(
        self,
        frame: RawOrVariable,
        regions: RawOrVariable,
        max_staleness=15,
        min_iou=0.25,
    ):
        super().__init__()

        self.frame = frame
        self.regions = regions
        self.max_staleness = max_staleness
        self.min_iou = min_iou

    def transform_stream(self, stream: Stream) -> Stream:
        with closing_if_closable(stream):
            track_manager = TrackManager(
                max_staleness=self.max_staleness, min_iou=self.min_iou
            )
            for obj in stream:
                frame, regions = self.prepare_input(obj, ("frame", "regions"))

                # Update trackers
                tracks = track_manager.update(frame, regions)

                yield self.prepare_output(obj, tracks)


@ReturnOutputs
@Output("frame")
class DrawTracks(Node):
    def __init__(self, frame: RawOrVariable, tracks: RawOrVariable):
        super().__init__()

        self.frame = frame
        self.tracks = tracks

        self._label_mapping = {}

    def transform(self, frame: np.ndarray, tracks: List[Track]):
        if frame.ndim == 2:
            canvas = skimage.color.gray2rgb(frame)
        else:
            canvas = frame.copy()

        colors = np.array(
            [
                skimage.color.rgb_colors.__dict__[n]
                for n in skimage.color.colorlabel.DEFAULT_COLORS
            ]
        )

        if canvas.dtype == np.dtype("uint8"):
            colors = skimage.util.img_as_ubyte(colors)

        for t in tracks:
            id = self._label_mapping.setdefault(t.id, len(self._label_mapping))

            color = colors[id % len(colors)]

            # TODO: Store short history of bboxes to display trace
            r0, c0, r1, c1 = t.bbox
            coords = skimage.draw.rectangle_perimeter(
                (r0, c0), (r1, c1), shape=canvas.shape
            )
            skimage.draw.set_color(canvas, coords, color)

        return canvas


class TrackWriter(Node):
    # area	area_bbox	area_convex	area_filled	axis_major_length	axis_minor_length eccentricity	equivalent_diameter_area	euler_number	extent	feret_diameter_max
    # orientation	perimeter	perimeter_crofton solidity

    def __init__(
        self,
        filename: RawOrVariable[str],
        tracks: RawOrVariable,
        *,
        meta: RawOrVariable = None,
        include_regionprops=False,
    ):
        super().__init__()

        self.filename = filename
        self.tracks = tracks
        self.meta = meta

        self.include_regionprops = include_regionprops

    def transform_stream(self, stream: Stream) -> Stream:
        with closing_if_closable(stream):
            for filename, substream in stream_groupby(stream, self.filename):
                with open(filename, "w", newline="") as f:
                    writer = None
                    for obj in substream:
                        tracks, meta = self.prepare_input(obj, ("tracks", "meta"))
                        tracks: List[Track]

                        for t in tracks:
                            data = {}
                            if self.include_regionprops:
                                data.update((k, t.region[k]) for k in t.region)

                            if meta is not None:
                                data.update(meta)

                            data.update(t.to_dict())

                            if writer is None:
                                writer = csv.DictWriter(f, data.keys())
                                writer.writeheader()

                            writer.writerow(data)

                        yield obj
