import os.path
from turtle import down
from typing import Optional

import av
import numpy as np
import PIL.Image
import skimage.filters
import skimage.morphology
import skimage.transform
from morphocut import Pipeline
from morphocut.core import (
    Call,
    Node,
    RawOrVariable,
    Stream,
    Variable,
    closing_if_closable,
)
from morphocut.file import CreateParents, Glob
from morphocut.filters import (
    BinomialFilter,
    ExponentialSmoothingFilter,
    MedianFilter,
    QuantileFilter,
)
from morphocut.image import RescaleIntensity
from morphocut.stream import Progress, Slice
from morphocut.utils import stream_groupby
from morphocut.video import VideoReader, VideoWriter
from sklearn.preprocessing import StandardScaler
import matplotlib.cm
import matplotlib.colors
from skimage.filters import apply_hysteresis_threshold
import glob
import fire
from tracking import DrawTracks, TrackObjects, TrackWriter
import skimage.measure

from tqdm import tqdm


class VideoBarcodeWriter(Node):
    def __init__(self, filename: RawOrVariable[str], frame: Variable):
        super().__init__()

        self.filename = filename
        self.frame = frame

    def transform_stream(self, stream: Stream) -> Stream:
        with closing_if_closable(stream):
            for filename, substream in stream_groupby(stream, self.filename):
                columns = []
                for obj in substream:
                    raw_frame: np.ndarray = self.prepare_input(obj, "frame")

                    columns.append(
                        raw_frame.mean(axis=1).astype(raw_frame.dtype).reshape(-1, 1, 3)
                    )

                    yield obj
                barcode = PIL.Image.fromarray(np.hstack(columns)).resize((800, 200))
                barcode.save(filename)


class TotalMeanWriter(Node):
    def __init__(self, filename: RawOrVariable[str], frame: Variable):
        super().__init__()

        self.filename = filename
        self.frame = frame

    def transform_stream(self, stream: Stream) -> Stream:
        with closing_if_closable(stream):
            for filename, substream in stream_groupby(stream, self.filename):
                scaler = StandardScaler()
                orig_info = None
                for obj in substream:
                    raw_frame: np.ndarray = self.prepare_input(obj, "frame")

                    if orig_info is None:
                        orig_info = raw_frame.shape, raw_frame.dtype

                    # Collapse image dimensions
                    scaler.partial_fit(raw_frame.reshape((1, -1)))

                    yield obj

                assert orig_info is not None
                total_mean = scaler.mean_.reshape(orig_info[0]).astype(orig_info[1])
                PIL.Image.fromarray(total_mean).save(filename)


class SumOfMasksWriter(Node):
    def __init__(
        self, filename: RawOrVariable[str], mask: Variable, cmap="gray", norm=None
    ):
        super().__init__()

        self.filename = filename
        self.mask = mask

        self.cmap = matplotlib.cm.get_cmap(cmap)

        if norm is None:
            norm = matplotlib.colors.Normalize()

        self.norm = norm

    def transform_stream(self, stream: Stream) -> Stream:
        with closing_if_closable(stream):
            for filename, substream in stream_groupby(stream, self.filename):
                acc = None
                for obj in substream:
                    mask: np.ndarray = self.prepare_input(obj, "mask")

                    if acc is None:
                        acc = np.zeros(mask.shape, "uint64")

                    acc += mask

                    yield obj

                assert acc is not None

                acc_normalized = self.norm(acc)
                img = self.cmap(acc_normalized)

                PIL.Image.fromarray((img * 255).astype("uint8")).save(filename)


class FirstFrameWriter(Node):
    def __init__(self, filename: RawOrVariable[str], frame: Variable):
        super().__init__()

        self.filename = filename
        self.frame = frame

    def transform_stream(self, stream: Stream) -> Stream:
        with closing_if_closable(stream):
            for filename, substream in stream_groupby(stream, self.filename):
                for i, obj in enumerate(substream):
                    if i == 0:
                        raw_frame: np.ndarray = self.prepare_input(obj, "frame")
                        PIL.Image.fromarray(raw_frame).save(filename)

                    yield obj


def create_pipeline(
    video_root,
    video_fn,
    output_root,
    max_frames=None,
    store_background_video=False,
    store_corrected_video=False,
    store_mask_video=False,
    store_mask_sum=False,
    store_tracks_video=False,
    downscale_factor=4,
    min_size=None,
):
    """
    Args:
        video_root (str):
        video_fn (str): Relative to video_root
    """

    with Pipeline() as p:
        video_name = Call(lambda video_fn: os.path.splitext(video_fn)[0], video_fn)

        # Filename for estimated background
        bg_video_fn = Call(os.path.join, output_root, "background", video_fn)
        CreateParents(bg_video_fn)

        corrected_video_fn = Call(os.path.join, output_root, "corrected", video_fn)
        CreateParents(corrected_video_fn)

        mask_video_fn = Call(os.path.join, output_root, "mask", video_fn)
        CreateParents(mask_video_fn)

        sum_of_masks_fn = Call(
            lambda video_name: os.path.join(output_root, "mask", video_name) + ".png",
            video_name,
        )

        tracks_video_fn = Call(os.path.join, output_root, "track", video_fn)
        tracks_fn = Call(
            lambda video_name: os.path.join(output_root, "track", video_name) + ".csv",
            video_name,
        )
        CreateParents(tracks_video_fn)

        video_path = Call(os.path.join, video_root, video_fn)

        # Read video file
        # Skip first 10 seconds to allow adaptation of the camera
        input_frame = VideoReader(video_path, skip_frames=10 * 30)

        # Convert to numpy array and slice off first 8 lines (because they are black)
        input_frame = Call(
            lambda input_frame: input_frame.to_ndarray(format="gray8")[8:], input_frame
        )

        if max_frames is not None:
            Slice(max_frames)

        # Scale down to speed up processing
        frame_small = Call(
            lambda frame: skimage.transform.downscale_local_mean(
                frame, downscale_factor
            ),
            input_frame,
        )

        # Calculate Background
        # Window of ~10 sek (300 frames):
        # Update every 15 Frames (0.5s), calculate quantile over 21 values
        bg_small = QuantileFilter(frame_small, 0.25, size=21, update_every=15)
        # bg_small = ExponentialSmoothingFilter(bg_small, alpha=0.1)

        # Smoothe estimated background
        bg_small = Call(lambda img: skimage.filters.gaussian(img, 8), bg_small)

        if store_background_video:
            # Store estimated background
            VideoWriter(
                bg_video_fn,
                Call(lambda arr: arr.astype("uint8"), bg_small),
                rate=30,
                pix_fmt="yuv420p",
                bit_rate=2048000,
            )

        # corrected_small = RescaleIntensity(
        #     frame_small / (bg_small + 1), in_range=(0, 1.2), dtype="uint8"
        # )
        # corrected_small = Call(lambda frame_small, bg_small: np.abs(frame_small - bg_small), frame_small, bg_small)
        corrected_small = frame_small - bg_small
        # Call(lambda corrected_small: print(np.quantile(corrected_small, [0, 0.05,0.95, 1])), corrected_small)
        corrected_small = RescaleIntensity(
            corrected_small, in_range=(-16, 128), dtype="uint8"
        )

        if store_corrected_video:
            VideoWriter(
                corrected_video_fn,
                corrected_small,
                rate=30,
                pix_fmt="yuv420p",
                bit_rate=2048000,
            )

        # mask = corrected_small > 64

        mask = Call(
            lambda img: apply_hysteresis_threshold(img, 48, 64), corrected_small
        )

        footprint = skimage.morphology.disk(3)
        mask = Call(
            lambda mask: skimage.morphology.binary_closing(mask, footprint), mask
        )

        if min_size is not None:
            mask = Call(
                lambda mask: skimage.morphology.remove_small_objects(mask, min_size),
                mask,
            )

        if store_mask_sum:
            SumOfMasksWriter(sum_of_masks_fn, mask, "viridis")

        if store_mask_video:
            VideoWriter(
                mask_video_fn, mask, rate=30, pix_fmt="yuv420p", bit_rate=2048000
            )

        regions = Call(
            lambda mask: skimage.measure.regionprops(
                skimage.measure.label(mask, connectivity=2)
            ),
            mask,
        )

        tracks = TrackObjects(corrected_small, regions, min_iou=0.1)

        if store_tracks_video:
            canvas = DrawTracks(corrected_small, tracks)
            VideoWriter(
                tracks_video_fn, canvas, rate=30, pix_fmt="yuv420p", bit_rate=2048000
            )

        TrackWriter(tracks_fn, tracks, include_regionprops=True)

        Progress()

    return p


def main():
    video_root = "/media/mschroeder/Barbara-FC/"
    output_root = "/data1/mschroeder/22-SquidDetection/data/out"

    # video_fn = "FishCam MOVIES Leg1 - Camera 1/2019-12-14_02.17.22.mp4"

    for video_path in tqdm(
        glob.glob(os.path.join(video_root, "FishCam MOVIES Leg1 - Camera 1/*.mp4"))
    ):
        video_fn = os.path.relpath(video_path, video_root)

        p = create_pipeline(
            video_root, video_fn, output_root, downscale_factor=4, store_mask_sum=True
        )

        p.run()


def test():
    video_root = "/media/mschroeder/Barbara-FC/"
    output_root = "/data1/mschroeder/22-SquidDetection/data/out"
    # video_fn = "FishCam MOVIES Leg1 - Camera 1/2019-12-14_02.17.22.mp4"
    video_fn = "FishCam MOVIES Leg1 - Camera 1/2019-10-23_17.12.46.mp4"

    p = create_pipeline(
        video_root,
        video_fn,
        output_root,
        downscale_factor=4,
        min_size=16,  # Minimum size of objects (Mind factor!)
        store_mask_sum=True,
        store_mask_video=True,
        store_tracks_video=True,
        max_frames=1000,
    )

    p.run()


if __name__ == "__main__":
    fire.Fire()
