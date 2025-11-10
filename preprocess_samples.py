import argparse
import copy
import os
from collections import defaultdict
from functools import partial
from pathlib import Path

import pandas as pd
from pyannote.core import Annotation, Segment, Timeline
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from utils.feature_utils import get_shard_range

def load_one_uri(uri_turns: tuple[str, pd.DataFrame], keep_type: str = "SPEAKER") -> Annotation:
    uri, turns = uri_turns
    annotation = Annotation(uri=uri)
    for i, turn in turns.iterrows():
        segment = Segment(turn.start, turn.start + turn.duration)
        annotation[segment, i] = "SPEECH" #turn.speaker
    return annotation


def read_rttm(file_rttm: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        file_rttm,
        names=["start","duration","speaker","uri"],
        header=0,
        dtype={"uri": str, "start": int, "duration": int, "speaker": str},
        sep=",",
        keep_default_na=True,
    )


def post_process(annotation: Annotation, min_duration_on: float, min_duration_off: float) -> Annotation:
    active = copy.deepcopy(annotation)
    if min_duration_off > 0.0:
        active = active.support(collar=min_duration_off)
    if min_duration_on > 0:
        for segment, track in list(active.itertracks()):
            if segment.duration < min_duration_on:
                noise_padding = (min_duration_on - segment.duration)/2
                start = max(segment.start - noise_padding, 0)
                new_segment = Segment(start, start+min_duration_on)
                del active[segment, track]
                active[new_segment, track] = "SPEECH"
    return active


def gaps_filled(original: Annotation, new: Annotation) -> Annotation:
    return new.extrude(Timeline(original.itersegments()))


def segment_to_gaps(original: Annotation, new: Annotation) -> dict[Segment, Timeline]:
    mapping = defaultdict(list)
    for seg, gap in new.co_iter(gaps_filled(original, new)):
        mapping[seg[0]].append(gap[0])
    return {seg: Timeline(gaps) for seg, gaps in mapping.items()}


def cut_segment(segment: Segment, holes: Timeline, min_duration_on: float, max_duration_on: float) -> Timeline:
    timeline = Timeline([segment])
    if timeline.duration() <= min_duration_on:
        return Timeline([])
    if timeline.duration() <= max_duration_on:
        return timeline
    if not holes:
        left = cut_segment(Segment(segment.start, segment.middle), holes, min_duration_on, max_duration_on)
        right = cut_segment(Segment(segment.middle, segment.end), holes, min_duration_on, max_duration_on)
        return Timeline(left.segments_list_ + right.segments_list_)
    longest_hole = max(holes, key=lambda x: x.duration)
    timeline = timeline.extrude(longest_hole)
    new_timelines = [cut_segment(seg, holes.crop(seg), min_duration_on, max_duration_on) for seg in timeline]
    return Timeline([s for t in new_timelines for s in t.segments_list_])


def new_post_process(
    original: Annotation,
    min_duration_on: float,
    min_duration_off: float,
    max_duration_on: int,
) -> Annotation:
    new = post_process(original, min_duration_on, min_duration_off)
    mapping = segment_to_gaps(original, new)
    new_segments = []
    for seg in new.itersegments():
        if seg not in mapping:
            mapping[seg] = Timeline([])
        new_segments += list(cut_segment(seg, mapping[seg], min_duration_on, max_duration_on).segments_list_)
    annotation = Annotation(uri=original.uri)
    for i, segment in enumerate(new_segments):
        annotation[segment, i] = "SPEECH"
    return f"{annotation.to_rttm()}\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Post-process pyannote segments using min_duration_on and min_duration_off")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=str)
    parser.add_argument("--min-duration-on", type=float, default=2)
    parser.add_argument("--min-duration-off", type=float, default=2.0)
    parser.add_argument("--max-duration-on", type=float, default=30.0)#max no change
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    if 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        args.rank = int(os.environ['SLURM_ARRAY_TASK_ID'])


    data = read_rttm(args.input)
    data["start"] = data["start"]/1000
    data["duration"] = data["duration"]/1000
    by_uri = list(data.groupby("uri"))
    chunksize = 1 + (len(by_uri) // 4 * args.workers)
    print("data loaded")
    rttm = process_map(load_one_uri, by_uri, max_workers=args.workers, chunksize=chunksize, desc="Loading RTTM")
    processor = partial(
        new_post_process,
        min_duration_on=args.min_duration_on,
        min_duration_off=args.min_duration_off,
        max_duration_on=args.max_duration_on,
    )
    processed: list[str] = process_map(
        processor,
        rttm,
        max_workers=args.workers,
        chunksize=chunksize,
        desc="Post-processing",
    )

        
    with open(args.output, "w") as f:
        f.write("uri,start,duration")
        for annotation in tqdm(processed, desc="Saving RTTM"):
            annotation = [a.split(" ") for a in annotation.split("\n")[:-2]]
            annotation = [a[1] + "," + a[3] + "," + a[4] for a in annotation]
            str_annotation = "\n" + "\n".join(annotation) 
            f.write(str_annotation)
