"""Download the small real-data subsets used in the RATFA experiments.

Sources (public mirrors on Hugging Face):
  * UA-DETRAC  : https://huggingface.co/datasets/abhineet123/ua_detrac   (CC-BY-4.0 mirror)
                 vehicle boxes + track IDs per frame (annotations.csv per sequence)
  * BDD100K val: https://huggingface.co/datasets/dgural/bdd100k          (BSD-3 licence)
                 vehicle boxes + weather / time-of-day labels

Only selected files are fetched from the remote zip archives using HTTP range
requests, so the full 10 GB UA-DETRAC archive is never downloaded.

Usage:  python experiments/download_datasets.py
Output: datasets/UA-DETRAC/{train,test}/<sequence>/  and  datasets/BDD100K/
"""
import csv
import io
import json
import os
import random
import sys
import threading
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "datasets")
HF = "https://huggingface.co/datasets"
DETRAC_TRAIN_ZIP = f"{HF}/abhineet123/ua_detrac/resolve/main/ua_detrac_training_set.zip"
DETRAC_TEST_ZIP = f"{HF}/abhineet123/ua_detrac/resolve/main/ua_detrac_test_set.zip"
BDD_META = f"{HF}/dgural/bdd100k/resolve/main/samples.json"
BDD_IMG = f"{HF}/dgural/bdd100k/resolve/main/"

# Subset definition (fixed for reproducibility)
DETRAC_TRAIN_SEQS = ["MVI_20011", "MVI_39761", "MVI_40131", "MVI_40191",
                     "MVI_40732", "MVI_40962", "MVI_63521", "MVI_20051"]
DETRAC_TRAIN_STRIDE = 15          # every 15th frame
DETRAC_TRAIN_MAX = 60             # frames per training sequence
DETRAC_TEST_SEQS = ["MVI_39031", "MVI_40701", "MVI_40852", "MVI_40903"]
DETRAC_TEST_FRAMES = 150          # consecutive frames per test sequence (6 s @ 25 fps)
BDD_PER_CLASS = 120
BDD_VEHICLES = {"car", "bus", "truck"}
SEED = 42


class _HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        self.size = int(head.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req, timeout=120).read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _open_zip(url):
    return zipfile.ZipFile(io.BufferedReader(_HttpFile(url), 1 << 20))


_local = threading.local()


def _thread_zip(url):
    zs = getattr(_local, "zips", None)
    if zs is None:
        zs = _local.zips = {}
    if url not in zs:
        zs[url] = _open_zip(url)
    return zs[url]


def _fetch_member(url, member, dest):
    if os.path.exists(dest):
        return
    data = _thread_zip(url).read(member)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)


def download_detrac():
    jobs = []
    for split, url, seqs in (("train", DETRAC_TRAIN_ZIP, DETRAC_TRAIN_SEQS),
                             ("test", DETRAC_TEST_ZIP, DETRAC_TEST_SEQS)):
        z = _open_zip(url)
        names = z.namelist()
        for seq in seqs:
            folder = next(n.split("/")[0] for n in names if n.split("/")[0].endswith(seq))
            frames = sorted(n for n in names if n.startswith(folder + "/") and n.endswith(".jpg"))
            if split == "train":
                frames = frames[::DETRAC_TRAIN_STRIDE][:DETRAC_TRAIN_MAX]
            else:
                frames = frames[:DETRAC_TEST_FRAMES]
            keep = {os.path.basename(f) for f in frames}
            seq_dir = os.path.join(OUT, "UA-DETRAC", split, seq)
            os.makedirs(seq_dir, exist_ok=True)
            # annotations: keep only rows of the selected frames
            rows = list(csv.DictReader(io.StringIO(z.read(f"{folder}/annotations.csv").decode())))
            with open(os.path.join(seq_dir, "annotations.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["filename", "width", "height", "xmin", "ymin", "xmax", "ymax", "target_id"])
                for r in rows:
                    if r["filename"] in keep:
                        w.writerow([r["filename"], r["width"], r["height"], r["xmin"], r["ymin"],
                                    r["xmax"], r["ymax"], r["target_id"]])
            for fr in frames:
                jobs.append((url, fr, os.path.join(seq_dir, os.path.basename(fr))))
            print(f"[UA-DETRAC] {split}/{seq}: {len(frames)} frames")
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(lambda j: _fetch_member(*j), jobs))


def _condition(sample):
    weather = sample["weather"]["label"]
    tod = sample["timeofday"]["label"]
    if weather == "foggy":
        return "fog"
    if weather == "rainy" and tod == "daytime":
        return "rain"
    if weather == "clear" and tod == "night":
        return "dark"
    if weather == "clear" and tod == "daytime":
        return "clear"
    return None


def download_bdd():
    bdd_dir = os.path.join(OUT, "BDD100K")
    img_dir = os.path.join(bdd_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    meta_path = os.path.join(bdd_dir, "_samples_full.json")
    if not os.path.exists(meta_path):
        urllib.request.urlretrieve(BDD_META, meta_path)
    samples = json.load(open(meta_path))["samples"]
    by_cond = {"clear": [], "rain": [], "fog": [], "dark": []}
    for s in samples:
        c = _condition(s)
        if c:
            by_cond[c].append(s)
    rng = random.Random(SEED)
    records = []
    for cond, items in by_cond.items():
        items = sorted(items, key=lambda s: s["filepath"])
        rng.shuffle(items)
        for s in items[:BDD_PER_CLASS]:
            w, h = s["metadata"]["width"], s["metadata"]["height"]
            boxes = []
            for d in s["detections"]["detections"]:
                if d["label"] in BDD_VEHICLES:
                    x, y, bw, bh = d["bounding_box"]
                    boxes.append([x * w, y * h, (x + bw) * w, (y + bh) * h])
            records.append({"file": os.path.basename(s["filepath"]), "condition": cond,
                            "weather": s["weather"]["label"], "timeofday": s["timeofday"]["label"],
                            "width": w, "height": h, "boxes": boxes})
    with open(os.path.join(bdd_dir, "bdd100k_ratfa_subset.json"), "w") as f:
        json.dump(records, f, indent=1)

    def get(rec):
        dest = os.path.join(img_dir, rec["file"])
        if not os.path.exists(dest):
            urllib.request.urlretrieve(BDD_IMG + "data/" + rec["file"], dest)

    with ThreadPoolExecutor(12) as ex:
        list(ex.map(get, records))
    os.remove(meta_path)  # 71 MB full metadata is not needed after sub-selection
    for c in by_cond:
        print(f"[BDD100K] {c}: {sum(r['condition'] == c for r in records)} images")


if __name__ == "__main__":
    which = sys.argv[1:] or ["detrac", "bdd"]
    if "detrac" in which:
        download_detrac()
    if "bdd" in which:
        download_bdd()
    print("done ->", OUT)
