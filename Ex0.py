"""Run: python Ex0.py file.nrrd file.csv file.mp4"""
import csv
import gzip
import math
import sys
from pathlib import Path

import cv2
import numpy as np

EPS = 0.001

def load_volume(path):
    """Read a simple single-file raw or gzip NRRD volume."""
    raw = Path(path).read_bytes()
    end = raw.find(b"\n\n"); separator = 2
    if end < 0:
        end = raw.find(b"\r\n\r\n"); separator = 4
    if end < 0:
        raise ValueError("NRRD header is incomplete")
    header, data = raw[:end].decode("ascii"), raw[end + separator:]
    fields = {}
    for line in header.splitlines()[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
    if fields.get("dimension") != "3":
        raise ValueError("NRRD must be 3D")
    sizes = [int(v) for v in fields["sizes"].split()]
    if len(sizes) != 3:
        raise ValueError("NRRD needs three sizes")
    types = {"uchar":"u1", "unsigned char":"u1", "ushort":"u2", "unsigned short":"u2",
             "uint":"u4", "unsigned int":"u4", "uint8":"u1", "uint16":"u2", "uint32":"u4", "short":"i2", "int":"i4"}
    if fields.get("type", "").lower() not in types:
        raise ValueError("use 8, 16, or 32 bit integer NRRD data")
    dtype = np.dtype(types[fields["type"].lower()])
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<" if fields.get("endian", "little") == "little" else ">")
    encoding = fields.get("encoding", "raw").lower()
    if encoding in ("gzip", "gz"):
        data = gzip.decompress(data)
    elif encoding != "raw":
        raise ValueError("NRRD encoding must be raw or gzip")
    values = np.frombuffer(data, dtype=dtype)
    if values.size != math.prod(sizes):
        raise ValueError("NRRD data size does not match its header")
    sx, sy, sz = sizes
    return values.reshape((sz, sy, sx)).transpose((2, 1, 0))

def distance(a, b):
    return float(np.linalg.norm(b - a))

def read_plane(row):
    """Read one CSV plane and return its two moving edges."""
    if len(row) == 13 and row[0].strip() in ("3", "4"):
        count = int(row[0])
        row = row[1:1 + count * 3]
    if len(row) not in (9, 12):
        raise ValueError("row must contain 3 or 4 points")
    try:
        points = np.array([float(v) for v in row]).reshape((-1, 3))
    except (ValueError, TypeError):
        raise ValueError("coordinates must be numbers")
    if not np.isfinite(points).all():
        raise ValueError("coordinates must be finite")

    if len(points) == 3:
        a, b, c = points
        length = distance(a, b)
        if length <= EPS or abs(length - distance(a, c)) > EPS:
            raise ValueError("triangle needs two equal sides from a")
        if np.linalg.norm(np.cross(b-a, c-a)) <= EPS * max(length, 1):
            raise ValueError("triangle points must not be on one line")
        start1, end1, start2, end2 = a, b, c, b + c - a
        width = distance(a, c)
    elif len(points) == 4:
        a1, a2, b, c = points
        length = distance(a1, b)
        if length <= EPS or abs(length - distance(a2, c)) > EPS:
            raise ValueError("trapezoid needs equal sides a1-b and a2-c")
        if distance(a1, a2) > distance(b, c) + EPS:
            raise ValueError("trapezoid needs dist(a1,a2) <= dist(b,c)")
        normal = np.cross(a2 - a1, b - a1)
        if np.linalg.norm(normal) <= EPS or abs(np.dot(normal, c-a1)) > EPS * np.linalg.norm(normal) * max(distance(a1, c), 1):
            raise ValueError("trapezoid points must be on one plane")
        start1, end1, start2, end2 = a1, b, a2, c
        width = distance(b, c)
    else:
        raise ValueError("give 3 triangle points or 4 trapezoid points")

    if width <= EPS:
        raise ValueError("plane is too small")
    return start1, end1, start2, end2, length, width

def project(volume, plane):
    start1, end1, start2, end2, length, width = plane
    image = np.zeros((math.ceil(length) + 1, math.ceil(width) + 1), dtype=np.uint8)
    max_value = np.iinfo(volume.dtype).max
    low, high = float(volume.min()), float(volume.max())
    for y in range(image.shape[0]):
        # Move both edges one step, as in the exercise pseudocode.
        t = min(y / length, 1.0)
        left = start1 + t * (end1 - start1)
        right = start2 + t * (end2 - start2)
        row_length = distance(left, right)
        steps = np.arange(math.floor(row_length) + 1)
        direction = (right - left) / row_length if row_length else np.zeros(3)
        points = np.rint(left + steps[:, None] * direction).astype(int)
        columns = np.rint(width / 2 - row_length / 2 + steps).astype(int)
        inside = ((points[:, 0] >= 0) & (points[:, 0] < volume.shape[0]) &
                  (points[:, 1] >= 0) & (points[:, 1] < volume.shape[1]) &
                  (points[:, 2] >= 0) & (points[:, 2] < volume.shape[2]) &
                  (columns >= 0) & (columns < image.shape[1]))
        values = volume[points[inside, 0], points[inside, 1], points[inside, 2]]
        if volume.dtype.kind == "i":
            # Scale the MRI values to visible grayscale.
            gray = np.zeros(values.shape) if high == low else np.clip((values.astype(float)-low) * 255 / (high-low), 0, 255)
        else:
            gray = values.astype(float) * 255 / max_value
        image[y, columns[inside]] = gray.astype(np.uint8)
    return image

def main():
    if len(sys.argv) != 4:
        print("Usage: python Ex0.py file.nrrd file.csv file.mp4", file=sys.stderr)
        return 2
    try:
        volume = load_volume(sys.argv[1])
        if volume.dtype.kind not in ("u", "i") or volume.dtype.itemsize not in (1, 2, 4):
            raise ValueError("volume must use 8, 16, or 32 bit integer values")
        with open(sys.argv[2], newline="", encoding="utf-8-sig") as f:
            rows = [(i, row) for i, row in enumerate(csv.reader(f), 1) if row and any(v.strip() for v in row)]
        if rows and rows[0][1][0].strip().lower().startswith("number of points"):
            rows = rows[1:]  # skip the CSV header
        if not rows:
            raise ValueError("CSV has no plane rows")

        planes = []
        width, height = 320, 240
        for line, row in rows:
            try:
                plane = read_plane(row)
                planes.append((line, plane, None))
                height = max(height, math.ceil(plane[4]) + 25)
                width = max(width, math.ceil(plane[5]) + 1)
            except ValueError as e:
                planes.append((line, None, str(e)))

        output = Path(sys.argv[3]); output.parent.mkdir(parents=True, exist_ok=True)
        video = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 1, (width, height))
        if not video.isOpened():
            raise ValueError("could not open MP4 encoder")
        try:
            for line, plane, error in planes:
                frame = np.zeros((height, width), dtype=np.uint8)
                if error:
                    label = f"ERROR on CSV line {line}: {error}"
                    print(label, file=sys.stderr)
                else:
                    image = project(volume, plane)
                    # Enlarge small planes so they are visible in the video.
                    scale = min((width - 8) / image.shape[1], (height - 32) / image.shape[0])
                    if scale != 1:
                        size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
                        image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
                    x = (width - image.shape[1]) // 2
                    y = (height - 24 - image.shape[0]) // 2
                    frame[y:y+image.shape[0], x:x+image.shape[1]] = image
                    label = f"CSV line {line}: OK"
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.putText(frame, label[:90], (4, height-6), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
                video.write(frame)
        finally:
            video.release()
        print(f"Wrote {len(planes)} frames to {output}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())










