# Ex1 Plane Projection

## What the program does

`Ex0.py` reads the supplied `MRHead_1.nrrd` volume and samples planes listed in a CSV file. It writes one MP4 frame for each plane. The coordinates are voxel indices `(x,y,z)` in this NRRD.

## Run

```sh
python -m pip install numpy opencv-python
python Ex0.py data/MRHead_1.nrrd data/planes.csv results/sample_output.mp4
```

The CSV has a header, then a point count (3 or 4) and the x,y,z coordinates. The program reads every row.

## How it works

For a 3-point row, the program calculates a fourth corner with `d = b + c - a`. It samples the full four-corner area, so a right-angle triangle produces a full rectangle. For a 4-point row, it samples between the two edges. Each sample is rounded to the nearest voxel; points outside the volume are black.

The MRI values are scaled to grayscale using the volume's minimum and maximum values. Small planes are enlarged in the video so they are visible.

## Files

- `Ex0.py` — Python program.
- `SRS.md` — short requirements.
- `data/MRHead_1.nrrd` — the NRRD supplied with the exercise.
- `data/planes.csv` — 280 rows: 220 with 4 points and 60 with 3 points. The video has one frame per row at 1 fps (4 minutes 40 seconds). The CSV reaches z=219, while this scan has z indices 0–129; samples beyond the scan are black.
- `results/sample_output.mp4` — video made from that NRRD and CSV.
- `SUBMISSION.txt` — fill in your student ID and repository URL.

To try other planes, replace the CSV numbers with points inside this volume.




