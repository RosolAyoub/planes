# Software Requirements Specification

- Run: `python Ex0.py file.nrrd file.csv file.mp4`.
- Read the supplied 3D `MRHead_1.nrrd` volume and a CSV list of planes.
- Read 3 triangle points or 4 trapezoid points from each CSV row. For 3 points, calculate the fourth corner and sample the full four-corner area.
- Sample using the nearest voxel; points outside the volume are black.
- Write one 1 fps MP4 frame per CSV row. Invalid planes get an error frame and processing continues.
- CSV coordinates are voxel indices `(x,y,z)`.
- Scale MRI intensities from the volume minimum and maximum to gray.


