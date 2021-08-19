Center Pivot Brazil
========================

This project contains application source codes for the identification of circles of center pivots systems in remote sensing images using the Circular Hough Transform (CHT) method, the Balanced Random Forest (BRF) classifier and Land Use and Land Cover (LULC) data. Essentially, the repository has the main program (DetectingCenterPivot.py) to detected circles based on images of SAVI amplitudes and extract spectral/geometrical statistical information from these areas. In addition, auxiliaries programs (ClassifyPivotsAllTiles.py and PlotPivotsClassifAllTiles.py) it is used to classify information extracted from candidate circles of pivots to identify and view truly pivots against official mapped pivots by ANA in Brazil.

Release Notes
-------------

- Support files with single band (Amplitude/Greenest) pixels from LANDSAT 8

Installation
------------

**Dependencies**

    Python 3.6.X, Numpy, OpenCV, GDAL, Geopandas and Shapely
    

Build Steps
-----------

#ToDo


Usage
-----  

#ToDo

Data Processing Requirements
----------------------------

This version of the application requires the input files to be in the GeoTIFF format, compressed or not with zip or gzip.


Disclaimer
----------

This software is preliminary or provisional and is subject to revision. It is being provided to meet the need for timely best science. The software has not received final approval from the National Institute for Space Research (INPE). No warranty, expressed or implied, is made by the INPE or the Brazil Government as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. The software is provided on the condition that neither the INPE nor the Brazil Government shall be held liable for any damages resulting from the authorized or unauthorized use of the software.


License
-------

MIT License

Copyright (c) 2021 Rodrigues et al.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Authors
-------

`Rodrigues et al., (2021) <marcos.rodrigues@inpe.br>`_
