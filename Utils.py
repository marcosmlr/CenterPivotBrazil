# coding: utf-8

# --------------------------
#        Imports
# --------------------------
from osgeo import ogr
import os
import json
import errno

"""Disclaimer
This software is preliminary or provisional and is subject to revision. It is being provided to meet the need for timely best science. The software
has not received final approval by the National Institute for Space Research (INPE). No warranty, expressed or implied, is made by the INPE or the
Brazil Government as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. The software
is provided on the condition that neither the INPE nor the Brazil Government shall be held liable for any damages resulting from the authorized or
unauthorized use of the software.

License
MIT License
Copyright (c) 2021 Rodrigues et al.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE."""


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def norm_minmax(array, min, max):
    """Function to normalization
    :param array: array values
    :param min: minimum value to normalize
    :param max: maximum value to normalize
    :return:
       array normalized between min/max values
    """

    """Feature scaling is used to bring all values into the range [0,1]. This is also called unity-based
       normalization. This can be generalized to restrict the range of values in the dataset between any arbitrary
       points a and b using:
       X ′ = a + ( X − Xmin ) ( b − a ) / (Xmax − Xmin)"""
    
    return (array - array.min()) / (array.max() - array.min()) * (max - min) + min


# https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html#count-geometries-in-a-geometry
def create_geoCollection(coords):
    # Create a geometry collection
    geomcol = ogr.Geometry(ogr.wkbGeometryCollection)

    for coord in coords:
        wkt = "POINT ("+str(coord[0])+" "+str(coord[1])+")"
        pt = ogr.CreateGeometryFromWkt(wkt)
        # Add a point
        geomcol.AddGeometry(pt)

        bufferDistance = coord[2]
        poly = pt.Buffer(bufferDistance)
        #print("%s buffered by %d is %s" % (pt.ExportToWkt(), bufferDistance, poly.ExportToWkt()))
        # Add a polygon
        geomcol.AddGeometry(poly)

    # print(geomcol.ExportToWkt())
    print("Geometry Collection has %i geometries" % (geomcol.GetGeometryCount()))

    return geomcol


def create_geoPolygon(coords):
    # Add the points to the ring
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for point in coords:
        lon = point[0]
        lat = point[1]
        ring.AddPoint(lon, lat)

    # Add first point again to ring to close polygon
    ring.AddPoint(coords[0][0], coords[0][1])

    # Add the ring to the polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    print(poly.ExportToWkt())

    return poly


"""
The ESRI Shapefile driver supports one geometry type per layer, and this cannot be a geometry collection type.
So it would look something like this:
# this will create a directory with 1 or more shapefiles
dst = driver.CreateDataSource("common-border")
# create common-border/points.shp
pointLayer = dst.CreateLayer("points", spatialReference, ogr.wkbPoint)
# create common-border/polygons.shp
polygonLayer = dst.CreateLayer("polygons", spatialReference, ogr.wkbPolygon)
# ... as needed
dst.GetLayerCount()  # 2 layers / shapefiles
And you would need to add each geometry type to the appropriate layer. There are 
probably some really smart things you can do with dictionaries to do the mapping, 
and if a key does not exist for the geometry type, it could create the required 
layer, etc.
"""


def write_shapefile(geoCollection, spatialRef, geomType, out_shp):
    """
    Adapted from:
    https://gis.stackexchange.com/a/52708/8104
    """
    # Now convert it to a shapefile with OGR
    driver = ogr.GetDriverByName('ESRI Shapefile')

    if os.path.exists(out_shp):
        driver.DeleteDataSource(out_shp)
    else:
        mkdir_p(os.path.split(out_shp)[0])

    ds = driver.CreateDataSource(out_shp)

    layer = ds.CreateLayer('',
                           geom_type=geomType,
                           srs=spatialRef)

    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    
    for id, geom in enumerate(geoCollection):
        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('id', id)

        # Set new geometry
        feat.SetGeometry(geom)

        # Add new feature to output Layer
        layer.CreateFeature(feat)

        feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None


# https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html
def write_geojson(geoCollection, spatialRef, geomType, out_json):

    # Now convert it to a GeoJSON with OGR
    driver = ogr.GetDriverByName('GeoJSON')

    add_metadata = False
    if os.path.exists(out_json):
        geo_objects = json.load(open(out_json))
        if 'properties' in geo_objects and 'bbox' in geo_objects:
            add_metadata = True
        driver.DeleteDataSource(out_json)
    else:
        mkdir_p(os.path.split(out_json)[0])

    ds = driver.CreateDataSource(out_json)

    layer = ds.CreateLayer('',
                           geom_type=geomType,
                           srs=spatialRef)

    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    # If there are multiple geometries, put the "for" loop here
    for id, geom in enumerate(geoCollection):
        if geom.GetGeometryType() == ogr.wkbPolygon:
            # Create a new feature (attribute and geometry)
            feat = ogr.Feature(defn)
            feat.SetField('id', id)
            feat.SetGeometry(geom)

            layer.CreateFeature(feat)
            feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None

    #Append metadata info:
    if add_metadata:
        geo_objects_tmp = json.load(open(out_json))
        geo_objects_tmp['properties'] = geo_objects['properties']
        geo_objects_tmp['bbox'] = geo_objects['bbox']
        with open(out_json, 'w') as outfile:
            json.dump(geo_objects_tmp, outfile)

    return
