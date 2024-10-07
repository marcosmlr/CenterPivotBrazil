# coding: utf-8

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

#--------------------------
#        Imports
#--------------------------
from osgeo import osr, gdal, ogr, gdal_array
from gdalconst import *             # importar constantes
gdal.UseExceptions()                # informar o uso de exceções
from zipfile import ZipFile, is_zipfile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap  # Linear interpolation for color maps
import numpy as np
import cv2
from timeit import default_timer as timer
import glob
import Utils
import subprocess, os, sys
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean
from scipy.stats import mstats #Statistical functions that can be used with masked arrays.
from multiprocessing.pool import ThreadPool as Pool #https://stackoverflow.com/a/15144765
from multiprocessing import cpu_count, Lock  #Updating a File from Multiple Threads in Python - https://www.novixys.com/blog/updating-file-multiple-threads-python/#2_Read_and_Write_a_File_Without_Locking




# Fonte de apoio:
# https://github.com/gqueiroz/ser347/blob/master/2018/aula-18/gdal_1.ipynb

# Intersting:
# - https://medium.com/planet-stories/a-gentle-introduction-to-gdal-part-1-a3253eb96082
# - https://publicwiki.deltares.nl/display/OET/Python+convert+coordinates
# Python multiprocessing: understanding logic behind `chunksize` - https://stackoverflow.com/a/54032744

#--------------------------
#        Functions
#--------------------------
def get_band_array(filename, band_num=1, info=False, scale=None):

    global dataset
    if is_zipfile(filename):
        with ZipFile(filename) as theZip:
            fileNames = theZip.namelist()
            for fileName in fileNames:
                if fileName.endswith('.tif'):
                    print("Tentar abrir " + fileName)

                    try:
                        start = timer()
                        dataset = gdal.Open('/vsizip/%s/%s' % (filename, fileName))
                        end = timer()
                        print("File unziped in: ", end - start, " seconds.")

                    except:
                        del (dataset)
                        print("Erro na abertura do arquivo!")
    else:
        print("Tentar abrir " + filename)

        try:
            dataset = gdal.Open(filename)
        except:
            del(dataset)
            print("Erro na abertura do arquivo!")

    geotransform = dataset.GetGeoTransform()
    print(geotransform)

    lrx = geotransform[0] + (dataset.RasterXSize * geotransform[1])
    lry = geotransform[3] + (dataset.RasterYSize * geotransform[5])

    corners = [geotransform[0], lry, lrx, geotransform[3]]
    print("The extent should be inside: " + str(corners))


    center_pivot = 500.  # tipycally radii meters
    global min_dist_px
    global min_radii_px
    global max_radii_px

    
    # Setup the source projection
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(dataset.GetProjectionRef())
   
    center_lat = np.mean([lry, geotransform[3]])
    
    if spatial_reference.GetAttrValue('unit') == 'degree': #https://gis.stackexchange.com/a/60372
        #https://stackoverflow.com/a/23875713
        #Latitude:  1 deg = 110.54 km = 110540 m
        #Longitude: 1 deg = 111.320*cos(latitude) km = 111320 * cos(latitude) m
        geotransform = list(geotransform)

        geotransform[5] =  geotransform[5] * 110540.0
        geotransform[1] = geotransform[1] * (111320.0 * np.cos(np.deg2rad(center_lat)))
        
    min_dist_px = (center_pivot) / geotransform[1] #min distance between centroid of neighboring circles 500m
    min_radii_px = (center_pivot - 300.) / geotransform[1] #pivots min with 200m of radii
    max_radii_px = (center_pivot + 300.) / geotransform[1]
    

    global res_x
    global res_y
    res_x = geotransform[1]
    res_y = -geotransform[5]

    if info:
        # [0] is the x coordinate of the upper left cell in raster
        # [1] is the width of the elements in the raster
        # [2] is the element rotation in x, is set to 0 if a north up raster
        # [3] is the y coordinate of the upper left cell in raster
        # [4] is the element rotation in y, is set to 0 if a north up raster
        # [5] is the height of the elements in the raster (negative)

        latitude = geotransform[3]
        longitude = geotransform[0]
        resolucao_x = geotransform[1]
        resolucao_y = -geotransform[5]

        print("Latitude inicial do dataset:", latitude)
        print("Longitude inicial do dataset:", longitude)
        print("Resolução (x) do dataset:", resolucao_x)
        print("Resolução (y) do dataset:", resolucao_y)
        print("Numero de linhas:", dataset.RasterYSize)
        print("Numero de colunas:", dataset.RasterXSize)

        # quantidade de bandas
        bandas = dataset.RasterCount

        print("Número de bandas:", bandas)

    # no caso da imagem RapidEye, as bandas 5
    # e 3 correspondem às bandas NIR e RED
    banda = dataset.GetRasterBand(band_num)

    # verificar algumas propriedades das bandas, como o histograma
    # plt.plot(banda.GetHistogram())

    if info:
        print("Tipos de dados:")
        print(" - banda :", gdal.GetDataTypeName(banda.DataType))

        (menor_valor, maior_valor) = banda.ComputeRasterMinMax()
        print("Menor valor:", menor_valor)
        print("Maior valor:", maior_valor)
        print("NO DATA VALUE:", banda.GetNoDataValue())

    # obtencao dos arrays numpy das bandas
    array = banda.ReadAsArray()
    print(array.min(), array.max())

    array = np.ma.masked_array(array, np.isnan(array))   

    marray = array
    
    # Exclude the pixels with no data value and normalize data
    NoData = banda.GetNoDataValue()
    if NoData == None or ('CBERS_4_MUX' in filename and 'L5' in filename):
        NoData = 0.

    print(gdal.GetDataTypeName(banda.DataType))
    # marray = np.ma.masked_array(array, mask=np.isin(array, NoData))
    # scale = 255. if array.max() < 256 else 10000.

    if scale == None:
        # Water bodies Byte(tif) values range between 0 and 1:
        if gdal.GetDataTypeName(banda.DataType) == 'Byte' and array.max() > 1:
            scale = 255.
        elif gdal.GetDataTypeName(banda.DataType) == 'Int16':
            scale = 10000.
        elif gdal.GetDataTypeName(banda.DataType) == 'UInt16' and 'LC08' in filename:
            scale = 65535. #Landsat8
        elif gdal.GetDataTypeName(banda.DataType) == 'UInt16' and 'CBERS_4_MUX' in filename:
            scale = 10000. #CBERS Coupled Moderate Products for Atmospheric Correction (CMPAC) Level 5 product
        elif gdal.GetDataTypeName(banda.DataType) == 'Float64' or \
             (gdal.GetDataTypeName(banda.DataType) == 'Byte' and array.max() <= 1) or \
             gdal.GetDataTypeName(banda.DataType) == 'Float32': #Water bodies Byte(tif) values range between 0 and 1
                                                                #NDVI and SAVI Greenest pixel Float32(tif) from Earth Engine
            scale = 1.  # Nothing to do
        else:
            raise TypeError("band type:",gdal.GetDataTypeName(banda.DataType),"not allowed!")


    marray = np.ma.masked_where(array == NoData, array / scale)
    print(marray.min(), marray.max())
    if scale != 1.:
        np.clip(marray, 0., 1., out=marray)
        print(marray.min(), marray.max())

    return marray

    # fechar o dataset e liberar memória
    dataset = None




def salvar_banda(matriz_de_pixels, nome_do_arquivo, dataset_de_referencia, NoData=None):
    # obter metadados
    linhas = dataset_de_referencia.RasterYSize
    colunas = dataset_de_referencia.RasterXSize
    bandas = 1
    # definir driver
    driver = gdal.GetDriverByName('GTiff')
    # copiar tipo de dados da banda já existente
    #data_type = dataset_de_referencia.GetRasterBand(1).DataType
    data_type = gdal_array.NumericTypeCodeToGDALTypeCode(matriz_de_pixels.dtype)
    # criar novo dataset
    dataset_output = driver.Create(nome_do_arquivo, colunas, linhas, bandas, data_type)
    # copiar informações espaciais da banda já existente
    dataset_output.SetGeoTransform(dataset_de_referencia.GetGeoTransform())
    # copiar informações de projeção
    dataset_output.SetProjection(dataset_de_referencia.GetProjectionRef())
    # escrever dados da matriz NumPy na banda
    dataset_output.GetRasterBand(1).WriteArray(matriz_de_pixels)
    # define no data value if required
    if NoData != None:
        dataset_output.GetRasterBand(1).SetNoDataValue(NoData)
        dataset_output.GetRasterBand(1).WriteArray(np.ma.filled(matriz_de_pixels,fill_value=NoData))
    # salvar valores
    dataset_output.FlushCache()
    # fechar dataset
    dataset_output = None


def export_circles_detected(circles, filename_out, type='shp', targetEPSG=4326):
    if circles is not None:
        if circles.dtype == 'float32':
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

        geotransform = dataset.GetGeoTransform()

        circles_features = []
        # loop over the (x, y) coordinates and radius of the circles
        for x, y, r in circles:
            lrx = geotransform[0] + (x * geotransform[1])
            lry = geotransform[3] + (y * geotransform[5])

            #print('Index (x,y):',x,y)
            #print('Coordinators(lon,lat):',lrx,lry)
            circles_features.append([lrx,lry,r * geotransform[1]])

        geoCollection = Utils.create_geoCollection(circles_features)

        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromWkt(dataset.GetProjectionRef())
        target = spatial_reference

        """
        The coordinate reference system for all GeoJSON coordinates is a geographic coordinate reference system,
        using the World Geodetic System 1984 (WGS 84) [WGS84] datum, with longitude and latitude units of decimal
        degrees.
        ...
        Note: the use of alternative coordinate reference systems was specified in [GJ2008], but it has been
        removed from this version of the specification because the use of different coordinate reference systems
        - - especially in the manner specified in [GJ2008] - - has proven to have interoperability issues.In general,
        GeoJSON processing software is not expected to have access to coordinate reference system databases or to have
        network access to coordinate reference system transformation parameters.However, where all involved parties 
        have a prior arrangement, alternative coordinate reference systems can be used without risk of data being 
        misinterpreted. Source: RFC 7946 Butler, et al. (2016) https://tools.ietf.org/html/rfc7946#page-3"""

        #spatial reference Authority for WGS84 is 4326:
        if type == 'geojson' and spatial_reference.GetAttrValue("AUTHORITY", 1) != 4326:
            target = osr.SpatialReference()
            target.ImportFromEPSG(4326)
            if float('.'.join(gdal.__version__.split('.')[0:2])) >= 3.0:
              target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) #About changes in GDAL since 3.0 https://gdal.org/tutorials/osr_api_tut.html - CRS and axis order

            transform = osr.CoordinateTransformation(spatial_reference, target)
            geoCollection.Transform(transform)
        elif spatial_reference.GetAttrValue("AUTHORITY", 1) != targetEPSG:
            print('Origem:\n', spatial_reference)

            target = osr.SpatialReference()
            target.ImportFromEPSG(targetEPSG)
            if float('.'.join(gdal.__version__.split('.')[0:2])) >= 3.0:
              target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) #About changes in GDAL since 3.0 https://gdal.org/tutorials/osr_api_tut.html - CRS and axis order

            print('Destino:\n', target)

            transform = osr.CoordinateTransformation(spatial_reference, target)
            geoCollection.Transform(transform)


        if type == 'shp':
            destino = './shp'
            Utils.mkdir_p(destino)

            Utils.write_shapefile(geoCollection,
                                  target,
                                  ogr.wkbPolygon,
                                  os.path.join(destino,filename_out+'_polygon.shp'))
        elif type == 'geojson':
            destino = './geojson'
            Utils.mkdir_p(destino)

            Utils.write_geojson(geoCollection,
                                target,
                                ogr.wkbGeometryCollection,
                                os.path.join(destino, filename_out+'.geojson'))
    else:
        print('None circles detected!')


def export_raster_feature_extent(filename_out, array_data, type='shp', NoValue=None, targetEPSG=4326):

    geotransform = dataset.GetGeoTransform()
    raster_feature = []

    #Minimum box of valid values or extent of raster
    if NoValue is not None:
        image = dataset.GetRasterBand(1).ReadAsArray()

        points = np.zeros((4, 2), dtype=int)

        edge0 = np.ma.notmasked_edges(array_data, axis=0)
        
        # bottow left(y,x):
        points[3,:] = [edge0[-1][0][0]+1, edge0[-1][1][0]]
        
        # upper right(y,x):
        points[1, :] = [edge0[0][0][-1], edge0[0][1][-1]+1]
        
        edge1 = np.ma.notmasked_edges(array_data, axis=1)
        # upper left(y,x):
        points[0, :] = [edge1[0][0][0], edge1[0][1][0]]

        # bottow right(y,x):
        points[2, :] = [edge1[1][0][-1]+1, edge1[1][1][-1]+1]

        for y,x in points:
            lrx = geotransform[0] + (x * geotransform[1])
            lry = geotransform[3] + (y * geotransform[5])
            raster_feature.append([lrx, lry])

    else:
        raster_feature.append([geotransform[0],geotransform[3]]) #upper left cell in raster

        lrx = geotransform[0] + (dataset.RasterXSize * geotransform[1])
        lry = geotransform[3] + (0 * geotransform[5])
        raster_feature.append([lrx, lry])  # upper right cell in raster

        lrx = geotransform[0] + (dataset.RasterXSize * geotransform[1])
        lry = geotransform[3] + (dataset.RasterYSize * geotransform[5])
        raster_feature.append([lrx, lry])  # lower right cell in raster

        lrx = geotransform[0] + (0 * geotransform[1])
        lry = geotransform[3] + (dataset.RasterYSize * geotransform[5])
        raster_feature.append([lrx, lry])  # lower left cell in raster


    geoPolygon = Utils.create_geoPolygon(raster_feature)

    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(dataset.GetProjectionRef())
    target = spatial_reference

    if spatial_reference.GetAttrValue("AUTHORITY", 1) != targetEPSG:
        print('Origem:\n', spatial_reference)

        target = osr.SpatialReference()
        target.ImportFromEPSG(targetEPSG)
        print('Destino:\n', target)

        transform = osr.CoordinateTransformation(spatial_reference, target)
        geoPolygon.Transform(transform)


    destino = './shp'
    Utils.mkdir_p(destino)

    Utils.write_shapefile(geoPolygon,
                          target,
                          ogr.wkbPolygon,
                          os.path.join(destino,filename_out+'_polygon.shp'))



"""Feature scaling is used to bring all values into the range [0,1]. This is also called unity-based
normalization. This can be generalized to restrict the range of values in the dataset between any arbitrary
points a and b using:

X ′ = a + ( X − Xmin ) ( b − a ) / (Xmax − Xmin)"""


def norm_minmax(array, min, max):
    return (array - array.min()) / (array.max() - array.min()) * (max - min) + min


# Adapted from https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.ma.median(image)
    # v = np.average(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    start = timer()
    edged = cv2.Canny(image, lower, upper)
    end = timer()
    print("Edged in: ", end - start," seconds.")
    print('median:', v, 'lower:', lower, ' upper:', upper)

    # return the edged image
    return edged

def auto_sharr(image, sigma=0.33):
    #https://docs.opencv.org/3.3.1/d5/d0f/tutorial_py_gradients.html
    #https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html#sobel-and-scharr-derivatives
    """In our last example, output datatype is cv2.CV_8U or np.uint8. But there
     is a slight problem with that. Black-to-White transition is taken as Positive
     slope (it has a positive value) while White-to-Black transition is taken as
     a Negative slope (It has negative value). So when you convert data to np.uint8,
     all negative slopes are made zero. In simple words, you miss that edge.

     If you want to detect both edges, better option is to keep the output datatype
     to some higher forms, like cv2.CV_16S, cv2.CV_64F etc, take its absolute value
     and then convert back to cv2.CV_8U. """
    
    # compute the median of the single channel pixel intensities
    v = np.ma.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    blurred = cv2.GaussianBlur(image, (3, 3), 2)
                               
    #https://docs.opencv.org/3.3.1/d3/d63/edge_8cpp-example.html
    dx = cv2.Scharr(blurred, cv2.CV_16S, 1, 0)
    dy = cv2.Scharr(blurred, cv2.CV_16S, 0, 1)

    start = timer()
    edged = cv2.Canny(dx,dy, threshold1=lower, threshold2=upper, L2gradient=False)
    end = timer()
    print("Edged in: ", end - start," seconds.")
    print('median:', v, 'lower:', lower, ' upper:', upper)

    # return the edged image
    return edged


# Adapted from https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def finding_circles(auto, dp_value=1, min_dist_px=40,
                    param1=50,param2=15,
                    min_radii_px=15, max_radii_px=40):
    #######################################################
    # Hough Transform to detect Lines and Circles:
    ######################################
    """
    https://alyssaq.github.io/2014/understanding-hough-transform/
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

    Hough Transform to detect Circles:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
    https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    method = cv2.HOUGH_GRADIENT --> "A  COMPARATIVE  STUDY  OF  HOUGH  TRANSFORM METHODS  FOR  CIRCLE  FINDING"""

    """Center pivots are typically less than 1600 feet (500 meters) in length (circle radius)
    with the most common size being the standard 1/4 mile (400 m) machine. A typical 1/4 mile
    radius crop circle covers about 125 acres of land"""

    print('Parameters for HoughCircles: dp:', dp_value,'minDist:', min_dist_px,
          'param1:',param1,'param2:',param2,'minRadius:', min_radii_px,
          'maxRadius:',max_radii_px)

    # detect circles in the image
    circles = cv2.HoughCircles(auto, cv2.HOUGH_GRADIENT,
                               dp=dp_value, minDist=min_dist_px,
                               param1=param1, param2=param2,
                               minRadius=int(min_radii_px), maxRadius=int(max_radii_px))


    # ensure at least some circles were found
    if circles is not None:
        # Remove circles with radii missing value:
        circles_tmp = np.delete(circles, np.where(circles[:, :, 2] == 0.), axis=1)

        if circles_tmp.size == 0:
            circles = None
        else:
            circles = circles_tmp
            print('Nr. Circles Detected:', circles.shape[1])

    return circles


def verif_minimum_circle(circle_coords, _array, contador, _array_savi, _array_ndvi, _array_amplitude, _array_lulc):
    """
    Function to create file with statistics information of circles detected with Hough Transform to help characterize circles of pivots or not
    
    :param circle_coords: tuple with (x,y,r) center and radii of circle
    :param _array: numpy array image edges
    :param contador: int number id of circle
    :param _array_savi: numpy array image SAVI values
    :param _array_ndvi: numpy array image NDVI values
    :param _array_amplitude: numpy array image Amplitude of NDVI or SAVI values
    :param _array_lulc: numpy array image Land Use Land Cover classification
    :return:
       None
    """

    a,b,radius = circle_coords
    ny, nx = _array.shape
    x, y = np.ogrid[-b:ny - b, -a:nx - a]
    mask = x*x + y*y <= (radius+2)*(radius+2) #Tolerância para recuperar bem as bordas do círculo gerada pelo Canny   

    teste = np.zeros_like(_array)
    teste[mask] = _array[mask]
    
    # Get the indices of maximum element in numpy array
    result = np.where(teste == np.amax(teste))
    result = (result[1], result[0]) #reverse order to collum/row
    points = np.column_stack(result) #points to convex - https://stackoverflow.com/a/40922937
    

    #https://lvngd.com/blog/convex-hull-graham-scan-algorithm-python
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
    hull = ConvexHull(points)

    """ Perimeter of ConvexHull - https://stackoverflow.com/a/52404024
        As an alternative, given that the data is 2D, you can use hull.area. That is
        the value returned in the above method is equal to the value of the area property.
        If you want the real area, you need to query hull.volume."""
    
    percent_points =  abs(100 * (len(hull.simplices) - len(points)) / len(points))
    mask_tmp = x*x + y*y <= (radius-5)*(radius-5) 
    pixels_within =  np.count_nonzero(_array[mask_tmp])
    mean_ndvi = np.ma.mean(_array_ndvi[mask])
    std_ndvi = np.ma.std(_array_ndvi[mask])
    mean_savi = np.ma.mean(_array_savi[mask])
    std_savi = np.ma.std(_array_savi[mask])
    mean_amplitude = np.ma.mean(_array_amplitude[mask])
    std_amplitude = np.ma.std(_array_amplitude[mask])
    mode_lulc = np.int(mstats.mode(_array_lulc[mask], axis=None)[0][0])
    
    lock.acquire()
    print(contador,',',a,',',b,',',radius,',',percent_points,',',pixels_within,',',mean_ndvi,',',std_ndvi,',',mean_savi,',',std_savi,
          ',',mean_amplitude,',',std_amplitude,',',mode_lulc,sep='',file=f)
    lock.release()


# Source: https://unix.stackexchange.com/a/590637
class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open("DetectPivots.log", "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)  

        def flush(self):
            #this flush method is needed for python 3 compatibility.
            #this handles the flush command by doing nothing.
            #you might want to specify some extra behavior here.
            pass    


def finding_circles_sliding_window(array):    
    winW = int(5000 / res_x)
    winH = int(5000 / res_y)
    step_size = int(4000 / res_x)
    
    edges_window = np.zeros_like(array)
                
    temp = []
    # loop over the sliding window for each array
    for (x, y, window) in sliding_window(array, stepSize=step_size, windowSize=(winW, winH)):
        
        if np.max(window) > 0.:
            
            # compute the median of the single channel pixel intensities
            v = np.ma.median(window)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            #print('lower:',lower,' upper:',upper)
            
            #https://docs.opencv.org/3.3.1/d5/d0f/tutorial_py_gradients.html
            #https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html#sobel-and-scharr-derivatives
            """In our last example, output datatype is cv2.CV_8U or np.uint8. But there
             is a slight problem with that. Black-to-White transition is taken as Positive
             slope (it has a positive value) while White-to-Black transition is taken as
             a Negative slope (It has negative value). So when you convert data to np.uint8,
             all negative slopes are made zero. In simple words, you miss that edge.
    
             If you want to detect both edges, better option is to keep the output datatype
             to some higher forms, like cv2.CV_16S, cv2.CV_64F etc, take its absolute value
             and then convert back to cv2.CV_8U. """

            blurred = cv2.GaussianBlur(window, (3, 3), 2)
                       
            #https://docs.opencv.org/3.3.1/d3/d63/edge_8cpp-example.html
            dx = cv2.Scharr(blurred, cv2.CV_16S, 1, 0)
            dy = cv2.Scharr(blurred, cv2.CV_16S, 0, 1)
    
            auto_custom = cv2.Canny(dx,dy, threshold1=lower, threshold2=upper, L2gradient=False)
            auto_custom = np.ma.masked_where(auto_custom != 255, auto_custom)
            edges_window[y:y+window.shape[0],x:x+window.shape[1]] = auto_custom

            dp_value=1; par1=50; par2=15
            """min_dist_px, min_radii_px e max_raddi_px são definidos
            na função que lê o raster pois dependem da resolução espacial do mesmo"""

            circles = finding_circles(auto_custom,dp_value=dp_value,min_dist_px=min_dist_px,
                                      param1=par1,param2=par2,
                                      min_radii_px=min_radii_px,max_radii_px=max_radii_px)
    
            if circles is not None:
                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")
                circles[:,0] += x
                circles[:, 1] += y

                temp.append(circles)

    circles = np.unique(np.concatenate(temp, axis=0), axis=0) #Concatena os circulos encontrados pelo processo da janela deslizante
    print('Nr. Circles detected:', circles.shape[0])
    
    #create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sc, cax=cax, label=r'NDVI')
    
    destino = './png'
    output_name = 'PivotDetect_SlideWindow_'+ sat_name + '.png'
    plt.savefig(os.path.join(destino,output_name), dpi=150) 

    # to crop png https://askubuntu.com/questions/351767/how-to-crop-borders-white-spaces-from-image
    subprocess.call(["mogrify", "-trim", os.path.join(destino,output_name)])

    return circles
                        
    


#--------------------------
#   MAIN PROGRAM     
#--------------------------
if __name__ == '__main__':
    #the previous line is necessary under windows to not execute
    # main module on each child under window

    sys.stdout = Logger() # Direct std out to both terminal and to log file
    
    #Starts modifications to parallelize loops using multiprocessing module
    #Useful links:
    # Documentation of Parallel Processing and Multiprocessing in Python - https://wiki.python.org/moin/ParallelProcessing
    # https://data-flair.training/blogs/python-multiprocessing/
    # Shared variable example - https://svitla.com/blog/parallel-computing-and-multiprocessing-in-python


    """First in-Flight Radiometric Calibration of MUX and WFI on-Board CBERS-4 
       https://www.mdpi.com/2072-4292/8/5/405

    The gain coefficients are now available: 1.68, 1.62, 1.59 and 1.42 for MUX 
    and 0.379, 0.498, 0.360 and 0.351 for WFI spectral bands blue, green, red and NIR, 
    respectively, in units of (W/(m2·sr·μm))/DN."""
    
    PROCESSES = cpu_count() - 4
    print("Parent ID",os.getpid())

    # Inverse colormaps
    summer_r_cm = LinearSegmentedColormap(
        'summer_r', plt.cm.revcmap(plt.cm.summer._segmentdata))
    

    #######################################################################
    # Define params to processing:
    ####################################
    v_index = 'SAVI' #or 'NDVI'
    typeOfData =  'Amplitude' #'Greenest'
    year = '2017'
    imgs_folder = "./landsat_"+typeOfData+"SR"
    

    for file in glob.glob(os.path.join(imgs_folder,typeOfData+'SR_'+v_index+'_composite_'+year+'_*.tif')):
        year = os.path.basename(file).split('_')[3]
        path = os.path.basename(file).split('_')[4]
        row = os.path.basename(file).split('_')[5].split('.')[0]        
        sat_name = os.path.basename(file).split('.')[0]
        
        if os.path.exists(os.path.join('./shp','PivosCentraisANA'+year+'_Intersects_composite_'+path+'_'+row+'.shp')):
            file_stats_name = os.path.join('./stats',sat_name+'_ConvexHullStatistics.csv')

            if not os.path.exists(file_stats_name):

                """Satellite maps of vegetation show the density of plant growth over the entire 
                globe. The most common measurement is called the Normalized Difference Vegetation
                 Index (NDVI). Very low values of NDVI (0.1 and below) correspond to barren areas 
                 of rock, sand, or snow. Moderate values represent shrub and grassland (0.2 to 0.3), 
                 while high values indicate temperate and tropical rainforests (0.6 to 0.8).
                 https://earthobservatory.nasa.gov/Features/MeasuringVegetation"""

                #####################################################################
                # Normalizaded Difference Vegetation Index (Rouse et al.,1973):
                array_ndvi = get_band_array(file.replace('SAVI','NDVI').replace('Amplitude','Greenest'), info=False).astype('float')
                               
                ####################################################################
                # Soil Adjusted Vegetation Index (SAVI) (Huete et al.,1988):
                """An L value of 0.5 in reflectance space was found to minimize soil
                 brightness variations and eliminate the need for additional calibration
                 for different soils"""

                array_savi = get_band_array(file.replace('NDVI','SAVI').replace('Amplitude','Greenest'), info=False).astype('float')
                
                ####################################################################
                # Amplitude NDVI or SAVI:
                array_amplitude = get_band_array(file.replace('NDVI',v_index).replace('Greenest','Amplitude'), info=False).astype('float')

               
                if typeOfData == 'Amplitude':
                  print('\nUsing amplitude '+v_index+' information to delimit pivots!')    
                  array = np.uint8(norm_minmax(array_amplitude, 0, 255))
                elif typeOfData == 'Greenest':
                  if v_index == 'NDVI':
                    print('\nUsing Greenest '+v_index+' information to delimit pivots!')    
                    array = np.uint8(norm_minmax(array_ndvi, 0, 255)) 
                  elif v_index == 'SAVI':
                    print('\nUsing Greenest '+v_index+' information to delimit pivots!')    
                    array = np.uint8(norm_minmax(array_savi, 0, 255)) 

               

                #######################################################################
                # Land Use and Land Cover classification from MapBiomas Collection 5:
                file_lulc = './MapBiomas/LULC_Collection5_'+year+'_'+path+'_'+row+'.tif'
                array_lulc = get_band_array(file_lulc, info=False, scale=1.0).astype('uint8')  
               
                
                # Morphological Transformations
                # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
                kernel = np.ones((3,3),np.uint8) #Nao altera tanto o tamanho dos objetos, gera mais círculos

                """It is just opposite of erosion. Here, a pixel element is ‘1’ if atleast one pixel under the kernel
                   is ‘1’. So it increases the white region in the image or size of foreground object increases.
                   Source: https://medium.com/analytics-vidhya/morphological-transformations-of-images-using-opencv-image-processing-part-2-f64b14af2a38"""

                dilatation = cv2.dilate(np.ma.filled(array_lulc,fill_value=0),kernel,iterations = 1)
                array_lulc = dilatation
                
                
                # Detecting edges:
                edges = auto_sharr(array, sigma=0.33)                           

                print('\nMasking not crop areas using MapBiomas LULC!')
                mask = np.ones(array_lulc.shape,'int')
                mask[(array_lulc > 17) & (array_lulc < 22)] = 0 #Not masked values!
                mask[(array_lulc == 36)] = 0
                mask[(array_lulc > 38) & (array_lulc < 46)] = 0                
                edges[mask == 1] = 0
                
                dp_value=1; par1=50; par2=15
                """min_dist_px, min_radii_px e max_raddi_px são definidos
                na função que lê o raster pois dependem da resolução espacial do mesmo"""

                circles = finding_circles(edges,dp_value=dp_value,min_dist_px=min_dist_px,
                                          param1=par1,param2=par2,
                                          min_radii_px=min_radii_px,max_radii_px=max_radii_px)
                
                if circles is not None:
                    EPSG_value = 4326 #Transformation if necessary
                    if not os.path.exists('./geojson/'+sat_name+'.geojson'):
                        export_circles_detected(circles,sat_name, type='geojson', targetEPSG=EPSG_value)                         
                
                #Avalia feixo convexo com função:
                if circles is not None:
                    # convert the (x, y) coordinates and radius of the circles to integers
                    circles = np.round(circles[0, :]).astype("int")

                    f = open(file_stats_name, 'w')
                    print('i,x,y,radius,DifPointsPercent,PixelsWithin,Mean_ndvi,Std_ndvi,Mean_savi,Std_savi,Mean_amplitude,Std_amplitude,Mode_lulc',file=f)
                    
                    lock = Lock()
                    print('Creating pool with %d ThreadPool\n' % PROCESSES)
                    pool = Pool(PROCESSES)
                    start = timer()
                    for i in range(len(circles)):
                      pool.apply_async(verif_minimum_circle, (circles[i,:],edges,i,array_savi,array_ndvi,array_amplitude,array_lulc,))

                    pool.close()
                    pool.join()
                    f.close()
                    end = timer()
                    print("Targets processed using Convex Hull in: ", end - start," seconds.")       
            else:
              print('The file {} already processed!'.format(os.path.basename(file)))
        else:
          print('--------------------------------------')
          print('              Warning                 ')
          print('--------------------------------------')
          print('Aborting processing of file:',os.path.basename(file))
          print("Because there isn't a file with intersects of pivots from ANA!")

                          
                          
                          
                          

