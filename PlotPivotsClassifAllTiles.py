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
import rasterio as rio
from rasterio.plot import plotting_extent
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar #For scalebar
from geopy.distance import geodesic #Great Circle of Earth (scalebar)
import numpy as np
import os
import sys
import glob
import subprocess
from itertools import compress #filter list by boolean list
import pandas as pd
import gc
#Imports for plot using cartopy:
from cartopy import crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from math import floor
from matplotlib import patheffects
import matplotlib
if os.name == 'nt':
    matplotlib.rc('font', family='Arial')
else:  # might need tweaking, must support black triangle for N arrow
    matplotlib.rc('font', family='DejaVu Sans')


#---------------------------
# Functions:
#---------------------------
def read_pivots_ana(ano,posfix,threshold):
    source_ana = os.path.join('./shp','PivosCentraisANA'+str(ano)+'_Intersects_composite_'+posfix+'.shp')
    pivots_ana = gpd.read_file(source_ana)
    
    print('Pivos ANA antes do limiar NDVI/SAVI:',pivots_ana.shape)
    print('Pivos ANA antes do limiar NDVI/SAVI:',pivots_ana.shape,file=f_out)
    pivots_ana = pivots_ana[pivots_ana['_mean'] > threshold] #Limiar para vegetação fotossinteticamente ativa!
    print('Pivos ANA depois do limiar NDVI:',pivots_ana.shape)
    print('Pivos ANA depois do limiar NDVI:',pivots_ana.shape,file=f_out)
    
    return pivots_ana
    

def detect_intersects(pivots,pivots_target):
     
    pivots_tmp = gpd.GeoDataFrame(pivots.geometry.centroid,columns=['geometry'], crs=pivots.crs)
    try:
        pointInPolys = gpd.tools.sjoin(pivots_tmp, pivots_target, how='left', op='within')
        pivots_success = pivots.loc[pointInPolys.dropna().index].copy()
        pivots_success.drop_duplicates(subset=[col for col in pivots_success.columns if col != 'geometry'], inplace=True)

        pivots_miss = pivots.copy()
        pivots_miss.drop(pivots_success.index, inplace=True)
    except Exception:
        pivots_success = gpd.GeoDataFrame()
        pivots_miss = gpd.GeoDataFrame()
        

    return pivots_success, pivots_miss


# Source: https://unix.stackexchange.com/a/590637
class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open("PlotPivotsAllTilesClassifOverVegIndex.log", "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            #this flush method is needed for python 3 compatibility.
            #this handles the flush command by doing nothing.
            #you might want to specify some extra behavior here.
            pass
    
    
def utm_from_lon(lon):
    """
    utm_from_lon - UTM zone for a longitude

    Not right for some polar regions (Norway, Svalbard, Antartica)

    :param float lon: longitude
    :return: UTM zone number
    :rtype: int
    """
    return floor( ( lon + 180 ) / 6) + 1


def scale_bar(ax, proj, length, location=(0.5, 0.05), linewidth=3,
              units='km', m_per_unit=1000):
    """

    http://stackoverflow.com/a/35705477/1072212
    ax is the axes to draw the scalebar on.
    proj is the projection the axes are in
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
    length is the length of the scalebar in km.
    linewidth is the thickness of the scalebar.
    units is the name of the unit
    m_per_unit is the number of meters in a unit
    """
    # find lat/lon center to find best UTM zone
    x0, x1, y0, y1 = ax.get_extent(proj.as_geodetic())
    # Projection in metres
    utm = ccrs.UTM(utm_from_lon((x0+x1)/2))
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(utm)
    # Turn the specified scalebar location into coordinates in metres
    sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbcx - length * m_per_unit/2, sbcx + length * m_per_unit/2]
    # buffer for scalebar
    buffer = [patheffects.withStroke(linewidth=5, foreground="w")]
    # Plot the scalebar with buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
        linewidth=linewidth, path_effects=buffer)
    # buffer for text
    buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
    # Plot the scalebar label
    t0 = ax.text(sbcx, sbcy, str(length) + ' ' + units, transform=utm,
        horizontalalignment='center', verticalalignment='bottom',
        path_effects=buffer, zorder=2)
    left = x0+(x1-x0)*0.05
    # Plot the N arrow
    t1 = ax.text(left, sbcy, u'\u25B2\nN', transform=utm,
        horizontalalignment='center', verticalalignment='bottom',
        path_effects=buffer, zorder=2)
    # Plot the scalebar without buffer, in case covered by text buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
        linewidth=linewidth, zorder=3)


def far_optimization(thresh,pivots_ana_tmp,pivots):
    if len(thresh) != 2:
        raise Exception("The thresholds bounds must be have 2 values!")
    
    for threshold in np.arange(min(thresh),max(thresh)+0.01,0.01)[::-1]:        
        pivots_ana = pivots_ana_tmp[pivots_ana_tmp['_mean'] > threshold] #Limiar para vegetação fotossinteticamente ativa!
        print('Shape:',pivots_ana.shape)
        
        pivots_success, pivots_miss = detect_intersects(pivots,pivots_ana)    
        pivots_ana_success, pivots_ana_miss = detect_intersects(pivots_ana,pivots)
        
        intersects_count = pivots_ana_success.shape[0]
        try:
            percent = (intersects_count*100)/float(len(pivots_ana.index))
        except ZeroDivisionError:
            percent = 0.0
        #percent = min((intersects_count*100)/float(len(pivots_ana.index)),100)
        
        intersects_count = pivots_success.shape[0]
        try:
            far = ((len(pivots.index) - intersects_count)*100)/float(len(pivots.index))
        except ZeroDivisionError:
            far = 0.0
        #far = min(((len(pivots.index) - intersects_count)*100)/float(len(pivots.index)),100)
        print("Threshold:",threshold,'FAR:',far)
        
        if far <= 0.:
            break

    print('Pivots (sucess/miss):',len(pivots_success),len(pivots_miss))
    print('Pivots ANA intersects Pivots (sucess/miss):',len(pivots_ana_success),len(pivots_ana_miss))
            
    return pivots_ana,pivots_success,pivots_miss,intersects_count,percent,far


def PlotClassifOverVegInd(name, typeOfData, pivots_success, pivot_miss, pivots_ana, pivots_ana_miss, percent, far):
    outfp = '_'.join(name[0:6]) + '_pivotsClassified_all.png'
    
    if not os.path.exists(os.path.join('./png',outfp)):    
        raster_file = os.path.join('./landsat_'+typeOfData+'SR','_'.join(name[0:6])+'.tif')
       
        if os.path.exists(raster_file):
            print("Processing:\n",raster_file)

            with rio.open(raster_file) as src:
                if src.crs.to_string() != 'EPSG:4326':
                    output_file = read_reproject_raster(raster_file)
                    
                    with rio.open(output_file) as src:
                        ndvi = src.read()
                        ndvi_meta = src.profile
                        ndvi_extent = rio.plot.plotting_extent(src, transform=None)
                        ndvi = np.ma.masked_where(ndvi[0] == src.nodata, ndvi[0])
                else:
                    ndvi = src.read()
                    ndvi_meta = src.profile
                    ndvi_extent = rio.plot.plotting_extent(src, transform=None)
                    ndvi = np.ma.masked_where(ndvi[0] == src.nodata, ndvi[0])
                

            print('Vegetation Index crs: ', ndvi_meta['crs'])
            print('Vegetation Index ranges:',ndvi.min(),ndvi.max())

            fig, ax = plt.subplots(figsize=(10, 8))

            ax.set_title("Pivots detected: " + str(pivots_success.shape[0]) + ' Mapped by ANA: ' + str(len(pivots_ana.index)) +
                      ' Recall: ' + "{0:.2f}".format(percent)+'%' + ' FAR: '+ "{0:.2f}".format(far)+'%', fontsize=10);

            sc = ax.imshow(ndvi, vmin=0, vmax=1, cmap='summer_r', interpolation=None,
                      extent=ndvi_extent)

            if '_'.join(name[4:6]) == '220_72':
                ax.set_xlim(-47.79, -45.67) #only for figure paper to path/row 220/72
                ax.xaxis.set_major_locator(plt.MaxNLocator(8)) #only for figure paper
                ax.yaxis.set_major_locator(plt.MaxNLocator(7)) #only for figure paper
                
                #Custom labels coordinates:
                ax.set_xticklabels(["{0:.2f}°W".format(tick*(-1)) for tick in ax.get_xticks()], fontsize=10)
                ax.set_yticklabels(["{0:.2f}$\degree$S".format(tick*(-1)) for tick in ax.get_yticks()], fontsize=10)

                # Create scale bar
                x1, x2, y1, y2 = ax.axis()
                _y = (y1+y2)/2
                p1, p2 = (int(x1), _y), (int(x1)+1, _y)
                meter_per_deg = geodesic(p1, p2).meters
                scalebar = ScaleBar(meter_per_deg, units="m", location='lower center', color='k', box_alpha=0)
                ax.add_artist(scalebar)

                # buffer for text
                buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
                left = x1+(x2-x1)*0.05
                sbcy = y1+(y2-y1)*0.02
                # Plot the N arrow
                t1 = ax.text(left, sbcy, u'\u25B2\nN', horizontalalignment='center', verticalalignment='bottom',
                    path_effects=buffer, zorder=2)
                
                

            """ Create an axes on the right side of ax. The width of cax will be 5%
                of ax and the padding between cax and ax will be fixed at 0.05 inch.
                divider = make_axes_locatable(ax[0])"""
            divider = make_axes_locatable(ax)
            
            if '_'.join(name[4:6]) == '220_72':
                cax = divider.append_axes("right", size="5%", pad=3) #only for figure paper
            else:
                cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(sc, cax=cax, label=v_index+' AMPLITUDE')
            
            if '_'.join(name[4:6]) == '220_72':
                # inset axes.... #only for figure paper
                axins = ax.inset_axes([1.13, 0.53, 0.47, 0.47])
                axins.imshow(ndvi, vmin=0, vmax=1, cmap='summer_r', interpolation=None,
                      extent=ndvi_extent)
                # sub region of the original image
                x1, x2, y1, y2 = -47.3, -47, -16.8, -16.5
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.xaxis.set_major_locator(plt.MaxNLocator(4))
                axins.yaxis.set_major_locator(plt.MaxNLocator(4))
                #axins.axis('off')
                #axins.set_xticklabels('')
                #axins.set_yticklabels('')

                #Custom labels coordinates:
                axins.set_xticklabels(["{0:.2f}°W".format(tick*(-1)) for tick in axins.get_xticks()], fontsize=9)
                axins.set_yticklabels(["{0:.2f}°S".format(tick*(-1)) for tick in axins.get_yticks()], fontsize=9)

                # Create scale bar
                x1, x2, y1, y2 = axins.axis()
                _y = (y1+y2)/2
                p1, p2 = (int(x1), _y), (int(x1)+1, _y)
                meter_per_deg = geodesic(p1, p2).meters
                scalebar = ScaleBar(meter_per_deg, units="m", location="lower left", color='k', box_alpha=0)
                axins.add_artist(scalebar)

                ax.indicate_inset_zoom(axins, edgecolor="black", linestyle='--')

                # inset axes.... #only for figure paper
                axens = ax.inset_axes([1.13, 0, 0.47, 0.47])
                axens.imshow(ndvi, vmin=0, vmax=1, cmap='summer_r', interpolation=None,
                      extent=ndvi_extent)
                # sub region of the original image
                x1, x2, y1, y2 = -46.5, -46.2, -17.1, -16.8
                axens.set_xlim(x1, x2)
                axens.set_ylim(y1, y2)
                axens.xaxis.set_major_locator(plt.MaxNLocator(4))
                axens.yaxis.set_major_locator(plt.MaxNLocator(4))

                #Custom labels coordinates:
                axens.set_xticklabels(["{0:.2f}°W".format(tick*(-1)) for tick in axens.get_xticks()], fontsize=9)
                axens.set_yticklabels(["{0:.2f}°S".format(tick*(-1)) for tick in axens.get_yticks()], fontsize=9)

                # Create scale bar
                x1, x2, y1, y2 = axens.axis()
                _y = (y1+y2)/2
                p1, p2 = (int(x1), _y), (int(x1)+1, _y)
                meter_per_deg = geodesic(p1, p2).meters
                scalebar = ScaleBar(meter_per_deg, units="m", location="lower left", color='k', box_alpha=0)
                axens.add_artist(scalebar)


                ax.indicate_inset_zoom(axens, edgecolor="black", linestyle='--')

            leng_index = [False] * 3

            # Pivots mapped per ANA
            if len(pivots_ana) > 0:
                #pivots_ana.plot(ax=ax, color=colors[0])
                pivots_ana_miss.plot(ax=ax, color=colors[0])
                if '_'.join(name[4:6]) == '220_72':
                    pivots_ana_miss.plot(ax=axins, color=colors[0]) #only for figure paper
                    pivots_ana_miss.plot(ax=axens, color=colors[0]) #only for figure paper
                leng_index[0] = True
            
            # Pivots matching with ANA shape
            if len(pivots_success) > 0:
                pivots_success.plot(ax=ax, color=colors[1])
                if '_'.join(name[4:6]) == '220_72':
                    pivots_success.plot(ax=axins, color=colors[1]) #only for figure paper
                    pivots_success.plot(ax=axens, color=colors[1]) #only for figure paper
                leng_index[1] = True

            # Pivots not matching with ANA shape
            if len(pivots_miss) > 0:
                pivots_miss.plot(ax=ax, color=colors[2])
                if '_'.join(name[4:6]) == '220_72':
                    pivots_miss.plot(ax=axins, color=colors[2]) #only for figure paper
                    pivots_miss.plot(ax=axens, color=colors[2]) #only for figure paper
                leng_index[2] = True    


            #https://matplotlib.org/gallery/text_labels_and_annotations/custom_legends.html
            legend_elements = [Line2D([0], [0], marker='.', color='w', label="Mapped only by ANA",
                                  markerfacecolor=colors[0], markersize=14),
                               Line2D([0], [0], marker='.', color='w', label="CHT matching with ANA",
                                      markerfacecolor=colors[1], markersize=14),
                               Line2D([0], [0], marker='.', color='w', label="CHT not matching with ANA",
                                      markerfacecolor=colors[2], markersize=14)]
            
            if '_'.join(name[4:6]) == '220_72':
                ax.legend(handles=list(compress(legend_elements, leng_index)), loc='upper center', bbox_to_anchor=(0.8, -0.05), fancybox=False,
                       shadow=False, ncol=3, prop={'size': 12}, frameon=False, handletextpad=0.1) #title="CO2 flux" #only for figure paper  
            else:
                ax.legend(handles=list(compress(legend_elements, leng_index)), loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False,
                       shadow=False, ncol=3, prop={'size': 12}, frameon=False, handletextpad=0.1) #title="CO2 flux"

            #plt.show()
            

            # Save the figure as png file with resolution in dpi
            outfp = '_'.join(name[0:6]) + '_pivotsClassified_all.png'
            plt.savefig(os.path.join('./png',outfp))#, dpi=150)
            plt.clf() #clear all axes
            plt.close() #close current figure window

            # to crop png https://askubuntu.com/questions/351767/how-to-crop-borders-white-spaces-from-image
            subprocess.call(["mogrify", "-trim", os.path.join('./png',outfp)])
            

         
#---------------------------
# Main:
#---------------------------
ano=2017
v_index =  'SAVI' #'NDVI'
typeOfData = 'Amplitude' #'Greenest'

sys.stdout = Logger() # Direct std out to both terminal and to log file
f_out = open('ValidaPivos_AllTiles_Statistics.txt', 'a')

# Inverse colormaps
orig_map=plt.cm.get_cmap('summer')
summer_r_cm = orig_map.reversed('summer_r')


#https://matplotlib.org/3.1.0/gallery/color/named_colors.html
colors = ["tab:orange","tab:blue","tab:red"]

#Correct States from Brazil:
#https://www.ibge.gov.br/geociencias/downloads-geociencias.html > organizacao_do_territorio > malhas_territoriais > malhas_municipais > municipio_2020 > Brasil > BR
source_map = '/home/image/Downloads/BR_UF_2020/BR_UF_2020.shp'
brazil = gpd.read_file(source_map)

#Read file classified:
pivots = gpd.read_file('./geojson/'+typeOfData+'SR_'+v_index+'_composite_'+str(ano)+'_all_NotFiltered.geojson')
        

#Read pivots mapped by ANA:        
threshold_ana = 0.5

list_file_stats = sorted(glob.glob('./stats/'+typeOfData+'SR_'+v_index+'_composite_'+str(ano)+'_220_72_ConvexHullStatistics.csv'))

if len(list_file_stats) > 0:
    print('Reading spatial information of pivots mapped by ANA for this tiles...')
    print('Nr. of files Stats available:',len(list_file_stats),'reading spatial information of pivots mapped by ANA for this tiles...', file=f_out)
    pivots_ana = gpd.GeoDataFrame()
    gpd_df_final = gpd.GeoDataFrame()
      
    for cont,file_stats in enumerate(list_file_stats):
          name = os.path.basename(file_stats).split('_')
          pivots_ana_tmp = read_pivots_ana(ano,'_'.join(name[4:6]),threshold=threshold_ana)
          pivots_ana = pivots_ana.append(pivots_ana_tmp, ignore_index=True)

          #Verify recall per tile:
          path_row = '_'.join(os.path.basename(file_stats).split('_')[4:6])
          gpd_df_extent = gpd.read_file("./shp/GreenestSR_NDVI_composite_"+str(ano)+'_'+path_row+"_extent_polygon.shp")

          print(f'Finding circles classified as pivot that intersects scene ({path_row})...')
          
          pointInPolys = gpd.tools.sjoin(pivots, gpd_df_extent, how='left', op='within')
          pivots_tmp = pivots.loc[pointInPolys.dropna().index]
              
          pivots_success, pivots_miss = detect_intersects(pivots_tmp,pivots_ana_tmp)
          pivots_ana_success, pivots_ana_miss = detect_intersects(pivots_ana_tmp,pivots_tmp)
          print('Pivots in Tile',path_row,'(sucess/miss):',len(pivots_success),len(pivots_miss))
          print('Pivots ANA intersects Pivots (sucess/miss):',len(pivots_ana_success),len(pivots_ana_miss))
          print('Pivots in Tile',path_row,'(sucess/miss):',len(pivots_success),len(pivots_miss),file=f_out)
          print('Pivots ANA intersects Pivots (sucess/miss):',len(pivots_ana_success),len(pivots_ana_miss),file=f_out)
          

          intersects_count = pivots_ana_success.shape[0]
          try:
              percent = (intersects_count*100)/float(len(pivots_ana_tmp.index))
          except ZeroDivisionError:
              percent = 0.0

          intersects_count = pivots_success.shape[0]
          try:
              far = ((len(pivots_tmp.index) - intersects_count)*100)/float(len(pivots_tmp.index))
          except ZeroDivisionError:
              far = 0.0
          
          print('Pivots detected:',len(pivots_tmp.index),'Mapped by ANA:',len(pivots_ana_tmp.index),'Matching with ANA:',intersects_count,'Recall:',percent, 'FAR:',far)
          print('Pivots detected:',len(pivots_tmp.index),'Mapped by ANA:',len(pivots_ana_tmp.index),'Matching with ANA:',intersects_count,'Recall:',percent, 'FAR:',far,file=f_out)

          ########################
          # Plot
          #############
          #PlotClassifOverVegInd(name, typeOfData, pivots_success, pivots_miss, pivots_ana, pivots_ana_miss, percent, far)
          #exit()
          
          gpd_df_extent['path_row'] = path_row
          gpd_df_extent['recall'] = percent
          gpd_df_extent['far'] = far
          gpd_df_final = gpd_df_final.append(gpd_df_extent)
          del pivots_ana_tmp


    #Clean variables:
    gc.collect()                     

    if cont > 0:
        print('Finding intersect between pivots ANA and pivots detected for all tiles...')
        #Detect intersects:
        pivots_success, pivots_miss = detect_intersects(pivots,pivots_ana)
        pivots_ana_success, pivots_ana_miss = detect_intersects(pivots_ana,pivots)
        print('Pivots (sucess/miss):',len(pivots_success),len(pivots_miss))
        print('Pivots ANA intersects Pivots (sucess/miss):',len(pivots_ana_success),len(pivots_ana_miss))
        print('Pivots (sucess/miss):',len(pivots_success),len(pivots_miss),file=f_out)
        print('Pivots ANA intersects Pivots (sucess/miss):',len(pivots_ana_success),len(pivots_ana_miss),file=f_out)
              
      

        intersects_count = pivots_ana_success.shape[0]
        percent = min((intersects_count*100)/float(len(pivots_ana.index)),100)

        intersects_count = pivots_success.shape[0]
        far = min(((len(pivots.index) - intersects_count)*100)/float(len(pivots.index)),100)
        print('Pivots detected:',len(pivots.index),'Mapped by ANA:',len(pivots_ana.index),'Matching with ANA:',intersects_count,'Recall:',percent, 'FAR:',far)

        
        print('All tiles Pivots detected: ',len(pivots.index),'Mapped by ANA:',len(pivots_ana.index),'Matching with ANA:',
              intersects_count,'Recall:',percent, 'FAR:',far,file=f_out)       


        ################################################
        # Plotting with CartoPy and GeoPandas
        ######################################
        #Source: https://geopandas.org/gallery/cartopy_convert.html

        fig = plt.figure(figsize=(8,7), dpi=150) # open matplotlib figure

        # Define the CartoPy CRS object.
        crs = ccrs.PlateCarree()  #Error: proj_create_operations: Source and target ellipsoid do not belong to the same celestial body
                                   #Comment: https://github.com/SciTools/cartopy/pull/1252#issuecomment-508869567


        #######################################
        # Formatting the Cartopy plot
        #######################################
        # Source: https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy

        ax1 = plt.axes(projection=crs) # project using coordinate reference system (CRS)
        extent = [-74.3,-34.5, -34, 5.5] # Brazil bounds [xmin,xmax,ymin,ymax]
        ax1.set_extent(extent) # set extents

        brazil.boundary.plot(ax=ax1, edgecolor='black', linewidth=0.1)

        gpd_df_final.plot("recall", cmap='RdYlGn', edgecolor='white', vmin=0, vmax=100, ax=ax1, legend=False)

        #Add colorbar
        norm = Normalize(vmin=0, vmax=100)
        n_cmap = cm.ScalarMappable(norm=norm, cmap='RdYlGn')
        n_cmap.set_array([])
        ax1.get_figure().colorbar(n_cmap, ax=ax1, orientation='vertical', label='Recall (%)')

        #suptitle = 'Pivots detected: {0} Mapped by ANA: {1} Matching with ANA: {2}'.format(len(pivots.index),len(pivots_ana.index),intersects_count)
        #title = r'Recall: $\bf{{{0:.2f}}}$% Far: $\bf{{{1:.2f}}}$%'.format(percent,far)
        #plt.suptitle(suptitle,fontsize=10);
        print('-'*70)
        print('\nATENCAO - utilizando Pivot ANA Success para ilustrar o nr de pivos detectado:',len(pivots_ana_success))
        print('-'*70)
        #title = 'Pivots detected: {0} Mapped by ANA: {1} Matching with ANA: {2} \nRecall: {3:.2f}% Far: {4:.2f}%'.format(len(pivots.index),len(pivots_ana.index),intersects_count,percent,far)
        title = 'Pivots detected: {0} Mapped by ANA: {1} Recall: {2:.2f}% Far: {3:.2f}%\n'.format(len(pivots_ana_success),len(pivots_ana.index),percent,far)
        #title = 'Pivots detected: {0} Mapped by ANA: {1} Recall: {2:.2f}% Far: {3:.2f}%\n'.format(7358,8774,83.86,1.60)
        plt.title(title,{'fontsize':12});


        ax1.set_xticks(np.linspace(extent[0],extent[1],7),crs=ccrs.PlateCarree()) # set longitude indicators
        ax1.set_yticks(np.linspace(extent[2],extent[3],7)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
        lon_formatter = LongitudeFormatter(number_format='0.1f',degree_symbol='',dateline_direction_label=True) # format lons
        lat_formatter = LatitudeFormatter(number_format='0.1f',degree_symbol='') # format lats
        ax1.xaxis.set_major_formatter(lon_formatter) # set lons
        ax1.yaxis.set_major_formatter(lat_formatter) # set lats
        ax1.xaxis.set_tick_params(labelsize=8)
        ax1.yaxis.set_tick_params(labelsize=8)

        scale_bar(ax1, ccrs.PlateCarree(), 200)  # 200 km scale bar
        # or to use m instead of km
        # scale_bar(ax, ccrs.Mercator(), 100000, m_per_unit=1, units='m')
        # or to use miles instead of km
        # scale_bar(ax, ccrs.Mercator(), 60, m_per_unit=1609.34, units='miles')


        #plt.show()
        outfp = './png/ChoroplethMap_RecallPivots.png'                                              
        plt.savefig(outfp, dpi=150)  # Animacao dpi=100

        # to crop png https://askubuntu.com/questions/351767/how-to-crop-borders-white-spaces-from-image
        subprocess.call(["mogrify", "-trim", outfp])
        
f_out.close()
