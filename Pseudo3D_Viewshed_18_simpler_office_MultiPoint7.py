import timeit
import numpy as np
import matplotlib.pyplot as plt
import shapely.wkb
import csv
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
from qgis.PyQt.QtCore import QVariant
from itertools import combinations
from shapely.ops import cascaded_union

#cell_size = 1.767
sight_width = 4 #the width of the 2D line of sight in m

#Input specificiation
point_id = 'id' #has to be numeric
build_h = 'Bld_h_2018'
terr_h = 'Ter_2018'
type = 'type'

#timer
tic=timeit.default_timer()

#read inputs
observer_layer = qgis.utils.iface.mapCanvas().currentLayer()

#set up coor array and height array
all_pts = np.array([[f.geometry().asPoint()[0], f.geometry().asPoint()[1]]
                                        for f in observer_layer.getFeatures()], dtype="float")
height_ar = np.array([[f[build_h], f[terr_h]] for f in observer_layer.getFeatures()], dtype="float") #already rounded at input

#So that I can keep track which point is which
all_ids = np.array([f[point_id] for f in observer_layer.getFeatures()], dtype="int")

#set up empty array to fill out
feat_count = len([f for f in observer_layer.getFeatures()])
out_array = np.zeros((feat_count, feat_count), dtype=int, order='C')

#set up an empty layer 
fields = QgsFields()
fields.append(QgsField('id', QVariant.Int))
fields.append(QgsField('vs_score', QVariant.Int))
out_layer = QgsVectorLayer('MultiPolygon', 'viewsheds', 'memory')
out_layer.dataProvider().addAttributes(fields) #[2] is QgsFields
out_layer.updateFields()


#get a list of 1 degree line gradients, using tangent in range 90 270 so that it is continuous
sight_bands = []
for i in range(360):
    sight_bands.append((i, i+1))

toc=timeit.default_timer()
print "before loop : " + str((toc - tic)/60)

viewshed_feats = []
for point in observer_layer.selectedFeatures():
    
    #sight line beginning
    p1 = np.array(point.geometry().asPoint())
    p1h = np.array(point[build_h])
    p1terr = np.array(point[terr_h])
    p1_id = int(point[point_id])
    
    #remove p1 and save as new variable
    filter = (all_pts[:,0] != p1[0]) | (all_pts[:,1] != p1[1])
    all_pts_filtered = all_pts[filter]
    height_ar_filtered = height_ar[filter]
    all_ids_filtered = all_ids[filter]

    #direction vectors, distance and gradient to all others
    dir_vec = all_pts_filtered - p1
    dist = np.sqrt(np.einsum('ij,ij->i', dir_vec, dir_vec))
    gradients = np.divide(dir_vec[:,0], dir_vec[:,1], out=np.zeros_like(dir_vec[:,0]), where=dir_vec[:,0]!=0)
    point_angles = abs(np.degrees(np.arctan(gradients)))
    
    #get angles in correct form starting from the east and going counterclock wise
    x = dir_vec[:,0]
    y = dir_vec[:,1]
    point_angles[(x == 0) & (y > 0)] = 90
    point_angles[(x == 0) & (y < 0)] = 270
    point_angles[(y == 0) & (x > 0)] = 0
    point_angles[(y == 0) & (x < 0)] = 180
    
    point_angles[(x > 0) & (y > 0)] -= 90 #quadrant 1
    point_angles[(x > 0) & (y > 0)] = np.abs(point_angles[(x > 0) & (y > 0)])
    
    point_angles[(x < 0) & (y > 0)] += 90 #quadrant 2
    
    point_angles[(x < 0) & (y < 0)] -= 90 #quadrant 3
    point_angles[(x < 0) & (y < 0)] = np.abs(point_angles[(x < 0) & (y < 0)])
    point_angles[(x < 0) & (y < 0)] += 180
    
    point_angles[(x > 0) & (y < 0)] += 270 #quadrant 4
    
    h_diff = np.sum(height_ar_filtered, axis=1) - (p1h + p1terr)
    tan_z = np.divide(h_diff,  dist, out=np.zeros_like(dist), where=h_diff !=0)
    
    #set up h diff and tan_z arrays
    h_diff_roof_roof = np.sum(height_ar_filtered, axis=1) - (p1h + p1terr)
    h_diff_roof_middle = (height_ar_filtered[:,0] / 2 + height_ar_filtered[:,1] ) - (p1h + p1terr)
    h_diff_roof_bottom = (1.6 + height_ar_filtered[:,1] ) - (p1h + p1terr)
    
    h_diff_middle_roof = np.sum(height_ar_filtered, axis=1) - (p1h/2 + p1terr)
    h_diff_middle_middle = (height_ar_filtered[:,0] / 2 + height_ar_filtered[:,1] ) - (p1h/2 + p1terr)
    h_diff_middle_bottom = (1.6 + height_ar_filtered[:,1] ) - (p1h/2 + p1terr)
    
    h_diff_bottom_roof = np.sum(height_ar_filtered, axis=1) - (p1h/5 + p1terr)
    h_diff_bottom_middle = (height_ar_filtered[:,0] / 2 + height_ar_filtered[:,1] ) - (p1h/5 + p1terr)
    h_diff_bottom_bottom = (1.6 + height_ar_filtered[:,1] ) - (p1h/5 + p1terr)
    
    tan_z_r_r = np.divide(h_diff_roof_roof,  dist, out=np.zeros_like(dist), where=h_diff_roof_roof !=0)
    tan_z_r_m = np.divide(h_diff_roof_middle,  dist, out=np.zeros_like(dist), where=h_diff_roof_middle !=0)
    tan_z_r_b = np.divide(h_diff_roof_bottom,  dist, out=np.zeros_like(dist), where=h_diff_roof_bottom !=0)
    
    tan_z_m_r = np.divide(h_diff_middle_roof,  dist, out=np.zeros_like(dist), where=h_diff_middle_roof !=0)
    tan_z_m_m = np.divide(h_diff_middle_middle,  dist, out=np.zeros_like(dist), where=h_diff_middle_middle !=0)
    tan_z_m_b = np.divide(h_diff_middle_bottom,  dist, out=np.zeros_like(dist), where=h_diff_middle_bottom !=0)
    
    tan_z_b_r = np.divide(h_diff_bottom_roof,  dist, out=np.zeros_like(dist), where=h_diff_bottom_roof !=0)
    tan_z_b_m = np.divide(h_diff_bottom_middle,  dist, out=np.zeros_like(dist), where=h_diff_bottom_middle !=0)
    tan_z_b_b = np.divide(h_diff_bottom_bottom,  dist, out=np.zeros_like(dist), where=h_diff_bottom_bottom !=0)
    
    #sort based on distance to p1
    sort_dist = dist.argsort()
    dir_vec = dir_vec[sort_dist]
    dist = dist[sort_dist]
    gradients = gradients[sort_dist]
    point_angles = point_angles[sort_dist]
    tan_z = tan_z[sort_dist]
    height_ar_filtered = height_ar_filtered[sort_dist]
    all_pts_filtered = all_pts_filtered[sort_dist]
    all_ids_filtered = all_ids_filtered[sort_dist]
    
    tan_z_r_r = tan_z_r_r[sort_dist]
    tan_z_r_m = tan_z_r_m[sort_dist]
    tan_z_r_b = tan_z_r_b[sort_dist] 
                      
    tan_z_m_r = tan_z_m_r[sort_dist] 
    tan_z_m_m = tan_z_m_m[sort_dist] 
    tan_z_m_b = tan_z_m_b[sort_dist] 
                      
    tan_z_b_r = tan_z_b_r[sort_dist] 
    tan_z_b_m = tan_z_b_m[sort_dist] 
    tan_z_b_b = tan_z_b_b[sort_dist] 

    tan_z_all = np.stack((
                                tan_z_r_r, tan_z_r_m, tan_z_r_b, \
                                tan_z_m_r, tan_z_m_m, tan_z_m_b, \
                                tan_z_b_r, tan_z_b_m, tan_z_b_b
                                ),axis=0)


    #resulting 1d array
    point_res = np.zeros((1,1), dtype=[(str(id), 'i4') for id in all_ids_filtered])
    
    #initialize the viewshed with the geom of the observer point
    viewshed_geom = QgsGeometry.fromWkt(point.geometry().buffer(2,4).boundingBox().asWktPolygon()).buffer(0,2)
    for sb in sight_bands:
        #find points within half a cell size to the middle of the sight band using the sin of the angle between the point vector and the sight band middle line
        angl_mid_line = sb[0] + abs((sb[0] - sb[1])/2.0)
        angl_diff = (angl_mid_line) - (point_angles)
        sin_angl_between = np.sin(np.radians(angl_diff))
        dist_to_mid_line = np.abs(sin_angl_between * dist)
        
        #check in which half is the sb using the Y of the dir vec
        if sb[0] in range(0,90):
            in_sb = (((point_angles >= sb[0]) & (point_angles <= sb[1])) | ((dist_to_mid_line <= sight_width/2.0) & (dir_vec[:,0] >= 0) & (dir_vec[:,1] >= 0)))
        elif sb[0] in range(90,180):
            in_sb = (((point_angles >= sb[0]) & (point_angles <= sb[1])) | ((dist_to_mid_line <= sight_width/2.0)  & (dir_vec[:,0] <= 0) & (dir_vec[:,1] >= 0)))
        elif sb[0] in range(180,270):
            in_sb = (((point_angles >= sb[0]) & (point_angles <= sb[1])) | ((dist_to_mid_line <= sight_width/2.0)  & (dir_vec[:,0] <= 0) & (dir_vec[:,1] <= 0)))
        else:
            in_sb = (((point_angles >= sb[0]) & (point_angles <= sb[1])) | ((dist_to_mid_line <= sight_width/2.0)  & (dir_vec[:,0] >= 0) & (dir_vec[:,1] <= 0)))

        sb_pts = all_pts_filtered[in_sb]
        sb_ids = all_ids_filtered[in_sb]

        tan_z_in_sb = tan_z_all[:,in_sb]
        #check tangents in z dimension
        r_tan = -np.inf
        m_tan =  -np.inf
        b_tan = -np.inf
        
        for i in range(tan_z_in_sb.shape[1]):
            if tan_z_in_sb[tan_z_in_sb[:,i:] > r_tan].size == 0: #if no more cell to be seen
                break
            else:
                #top
                con1 = (tan_z_in_sb[0:3,i] > r_tan).sum()
                if con1 > 0:
                    r_tan = tan_z_in_sb[0,i]
                
                #mid
                con2 = (tan_z_in_sb[3:6,i] > m_tan).sum()
                if con2 > 0:
                    m_tan = tan_z_in_sb[3,i]

                #bottom
                con3 = (tan_z_in_sb[6:9,i] > b_tan).sum()
                if con3 > 0:
                    b_tan = tan_z_in_sb[6,i]
                
                s = con1 + con2 + con3
                if s > 0:
                    target_point = QgsGeometry().fromPoint(QgsPoint(sb_pts[i][0], sb_pts[i][1])).buffer(2,4)
                    target_point_poly = QgsGeometry.fromWkt(target_point.boundingBox().asWktPolygon())
                    viewshed_geom = viewshed_geom.combine(target_point_poly)
                    out_array[p1_id,sb_ids[i]] = s
                    out_array[sb_ids[i], p1_id] = s
                else:
                    continue
    #new feat
    print out_array
    score = int(np.sum(out_array[point[point_id],:]))
    viewshed_feat = QgsFeature(fields)
    viewshed_feat.setAttribute('id', point[point_id])
    viewshed_feat.setAttribute('vs_score', score)
    viewshed_feat.setGeometry(viewshed_geom)
    viewshed_feats.append(viewshed_feat)
    
    
toc=timeit.default_timer()
print "before file saving : " + str((toc - tic)/60)

out_layer.dataProvider().addFeatures(viewshed_feats)
out_layer.updateExtents()
#Write Output
error = QgsVectorFileWriter.writeAsVectorFormat(out_layer, r'D:\HiDriveFolder\HiDrive\Ivan\Viewshed\output\Viewshed_2018', "utf-8", None, 'SQLite', False, None, ['SPATIALITE=YES', ])
#error = QgsVectorFileWriter.writeAsVectorFormat(out_layer, r'S:\HiDrive\Ivan\Viewshed\output\Viewshed_1906', "utf-8", None, 'SQLite', False, None, ['SPATIALITE=YES', ])

np.savetxt(r'D:\HiDriveFolder\HiDrive\Ivan\Viewshed\output\Viewshed_2018.txt', out_array, fmt='%i', delimiter=',')
#np.savetxt(r'S:\HiDrive\Ivan\Viewshed\output\Viewshed_1906.txt', out_array, fmt='%i', delimiter=',')

toc=timeit.default_timer()
print "finished in : " + str((toc - tic)/60)