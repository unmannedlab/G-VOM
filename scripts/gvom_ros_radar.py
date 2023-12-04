import rospy
import numpy as np
import gvom
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import PointCloud2
import tf2_ros
import tf
import ros_numpy
import time

import matplotlib.pyplot as plt
import csv

voxel_mapper = None
listener = None
transformer = None
odom_data = None
tfBuffer = None
lidar_debug_pub = None
odom_frame = "robot/odom"
new_data = False
lidar_time = None

def odom_callback(data):
    global odom_data
    odom_data = (data.pose.pose.position.x,data.pose.pose.position.y,data.pose.pose.position.z)
    #print("got odom")

def lidar_callback(data):
    global new_data, lidar_time
    print("got lidar")
    lidar_time = data.header.stamp
    if odom_data == None:
        print("no odom")
        return

    if not voxel_mapper is None:
        #try:
            scan_time = time.time()
            lidar_frame = data.header.frame_id
            #listener.waitForTransform(odom_frame,lidar_frame,   data.header.stamp,rospy.Duration(0.01))
            trans = tfBuffer.lookup_transform(odom_frame,lidar_frame,   data.header.stamp,rospy.Duration(2.2))

            translation = np.zeros([3])
            translation[0] = trans.transform.translation.x
            translation[1] = trans.transform.translation.y
            translation[2] = trans.transform.translation.z

            rotation = np.zeros([4])
            rotation[0] = trans.transform.rotation.x
            rotation[1] = trans.transform.rotation.y
            rotation[2] = trans.transform.rotation.z
            rotation[3] = trans.transform.rotation.w

            tf_matrix = transformer.fromTranslationRotation(translation,rotation)
            
            pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
            voxel_mapper.Process_pointcloud(pc,odom_data,tf_matrix)
            new_data = True
            
            print("     pointcloud rate = " + str(1.0 / (time.time() - scan_time)))
        #except:
         #   print("bad tf")
    pass

def radar_callback(data):
    global new_data
    print("got radar")
   
    if odom_data == None:
        print("no odom")
        return

    if not voxel_mapper is None:
        #try:
            scan_time = time.time()
            lidar_frame = data.header.frame_id
            #listener.waitForTransform(odom_frame,lidar_frame,   data.header.stamp,rospy.Duration(0.01))
            trans = tfBuffer.lookup_transform(odom_frame,lidar_frame,   data.header.stamp,rospy.Duration(2.5))
            #print(trans)
            

            translation = np.zeros([3])
            translation[0] = trans.transform.translation.x
            translation[1] = trans.transform.translation.y
            translation[2] = trans.transform.translation.z

            rotation = np.zeros([4])
            rotation[0] = trans.transform.rotation.x
            rotation[1] = trans.transform.rotation.y
            rotation[2] = trans.transform.rotation.z
            rotation[3] = trans.transform.rotation.w

            tf_matrix = transformer.fromTranslationRotation(translation,rotation)

            pc2 = ros_numpy.point_cloud2.pointcloud2_to_array(data)
            
            N = pc2['x'].shape[0]# * pc2['x'].shape[1]
            pc3 = np.zeros([N,4],np.float32)

            pc3[...,0]=pc2['x'].flatten()
            pc3[...,1]=pc2['y'].flatten()
            pc3[...,2]=pc2['z'].flatten()
            pc3[...,3]=pc2['intensity'].flatten()

            voxel_mapper.Process_radar_pointcloud(pc3,odom_data,tf_matrix)
            new_data = True
            
            print("     radar rate = " + str(1.0 / (time.time() - scan_time)))
            #print("processed pc")
        #except:
         #   print("bad tf")
    pass

height_diff_mean = []
height_diff_dev = []
height_diff_t = []
def main():
    global voxel_mapper, listener, transformer, tfBuffer, lidar_debug_pub, new_data, height_diff_mean, height_diff_dev, height_diff_t, lidar_time

    printed = False

    rospy.init_node('voxel_mapping', anonymous=True)
    print("start")

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    transformer = tf.TransformerROS()

    xy_resolution = 0.80
    z_resolution = 0.4
    width = 256
    height = 64
    buffer_size = 4
    min_point_distance = 2.0
    positive_obstacle_threshold = 0.75 #0.50 lidar
    negative_obstacle_threshold = 0.5
    density_threshold = 50
    slope_obsacle_threshold = 1.0
    min_roughness = -10
    max_roughness = 0
    robot_height = 2.0
    robot_radius = 4.0
    ground_to_lidar_height = 1.0
    radar_ground_density_threshold= 19750
    radar_obs_density_threshold= 23500
    
    voxel_mapper = gvom.Gvom(xy_resolution,z_resolution,width,height,buffer_size,min_point_distance,positive_obstacle_threshold,
    negative_obstacle_threshold,slope_obsacle_threshold,robot_height,robot_radius,ground_to_lidar_height,2,2,radar_ground_density_threshold,radar_obs_density_threshold)

    rospy.Subscriber("/os_cloud_node/points", PointCloud2, lidar_callback,queue_size=1)
    rospy.Subscriber("/radar_to_pt_cloud_2/radar_pointcloud", PointCloud2, radar_callback,queue_size=1)
    rospy.Subscriber("/robot/dlo/odom_node/odom", Odometry, odom_callback,queue_size=1)
    
    ns = "/sim"

    s_obstacle_map_pub = rospy.Publisher(ns+"/local_planning/map/soft_obstacle",OccupancyGrid,queue_size = 1)
    p_obstacle_map_pub = rospy.Publisher(ns+"/local_planning/map/positive_obstacle",OccupancyGrid,queue_size = 1)
    n_obstacle_map_pub = rospy.Publisher(ns+"/local_planning/map/negative_obstacle",OccupancyGrid,queue_size = 1)
    h_obstacle_map_pub = rospy.Publisher(ns+"/local_planning/map/hard_obstacle",OccupancyGrid,queue_size = 1)
    g_certainty_pub = rospy.Publisher(ns+"/local_planning/map/ground_certainty",OccupancyGrid,queue_size = 1)
    a_certainty_pub = rospy.Publisher(ns+"/local_planning/map/all_ground_certainty",OccupancyGrid,queue_size = 1)
    r_map_pub = rospy.Publisher(ns+"/local_planning/map/roughness",OccupancyGrid,queue_size = 1)

    p_obstacle_r_map_pub = rospy.Publisher(ns+"/local_planning/map/radar/positive_obstacle",OccupancyGrid,queue_size = 1)
    visability_r_map_pub = rospy.Publisher(ns+"/local_planning/map/radar/visability",OccupancyGrid,queue_size = 1)
    slope_r_map_pub = rospy.Publisher(ns+"/local_planning/map/radar/slope",OccupancyGrid,queue_size = 1)
    
    lidar_debug_pub = rospy.Publisher(ns+"/local_planning/voxel/debug/lidar",PointCloud2,queue_size = 1)
    voxel_debug_pub = rospy.Publisher(ns+"/local_planning/voxel/debug/voxel",PointCloud2,queue_size = 1)
    radar_debug_pub = rospy.Publisher(ns+"/local_planning/voxel/debug/radar",PointCloud2,queue_size = 1)
    radar_debug_obs_pub = rospy.Publisher(ns+"/local_planning/voxel/debug/radar_obs",PointCloud2,queue_size = 1)
    voxel_hm_debug_pub = rospy.Publisher(ns+"/local_planning/voxel/debug/height_map",PointCloud2,queue_size = 1)
    voxel_radar_hm_debug_pub = rospy.Publisher(ns+"/local_planning/voxel/debug/radar_height_map",PointCloud2,queue_size = 1)
    voxel_inf_hm_debug_pub = rospy.Publisher(ns+"/local_planning/voxel/debug/inferred_height_map",PointCloud2,queue_size = 1)
    diff_hm_debug_pub  = rospy.Publisher(ns+"/local_planning/voxel/debug/diff_height_map",PointCloud2,queue_size = 1)


    start_t = None

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if(not new_data):
            if(not printed):
                print("No new data")
                printed = True
            rate.sleep()
            continue
        new_data = False
        printed = False

        print("combined maps")
        map_time = time.time()
        print("combine lidar")
        map_data = voxel_mapper.combine_maps()
        print("combine radar")
        radar_data = voxel_mapper.combine_radar_maps()
        print("done combine")

        radar_voxel_hm = None
        lidar_voxel_hm = None
        
        if not radar_data is None:
            print("radar map data")
            map_origin = radar_data[0]
            obs_map = radar_data[1]
            #neg_map = radar_data[2]
            #rough_map = radar_data[3]
            cert_map = radar_data[2]
            #print(obs_map)
            #print(obs_map.shape)
            #print(map_origin)

            radar_voxel = voxel_mapper.make_debug_radar_map()
            if not (radar_voxel is None):
                radar_voxel = np.core.records.fromarrays([radar_voxel[:,0],radar_voxel[:,1],radar_voxel[:,2],radar_voxel[:,3],radar_voxel[:,4],radar_voxel[:,5],radar_voxel[:,6],radar_voxel[:,7],radar_voxel[:,8]],names='x,y,z,count,intensity,grad_x,grad_y,grad_z,grad_xy')
                radar_debug_obs_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(radar_voxel,rospy.Time.now(),odom_frame))

            radar_voxel = voxel_mapper.make_debug_radar_map(15000)
            if not (radar_voxel is None):
                radar_voxel = np.core.records.fromarrays([radar_voxel[:,0],radar_voxel[:,1],radar_voxel[:,2],radar_voxel[:,3],radar_voxel[:,4],radar_voxel[:,5],radar_voxel[:,6],radar_voxel[:,7],radar_voxel[:,8]],names='x,y,z,count,intensity,grad_x,grad_y,grad_z,grad_xy')
                radar_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(radar_voxel,rospy.Time.now(),odom_frame))

            radar_voxel_hm = voxel_mapper.make_debug_radar_height_map()
            if not (radar_voxel_hm is None):

                voxel_hm = np.core.records.fromarrays([radar_voxel_hm[:,0],radar_voxel_hm[:,1],radar_voxel_hm[:,2],radar_voxel_hm[:,3],radar_voxel_hm[:,4],radar_voxel_hm[:,5],radar_voxel_hm[:,6],obs_map.flatten('F')],names='x,y,z,roughness,slope_x,slope_y,slope,obstacles')
                voxel_radar_hm_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(voxel_hm,rospy.Time.now(),odom_frame))


            out_map = OccupancyGrid()
            out_map.header.stamp = rospy.Time.now()
            out_map.header.frame_id = odom_frame
            out_map.info.resolution = xy_resolution
            out_map.info.width = width
            out_map.info.height = width
            out_map.info.origin.orientation.x = 0
            out_map.info.origin.orientation.y = 0
            out_map.info.origin.orientation.z = 0
            out_map.info.origin.orientation.w = 1
            out_map.info.origin.position.x = map_origin[0]
            out_map.info.origin.position.y = map_origin[1]
            out_map.info.origin.position.z = 0

            out_map.data = np.reshape(obs_map,-1,order='F').astype(np.int8)
            
            p_obstacle_r_map_pub.publish(out_map)

            out_map.data = np.reshape(cert_map*100,-1,order='F').astype(np.int8)

            visability_r_map_pub.publish(out_map)

            if not (voxel_hm is None):
                out_map.data = np.reshape(100.0 * voxel_hm.slope / np.pi,-1,order='F').astype(np.int8)
                
                slope_r_map_pub.publish(out_map)


        if not map_data is None:
            print("lidar map data")
            map_origin = map_data[0]
            obs_map = map_data[1]
            neg_map = map_data[2]
            rough_map = map_data[3]
            cert_map = map_data[4]
            #print(obs_map)
            #print(obs_map.shape)
            #print(map_origin)

            out_map = OccupancyGrid()
            out_map.header.stamp = rospy.Time.now()
            out_map.header.frame_id = odom_frame
            out_map.info.resolution = xy_resolution
            out_map.info.width = width
            out_map.info.height = width
            out_map.info.origin.orientation.x = 0
            out_map.info.origin.orientation.y = 0
            out_map.info.origin.orientation.z = 0
            out_map.info.origin.orientation.w = 1
            out_map.info.origin.position.x = map_origin[0]
            out_map.info.origin.position.y = map_origin[1]
            out_map.info.origin.position.z = 0

            out_map.data = np.reshape(np.maximum(100 * (obs_map > density_threshold),neg_map),-1,order='F').astype(np.int8)
            
            h_obstacle_map_pub.publish(out_map)

            out_map.data = np.reshape(100 * (obs_map <= density_threshold) * (obs_map > 0),-1,order='F').astype(np.int8)

            s_obstacle_map_pub.publish(out_map)

            out_map.data = np.reshape(cert_map*100,-1,order='F').astype(np.int8)

            g_certainty_pub.publish(out_map)
            a_certainty_pub.publish(out_map)

            out_map.data = np.reshape(neg_map,-1,order='F').astype(np.int8)

            n_obstacle_map_pub.publish(out_map)

            rough_map = ((np.maximum(np.minimum(rough_map,max_roughness),min_roughness) + min_roughness) / (max_roughness - min_roughness)) * 100
            out_map.data = np.reshape(rough_map,-1,order='F').astype(np.int8)

            r_map_pub.publish(out_map)

            voxel_pc = voxel_mapper.make_debug_voxel_map()
            if not (voxel_pc is None):
                voxel_pc = np.core.records.fromarrays([voxel_pc[:,0],voxel_pc[:,1],voxel_pc[:,2],voxel_pc[:,3],voxel_pc[:,4],voxel_pc[:,5],voxel_pc[:,6],voxel_pc[:,7]],names='x,y,z,solid factor,count,eigen1,eigen2,eigen3')

                voxel_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(voxel_pc,rospy.Time.now(),odom_frame))

            lidar_voxel_hm = voxel_mapper.make_debug_height_map()
            if not (lidar_voxel_hm is None):
                #plt.figure()
                #plt.scatter(obs_map.flatten('F'),voxel_hm[:,6])
                #plt.xlabel("obstacle density")
                #plt.ylabel("slope")
                #plt.show()

                voxel_hm = np.core.records.fromarrays([lidar_voxel_hm[:,0],lidar_voxel_hm[:,1],lidar_voxel_hm[:,2],lidar_voxel_hm[:,3],lidar_voxel_hm[:,4],lidar_voxel_hm[:,5],lidar_voxel_hm[:,6],obs_map.flatten('F')],names='x,y,z,roughness,slope_x,slope_y,slope,obstacles')

                voxel_hm_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(voxel_hm,rospy.Time.now(),odom_frame))
        
            voxel_inf_hm = voxel_mapper.make_debug_inferred_height_map()
            if not (voxel_inf_hm is None):
                voxel_inf_hm = np.core.records.fromarrays([voxel_inf_hm[:,0],voxel_inf_hm[:,1],voxel_inf_hm[:,2]],names='x,y,z')

                voxel_inf_hm_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(voxel_inf_hm,rospy.Time.now(),odom_frame))


            print("map rate = " + str(1.0 / (time.time() - map_time)))
            
        else:
            print("no map data")


        if (lidar_voxel_hm is not None) and (radar_voxel_hm is not None): #compare the lidar and radar height maps
            radar_hm = radar_voxel_hm.reshape((width,width,-1),order='F')
            lidar_hm = lidar_voxel_hm.reshape((width,width,-1),order='F')
            
            offset = np.rint((lidar_hm[0,0,:] - radar_hm[0,0,:])/xy_resolution)
            
            print(offset)

            difference_map = -10.0 * np.ones((width,width))
            

            shift_start = time.time()
            #radar_hm = np.roll(radar_hm,(int(-offset[0]),int(-offset[1])))
            shifted_lidar_hm = np.empty((width,width))
            shifted_lidar_hm[:] = np.nan

            shifted_lidar_hm[max(0,int(offset[0])):min(width,width + int(offset[0])),
                             max(0,int(offset[1])):min(width,width + int(offset[1]))] = lidar_hm[max(0,int(-offset[0])):min(width,width + int(-offset[0])),
                                                 max(0,int(-offset[1])):min(width,width + int(-offset[1])),2]

            #difference_map = np.abs(shifted_lidar_hm - radar_hm[:,:,2])
            difference_map = shifted_lidar_hm - radar_hm[:,:,2]
            #difference_map[np.isnan(shifted_lidar_hm)] = - 100.0

            print(difference_map[~np.isnan(difference_map)])
            print(difference_map[(~np.isnan(difference_map)) * (difference_map != -100)])

            print("shift rate = " + str(1.0 / (time.time() - shift_start)))
            diff_hm = np.core.records.fromarrays([radar_voxel_hm[:,0],radar_voxel_hm[:,1],radar_voxel_hm[:,2],difference_map.flatten('F')],names='x,y,z,diff')

            diff_hm_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(diff_hm,rospy.Time.now(),odom_frame))
                    
            mean_diff = np.mean(difference_map[(~np.isnan(difference_map)) * (~np.isnan(shifted_lidar_hm))])
            std_diff = np.std(difference_map[(~np.isnan(difference_map)) * (~np.isnan(shifted_lidar_hm))])

            height_diff_mean.append(mean_diff)
            height_diff_dev.append(std_diff)

            dist_map_x = radar_hm[:,:,0] - odom_data[0]
            dist_map_y = radar_hm[:,:,1] - odom_data[1]
            dist_map = np.sqrt(dist_map_x*dist_map_x + dist_map_y*dist_map_y)

            #plt.scatter(dist_map[~np.isnan(shifted_lidar_hm)],difference_map[~np.isnan(shifted_lidar_hm)])
            #plt.show()


            #if(start_t is None):
            #    start_t = rospy.Time.now()

            height_diff_t.append(lidar_time.to_sec())

        #rate.sleep()


    
    rospy.on_shutdown(shutdown)


def shutdown():
    print("end")
    with open('height_diff_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(height_diff_t)):
            writer.writerow([height_diff_t[i],height_diff_mean[i],height_diff_dev[i]])
    plt.figure()
    plt.plot(height_diff_t,height_diff_mean)
    plt.figure()
    plt.plot(height_diff_t,height_diff_dev)
    plt.show()

if __name__ == '__main__':
    main()
    #rospy.on_shutdown(shutdown)
