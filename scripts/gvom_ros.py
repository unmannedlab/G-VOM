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

voxel_mapper = None
listener = None
transformer = None
odom_data = None
tfBuffer = None
lidar_debug_pub = None
odom_frame = "/camera_init"
new_data = False

def odom_callback(data):
    global odom_data
    odom_data = (data.pose.pose.position.x,data.pose.pose.position.y,data.pose.pose.position.z)
    #print("got odom")

def lidar_callback(data):
    global new_data
    print("got scan")
    

   
    if odom_data == None:
        print("no odom")
        return

    if not voxel_mapper is None:
        #try:
            scan_time = time.time()
            lidar_frame = data.header.frame_id
            #listener.waitForTransform(odom_frame,lidar_frame,   data.header.stamp,rospy.Duration(0.01))
            trans = tfBuffer.lookup_transform(odom_frame,lidar_frame,   data.header.stamp,rospy.Duration(0.01))
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
            #print(tf_matrix)
            #print("x offset = " + str(tf_matrix[0,3]))
            
            pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
            #scan_center = (trans.transform.translation.x,trans.transform.translation.y,trans.transform.translation.z)
            voxel_mapper.Process_pointcloud(pc,odom_data,tf_matrix)
            new_data = True
            
            #transformed_pc = voxel_mapper.Process_pointcloud(pc,scan_center,tf_matrix).astype(np.float32)
            #print(transformed_pc.dtype)

            #transformed_pc = np.core.records.fromarrays([transformed_pc[:,0],transformed_pc[:,1],transformed_pc[:,2]],names='x,y,z')
            #print(transformed_pc)

            #debug_msg = ros_numpy.point_cloud2.array_to_pointcloud2(transformed_pc,data.header.stamp,odom_frame)
            #lidar_debug_pub.publish(debug_msg)
            print("     pointcloud rate = " + str(1.0 / (time.time() - scan_time)))
            #print("processed pc")
        #except:
         #   print("bad tf")
    pass

def main():
    global voxel_mapper, listener, transformer, tfBuffer, lidar_debug_pub, new_data

    printed = False

    rospy.init_node('voxel_mapping', anonymous=True)
    print("start")

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    transformer = tf.TransformerROS()

    xy_resolution = 0.40
    z_resolution = 0.2
    width = 256
    height = 64
    buffer_size = 4
    min_point_distance = 1.0
    positive_obstacle_threshold = 0.50
    negative_obstacle_threshold = 0.5
    density_threshold = 50
    slope_obsacle_threshold = 0.3
    min_roughness = -10
    max_roughness = 0
    robot_height = 2.0
    robot_radius = 4.0
    ground_to_lidar_height = 1.0
    
    voxel_mapper = gvom.Gvom(xy_resolution,z_resolution,width,height,buffer_size,min_point_distance,positive_obstacle_threshold,
    negative_obstacle_threshold,slope_obsacle_threshold,robot_height,robot_radius,ground_to_lidar_height)

    rospy.Subscriber("/velodyne_cloud_registered", PointCloud2, lidar_callback,queue_size=1)
    rospy.Subscriber("/aft_mapped_to_init_high_frec", Odometry, odom_callback,queue_size=1)
    
    s_obstacle_map_pub = rospy.Publisher("/local_planning/map/soft_obstacle",OccupancyGrid,queue_size = 1)
    p_obstacle_map_pub = rospy.Publisher("/local_planning/map/positive_obstacle",OccupancyGrid,queue_size = 1)
    n_obstacle_map_pub = rospy.Publisher("/local_planning/map/negative_obstacle",OccupancyGrid,queue_size = 1)
    h_obstacle_map_pub = rospy.Publisher("/local_planning/map/hard_obstacle",OccupancyGrid,queue_size = 1)
    g_certainty_pub = rospy.Publisher("/local_planning/map/ground_certainty",OccupancyGrid,queue_size = 1)
    a_certainty_pub = rospy.Publisher("/local_planning/map/all_ground_certainty",OccupancyGrid,queue_size = 1)
    r_map_pub = rospy.Publisher("/local_planning/map/roughness",OccupancyGrid,queue_size = 1)
    
    lidar_debug_pub = rospy.Publisher('/local_planning/voxel/debug/lidar',PointCloud2,queue_size = 1)
    voxel_debug_pub = rospy.Publisher('/local_planning/voxel/debug/voxel',PointCloud2,queue_size = 1)
    voxel_hm_debug_pub = rospy.Publisher('/local_planning/voxel/debug/height_map',PointCloud2,queue_size = 1)
    voxel_inf_hm_debug_pub = rospy.Publisher('/local_planning/voxel/debug/inferred_height_map',PointCloud2,queue_size = 1)


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
        map_data = voxel_mapper.combine_maps()
        

        if not map_data is None:
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
                voxel_pc = np.core.records.fromarrays([voxel_pc[:,0],voxel_pc[:,1],voxel_pc[:,2],voxel_pc[:,3],voxel_pc[:,4]],names='x,y,z,solid factor,count')

                voxel_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(voxel_pc,rospy.Time.now(),odom_frame))

            voxel_hm = voxel_mapper.make_debug_height_map()
            if not (voxel_hm is None):
                #plt.figure()
                #plt.scatter(obs_map.flatten('F'),voxel_hm[:,6])
                #plt.xlabel("obstacle density")
                #plt.ylabel("slope")
                #plt.show()

                voxel_hm = np.core.records.fromarrays([voxel_hm[:,0],voxel_hm[:,1],voxel_hm[:,2],voxel_hm[:,3],voxel_hm[:,4],voxel_hm[:,5],voxel_hm[:,6],obs_map.flatten('F')],names='x,y,z,roughness,slope_x,slope_y,slope,obstacles')

                voxel_hm_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(voxel_hm,rospy.Time.now(),odom_frame))
        
            voxel_inf_hm = voxel_mapper.make_debug_inferred_height_map()
            if not (voxel_inf_hm is None):
                voxel_inf_hm = np.core.records.fromarrays([voxel_inf_hm[:,0],voxel_inf_hm[:,1],voxel_inf_hm[:,2]],names='x,y,z')

                voxel_inf_hm_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(voxel_inf_hm,rospy.Time.now(),odom_frame))

            print("map rate = " + str(1.0 / (time.time() - map_time)))
            
        else:
            print("no map data")




        rate.sleep()
    print("end")


if __name__ == '__main__':
    main()
