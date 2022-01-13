#!/usr/bin/env python
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


class VoxelMapper:
    def __init__(self):

        self.odom_data = None

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        self.tf_transformer = tf.TransformerROS()

        self.odom_frame = rospy.get_param("~odom_frame", "/camera_init")
        self.xy_resolution = rospy.get_param("~xy_resolution", 0.40)
        self.z_resolution = rospy.get_param("~z_resolution", 0.2)
        self.width = rospy.get_param("~width", 256)
        self.height = rospy.get_param("~height", 64)
        self.buffer_size = rospy.get_param("~buffer_size", 4)
        self.min_point_distance = rospy.get_param("~min_point_distance", 1.0)
        self.positive_obstacle_threshold = rospy.get_param("~positive_obstacle_threshold", 0.50)
        self.negative_obstacle_threshold = rospy.get_param("~negative_obstacle_threshold", 0.5)
        self.density_threshold = rospy.get_param("~density_threshold", 50)
        self.slope_obsacle_threshold = rospy.get_param("~slope_obsacle_threshold", 0.3)
        self.min_roughness = rospy.get_param("~min_roughness", -10)
        self.max_roughness = rospy.get_param("~max_roughness", 0)
        self.robot_height = rospy.get_param("~robot_height", 2.0)
        self.robot_radius = rospy.get_param("~robot_radius", 4.0)
        self.ground_to_lidar_height = rospy.get_param("~ground_to_lidar_height", 1.0)
        self.freq = rospy.get_param("~freq", 10.) # Hz
        self.xy_eigen_dist = rospy.get_param("~xy_eigen_dist",1)
        self.z_eigen_dist = rospy.get_param("~z_eigen_dist",1)
        
        
        self.voxel_mapper = gvom.Gvom(
            self.xy_resolution,
            self.z_resolution,
            self.width,
            self.height,
            self.buffer_size,
            self.min_point_distance,
            self.positive_obstacle_threshold,
            self.negative_obstacle_threshold,
            self.slope_obsacle_threshold,
            self.robot_height,
            self.robot_radius,
            self.ground_to_lidar_height,
            self.xy_eigen_dist,
            self.z_eigen_dist
        )

        self.sub_cloud = rospy.Subscriber("~cloud", PointCloud2, self.cb_lidar,queue_size=1)
        self.sub_odom = rospy.Subscriber("~odom", Odometry, self.cb_odom,queue_size=1)
        
        self.s_obstacle_map_pub = rospy.Publisher("~soft_obstacle_map", OccupancyGrid, queue_size = 1)
        self.p_obstacle_map_pub = rospy.Publisher("~positive_obstacle_map", OccupancyGrid, queue_size = 1)
        self.n_obstacle_map_pub = rospy.Publisher("~negative_obstacle_map", OccupancyGrid, queue_size = 1)
        self.h_obstacle_map_pub = rospy.Publisher("~hard_obstacle_map", OccupancyGrid, queue_size = 1)
        self.g_certainty_pub = rospy.Publisher("~ground_certainty_map", OccupancyGrid, queue_size = 1)
        self.a_certainty_pub = rospy.Publisher("~all_ground_certainty_map", OccupancyGrid, queue_size = 1)
        self.r_map_pub = rospy.Publisher("~roughness_map", OccupancyGrid, queue_size = 1)

        self.timer = rospy.Timer(rospy.Duration(1./self.freq), self.cb_timer)
        
        self.lidar_debug_pub = rospy.Publisher('~debug/lidar', PointCloud2, queue_size = 1)
        self.voxel_debug_pub = rospy.Publisher('~debug/voxel', PointCloud2, queue_size = 1)
        self.voxel_hm_debug_pub = rospy.Publisher('~debug/height_map', PointCloud2, queue_size = 1)
        self.voxel_inf_hm_debug_pub = rospy.Publisher('~debug/inferred_height_map', PointCloud2, queue_size = 1)

    def cb_odom(self, data):
        self.odom_data = (data.pose.pose.position.x,data.pose.pose.position.y,data.pose.pose.position.z)

    def cb_lidar(self, data):
        # rospy.loginfo("got scan")
   
        if self.odom_data == None:
            print("no odom")
            return

        odom_data = self.odom_data

        scan_time = time.time()
        lidar_frame = data.header.frame_id
        trans = self.tfBuffer.lookup_transform(self.odom_frame, lidar_frame, data.header.stamp,rospy.Duration(1))

        translation = np.zeros([3])
        translation[0] = trans.transform.translation.x
        translation[1] = trans.transform.translation.y
        translation[2] = trans.transform.translation.z

        rotation = np.zeros([4])
        rotation[0] = trans.transform.rotation.x
        rotation[1] = trans.transform.rotation.y
        rotation[2] = trans.transform.rotation.z
        rotation[3] = trans.transform.rotation.w

        tf_matrix = self.tf_transformer.fromTranslationRotation(translation,rotation)
        
        pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
        self.voxel_mapper.Process_pointcloud(pc, odom_data, tf_matrix)

        # print("     pointcloud rate = " + str(1.0 / (time.time() - scan_time)))

    def cb_timer(self, event):

        map_data = self.voxel_mapper.combine_maps()

        if map_data is None:
            rospy.loginfo("map_data is None. returning.")
            return

        map_origin = map_data[0]
        obs_map = map_data[1]
        neg_map = map_data[2]
        rough_map = map_data[3]
        cert_map = map_data[4]

        out_map = OccupancyGrid()
        out_map.header.stamp = rospy.Time.now()
        out_map.header.frame_id = self.odom_frame
        out_map.info.resolution = self.xy_resolution
        out_map.info.width = self.width
        out_map.info.height = self.width
        out_map.info.origin.orientation.x = 0
        out_map.info.origin.orientation.y = 0
        out_map.info.origin.orientation.z = 0
        out_map.info.origin.orientation.w = 1
        out_map.info.origin.position.x = map_origin[0]
        out_map.info.origin.position.y = map_origin[1]
        out_map.info.origin.position.z = 0

        # Hard obstacles
        out_map.data = np.reshape(np.maximum(100 * (obs_map > self.density_threshold), neg_map),-1,order='F').astype(np.int8)
        self.h_obstacle_map_pub.publish(out_map)
        rospy.loginfo("published hard obstacle map.")

        # Soft obstacles
        out_map.data = np.reshape(100 * (obs_map <= self.density_threshold) * (obs_map > 0),-1,order='F').astype(np.int8)
        self.s_obstacle_map_pub.publish(out_map)
        rospy.loginfo("published soft obstacle map.")

        # Ground certainty
        out_map.data = np.reshape(cert_map*100,-1,order='F').astype(np.int8)
        self.g_certainty_pub.publish(out_map)
        self.a_certainty_pub.publish(out_map)
        rospy.loginfo("published ground certainty maps.")

        # Negative obstacles
        out_map.data = np.reshape(neg_map,-1,order='F').astype(np.int8)
        self.n_obstacle_map_pub.publish(out_map)
        rospy.loginfo("published negative obstacle map.")

        # Roughness
        rough_map = ((np.maximum(np.minimum(rough_map, self.max_roughness), self.min_roughness) + self.min_roughness) / (self.max_roughness - self.min_roughness)) * 100
        out_map.data = np.reshape(rough_map,-1,order='F').astype(np.int8)
        self.r_map_pub.publish(out_map)
        rospy.loginfo("published roughness map.")

        ### Debug maps

        # Voxel map
        voxel_pc = self.voxel_mapper.make_debug_voxel_map()
        if voxel_pc is not None:
            voxel_pc = np.core.records.fromarrays([voxel_pc[:,0],voxel_pc[:,1],voxel_pc[:,2],voxel_pc[:,3],voxel_pc[:,4],voxel_pc[:,5],voxel_pc[:,6],voxel_pc[:,7]],names='x,y,z,solid factor,count,eigen_line,eigen_surface,eigen_point')
            self.voxel_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(voxel_pc, rospy.Time.now(), self.odom_frame))
            rospy.loginfo("published voxel debug.")

        # Voxel height map
        voxel_hm = self.voxel_mapper.make_debug_height_map()
        if voxel_hm is not None:
            voxel_hm = np.core.records.fromarrays([voxel_hm[:,0],voxel_hm[:,1],voxel_hm[:,2],voxel_hm[:,3],voxel_hm[:,4],voxel_hm[:,5],voxel_hm[:,6],obs_map.flatten('F')],names='x,y,z,roughness,slope_x,slope_y,slope,obstacles')
            self.voxel_hm_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(voxel_hm, rospy.Time.now(), self.odom_frame))
            rospy.loginfo("published voxel height map debug.")
    
        # Inferred height map
        voxel_inf_hm = self.voxel_mapper.make_debug_inferred_height_map()
        if voxel_inf_hm is not None:
            voxel_inf_hm = np.core.records.fromarrays([voxel_inf_hm[:,0],voxel_inf_hm[:,1],voxel_inf_hm[:,2]],names='x,y,z')
            self.voxel_inf_hm_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(voxel_inf_hm, rospy.Time.now(), self.odom_frame))
            rospy.loginfo("published voxel inferred height map debug.")
            
if __name__ == '__main__':
    rospy.init_node('voxel_mapping')

    node = VoxelMapper()

    while not rospy.is_shutdown():
        rospy.spin()

    rospy.on_shutdown(node.on_shutdown)
