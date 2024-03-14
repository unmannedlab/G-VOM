import rclpy
import numpy as np
from rclpy.node import Node
from .gvom import Gvom
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid, Odometry
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from scipy.spatial.transform import Rotation as R
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros

class VoxelMapper(Node):
    def __init__(self):
        super().__init__('voxel_mapper')

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer, self)

        self.freq = 10 # Hz

        self.odom_sub = self.subscription = self.create_subscription(Odometry,'odom',self.odom_callback,1)
        self.lidar_sub = self.subscription = self.create_subscription(PointCloud2,'lidar_points',self.lidar_callback,1)
        self.radar_sub = self.subscription = self.create_subscription(PointCloud2,'radar_points',self.radar_callback,1)

        # self.s_obstacle_map_pub = self.create_publisher(OccupancyGrid, "~soft_obstacle_map")
        # self.p_obstacle_map_pub = self.create_publisher(OccupancyGrid,"~positive_obstacle_map")
        # self.n_obstacle_map_pub = self.create_publisher(OccupancyGrid,"~negative_obstacle_map")
        # self.h_obstacle_map_pub = self.create_publisher(OccupancyGrid,"~hard_obstacle_map")
        # self.g_certainty_pub = self.create_publisher(OccupancyGrid,"~ground_certainty_map")
        # self.a_certainty_pub = self.create_publisher(OccupancyGrid,"~all_ground_certainty_map")
        # self.r_map_pub = self.create_publisher(OccupancyGrid,"~roughness_map")

        self.timer = self.create_timer(1.0/self.freq, self.map_pub_callback)
        
        self.lidar_debug_pub = self.create_publisher(PointCloud2, 'debug/lidar', rclpy.qos.qos_profile_default)
        self.voxel_debug_pub = self.create_publisher(PointCloud2, 'debug/voxel', rclpy.qos.qos_profile_default)
        self.voxel_hm_debug_pub = self.create_publisher(PointCloud2, 'debug/height_map', rclpy.qos.qos_profile_default)
        self.voxel_inf_hm_debug_pub = self.create_publisher(PointCloud2, 'debug/inferred_height_map', rclpy.qos.qos_profile_default)

        self.gridmap_pub = self.create_publisher(GridMap,'gridmap', rclpy.qos.qos_profile_default)

        # Declare parameters
        self.declare_parameter('odom_frame', 'warty/base_link')
        self.declare_parameter('xy_resolution', 0.40)
        self.declare_parameter('z_resolution', 0.2)
        self.declare_parameter('width', 256)
        self.declare_parameter('height', 64)
        self.declare_parameter('buffer_size', 4)
        self.declare_parameter('min_point_distance', 1.0)
        self.declare_parameter('positive_obstacle_threshold', 0.50)
        self.declare_parameter('negative_obstacle_threshold', 0.5)
        self.declare_parameter('density_threshold', 50)
        self.declare_parameter('slope_obsacle_threshold', 0.3)
        self.declare_parameter('robot_height', 2.0)
        self.declare_parameter('robot_radius', 4.0)
        self.declare_parameter('ground_to_lidar_height', 1.0)
        self.declare_parameter('freq', 5.0)  # Hz
        self.declare_parameter('xy_eigen_dist', 1)
        self.declare_parameter('z_eigen_dist', 1)
        self.declare_parameter('radar_positive_obstacle_threshold', 0.50)
        self.declare_parameter('radar_ground_density_threshold', 18000)
        self.declare_parameter('radar_obs_density_threshold', 24500)

        # Get parameters
        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value
        self.xy_resolution = self.get_parameter('xy_resolution').get_parameter_value().double_value
        self.z_resolution = self.get_parameter('z_resolution').get_parameter_value().double_value
        self.width = self.get_parameter('width').get_parameter_value().integer_value
        self.height = self.get_parameter('height').get_parameter_value().integer_value
        self.buffer_size = self.get_parameter('buffer_size').get_parameter_value().integer_value
        self.min_point_distance = self.get_parameter('min_point_distance').get_parameter_value().double_value
        self.positive_obstacle_threshold = self.get_parameter('positive_obstacle_threshold').get_parameter_value().double_value
        self.negative_obstacle_threshold = self.get_parameter('negative_obstacle_threshold').get_parameter_value().double_value
        self.density_threshold = self.get_parameter('density_threshold').get_parameter_value().integer_value
        self.slope_obsacle_threshold = self.get_parameter('slope_obsacle_threshold').get_parameter_value().double_value
        self.robot_height = self.get_parameter('robot_height').get_parameter_value().double_value
        self.robot_radius = self.get_parameter('robot_radius').get_parameter_value().double_value
        self.ground_to_lidar_height = self.get_parameter('ground_to_lidar_height').get_parameter_value().double_value
        self.freq = self.get_parameter('freq').get_parameter_value().double_value
        self.xy_eigen_dist = self.get_parameter('xy_eigen_dist').get_parameter_value().integer_value
        self.z_eigen_dist = self.get_parameter('z_eigen_dist').get_parameter_value().integer_value
        self.radar_positive_obstacle_threshold = self.get_parameter('radar_positive_obstacle_threshold').get_parameter_value().double_value
        self.radar_ground_density_threshold = self.get_parameter('radar_ground_density_threshold').get_parameter_value().integer_value
        self.radar_obs_density_threshold = self.get_parameter('radar_obs_density_threshold').get_parameter_value().integer_value

        self.gvom = Gvom(
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
            self.z_eigen_dist,
            self.radar_positive_obstacle_threshold,
            self.radar_ground_density_threshold, 
            self.radar_obs_density_threshold
        )


    def map_pub_callback(self):


        map_data = self.gvom.combine_maps()

        if map_data is None:
            self.get_logger().warning("map_data is None. returning.")
            
            return

        map_origin = map_data[0] # the location of the 0,0 corner of the map in the odom frame
        obs_map = map_data[1]
        neg_map = map_data[2]
        rough_map = map_data[3]
        cert_map = map_data[4]
        x_slope_map = map_data[5]
        y_slope_map = map_data[6]
        height_map = map_data[7]

        output_map = GridMap()
        output_map.info.resolution = self.xy_resolution
        output_map.info.length_x = self.xy_resolution * self.width
        output_map.info.length_y = self.xy_resolution * self.width
        output_map.info.pose.orientation.x = 0.0
        output_map.info.pose.orientation.y = 0.0
        output_map.info.pose.orientation.z = 0.0
        output_map.info.pose.orientation.w = 1.0
        output_map.info.pose.position.x = map_origin[0] + 0.5 * self.xy_resolution * self.width # GridMap sets the map origin in the center of the map so we need to add an offset
        output_map.info.pose.position.x = map_origin[1] + 0.5 * self.xy_resolution * self.width
        output_map.header.stamp = self.get_clock().now().to_msg()
        output_map.header.frame_id = self.odom_frame


        output_map.layers = ["positve obstacles","negative obstacles","roughness","certainty","slope x","slope y","elevation"]
        output_map.data.append(np_to_Float32MultiArray(obs_map))
        output_map.data.append(np_to_Float32MultiArray(neg_map))
        output_map.data.append(np_to_Float32MultiArray(rough_map))
        output_map.data.append(np_to_Float32MultiArray(cert_map))
        output_map.data.append(np_to_Float32MultiArray(x_slope_map))
        output_map.data.append(np_to_Float32MultiArray(y_slope_map))
        output_map.data.append(np_to_Float32MultiArray(height_map))

        self.gridmap_pub.publish(output_map)
        self.get_logger().info("published gridmap.")
        ### Debug maps

        # Voxel map
        voxel_pc = self.gvom.make_debug_voxel_map()
        if voxel_pc is not None:

            fields = self.fields_from_names(['x','y','z','solid factor','count','eigen_line','eigen_surface','eigen_point'])
            points = [list(point) for point in zip(voxel_pc[:,0], voxel_pc[:,1], voxel_pc[:,2], voxel_pc[:,3], voxel_pc[:,4], voxel_pc[:,5], voxel_pc[:,6], voxel_pc[:,7])]
            msg = pc2.create_cloud(output_map.header,fields,points)
            self.voxel_debug_pub.publish(msg)
            self.get_logger().info("published voxel debug.")

        # Voxel height map
        voxel_hm = self.gvom.make_debug_height_map()
        if voxel_hm is not None:
            fields = self.fields_from_names(['x','y','z','roughness','slope_x','slope_y','abs_slope','obstacles'])
            points = [list(point) for point in zip(voxel_pc[:,0], voxel_pc[:,1], voxel_pc[:,2], voxel_pc[:,3], voxel_pc[:,4], voxel_pc[:,5], voxel_pc[:,6], obs_map.flatten('F'))]
            msg = pc2.create_cloud(output_map.header,fields,points)            
            self.voxel_hm_debug_pub.publish(msg)
            self.get_logger().info("published voxel height map debug.")
    
        # Inferred height map
        voxel_inf_hm = self.gvom.make_debug_inferred_height_map()
        if voxel_inf_hm is not None:
            fields = self.fields_from_names(['x','y','z'])
            points = [list(point) for point in zip(voxel_inf_hm[:,0],voxel_inf_hm[:,1],voxel_inf_hm[:,2])]
            msg = pc2.create_cloud(output_map.header,fields,points)            
            self.voxel_inf_hm_debug_pub.publish(msg)
            self.get_logger().info("published voxel inferred height map debug.")

        pass

    def lidar_callback(self,data):
        #if self.odom_data == None:
        #    self.logwarn("no odom")
        #    return

        #odom_data = self.odom_data

        self.get_logger().info("looking up " + self.odom_frame + " and " + data.header.frame_id)
        try:
        
            lidar_frame = data.header.frame_id
            trans = self.tfBuffer.lookup_transform(self.odom_frame, lidar_frame, rclpy.time.Time.from_msg(data.header.stamp), rclpy.duration.Duration(seconds=1))
            self.get_logger().info("got tf")
            translation = np.zeros([3])
            translation[0] = trans.transform.translation.x
            translation[1] = trans.transform.translation.y
            translation[2] = trans.transform.translation.z

            rotation = np.zeros([4])
            rotation[0] = trans.transform.rotation.x
            rotation[1] = trans.transform.rotation.y
            rotation[2] = trans.transform.rotation.z
            rotation[3] = trans.transform.rotation.w

            r = R.from_quat(rotation)
            rotation_matrix = r.as_matrix()

            tf_matrix = np.eye(4)
            tf_matrix[:3, :3] = rotation_matrix
            tf_matrix[:3, 3] = translation 
            
            structured_array = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))
            pc = np.stack([structured_array['x'], structured_array['y'], structured_array['z']], axis=-1)
            print(pc.shape)

            self.gvom.Process_pointcloud(pc, translation, tf_matrix)


        except Exception as e:
            self.get_logger().warn(f"{e}")

    def radar_callback(self,data):
        pass

    def odom_callback(self,data):
        pass

    def fields_from_names(self,names):
        fields = []

        current_offset = 0
        for name1 in names:
            fields.append(PointField(name=name1,offset=current_offset,datatype=PointField.FLOAT32, count=1))
            current_offset+=4

        return fields
        

def main(args=None):
    rclpy.init(args=args)
    voxel_mapper = VoxelMapper()
    rclpy.spin(voxel_mapper)


def np_to_Float32MultiArray(np_array_in):
    np_array = np_array_in.astype(np.float32)
    # Create a Float32MultiArray message
    float32_multi_array_msg = Float32MultiArray()
    
    # Flatten the NumPy array and assign it to the message's data field
    float32_multi_array_msg.data = np_array.ravel()
    
    # Set the layout of the MultiArray
    float32_multi_array_msg.layout.dim = []
    for dim in range(np_array.ndim):
        # Create a MultiArrayDimension for each dimension in the NumPy array
        dimension = MultiArrayDimension()
        dimension.label = f'dim{dim}'
        dimension.size = np_array.shape[dim]
        dimension.stride = np_array.strides[dim] // np_array.itemsize
        float32_multi_array_msg.layout.dim.append(dimension)
    
    float32_multi_array_msg.layout.data_offset = 0
    
    return float32_multi_array_msg

if __name__ == '__main__':
    main()