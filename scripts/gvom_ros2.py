import rclpy
import numpy as np
from rclpy.node import Node
import gvom
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid

class VoxelMapper(Node):
    def __init__(self):
        super().__init__('voxel_mapper')

        self.freq = 10 # Hz

        self.s_obstacle_map_pub = self.create_publisher("~soft_obstacle_map", OccupancyGrid, queue_size = 1)
        self.p_obstacle_map_pub = self.create_publisher("~positive_obstacle_map", OccupancyGrid, queue_size = 1)
        self.n_obstacle_map_pub = self.create_publisher("~negative_obstacle_map", OccupancyGrid, queue_size = 1)
        self.h_obstacle_map_pub = self.create_publisher("~hard_obstacle_map", OccupancyGrid, queue_size = 1)
        self.g_certainty_pub = self.create_publisher("~ground_certainty_map", OccupancyGrid, queue_size = 1)
        self.a_certainty_pub = self.create_publisher("~all_ground_certainty_map", OccupancyGrid, queue_size = 1)
        self.r_map_pub = self.create_publisher("~roughness_map", OccupancyGrid, queue_size = 1)

        self.timer = self.create_timer(1.0/self.freq, self.map_pub_callback)
        
        self.lidar_debug_pub = self.create_publisher('~debug/lidar', PointCloud2, queue_size = 1)
        self.voxel_debug_pub = self.create_publisher('~debug/voxel', PointCloud2, queue_size = 1)
        self.voxel_hm_debug_pub = self.create_publisher('~debug/height_map', PointCloud2, queue_size = 1)
        self.voxel_inf_hm_debug_pub = self.create_publisher('~debug/inferred_height_map', PointCloud2, queue_size = 1)


    def map_pub_callback(self):
        pass

def main(args=None):
    rclpy.init(args=args)
    voxel_mapper = VoxelMapper()
    rclpy.spin(voxel_mapper)




if __name__ == '__main__':
    main()