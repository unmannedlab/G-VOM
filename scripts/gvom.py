from cmath import nan
import numba
from numba import vectorize, jit, cuda
import numpy as np
import math
import time
import threading

class Gvom:

    """ 
    A class to take lidar scanns and create a costmap\n
    xy_resolution:  x,y resolution in metters of each voxel\n
    z_resolution:   z resolution in metters of each voxel\n
    xy_size:        Number of voxels in x,y\n
    z_size:         Number of voxels in z\n
    buffer_size:    Number of lidar scans to keep in memory\n
    min_distance:   Minimum point distance, any points closer than this will be discarded\n

    """

    def __init__(self, xy_resolution, z_resolution, xy_size, z_size, buffer_size, min_distance,positive_obstacle_threshold, 
    negative_obstacle_threshold, slope_obsacle_threshold, robot_height, robot_radius,ground_to_lidar_height,xy_eigen_dist,z_eigen_dist):

        # print("init")

        self.xy_resolution = xy_resolution
        self.z_resolution = z_resolution

        self.xy_size = xy_size
        self.z_size = z_size

        self.buffer_size = buffer_size
        self.radar_buffer_size = buffer_size

        self.metrics = np.array([[3,2]])

        self.min_distance = min_distance

        self.positive_obstacle_threshold = positive_obstacle_threshold
        self.radar_positive_obstacle_threshold = positive_obstacle_threshold#radar_positive_obstacle_threshold
        self.negative_obstacle_threshold = negative_obstacle_threshold
        self.slope_obsacle_threshold = slope_obsacle_threshold
        self.robot_height = robot_height
        self.robot_radius = robot_radius
        self.ground_to_lidar_height = ground_to_lidar_height

        self.xy_eigen_dist = xy_eigen_dist  # When calculating covariance eigenvalues all points in voxels within a raidus of [xy_eigen_dist] in xy and [z_eigen_dist] in z voxels will be used
        self.z_eigen_dist = z_eigen_dist    # This radius is in number of voxels, ie r = 0 -> just points within the voxel, r=1 a 3x3 voxel cube centered on the voxel

        self.radar_ground_threshold = 18000# Voxels with a return strength above this will be considederd solid for ground segmentation
        self.radar_obs_threshold = 18000# Voxels with a return strength above this will be considederd solid for obstacle segmentation

        self.metrics_count = 10 # Mean: x, y, z; Covariance: xx, xy, xz, yy, yz, zz; Covariance point count
        self.radar_metrics_count = 5 # Average return strength, radar_point_count, grad x,y,z

        self.index_buffer = []
        self.hit_count_buffer = []
        self.total_count_buffer = []
        self.metrics_buffer = []
        self.origin_buffer = []
        self.min_height_buffer = []
        self.semaphores = []

        self.combined_index_map = None
        self.combined_hit_count = None
        self.combined_total_count = None
        self.combined_min_height = None
        self.combined_origin = None
        self.combined_metrics = None
        self.combined_cell_count_cpu = None

        self.radar_index_buffer = []
        self.radar_hit_count_buffer = []
        self.radar_metrics_buffer = []
        self.radar_origin_buffer = []
        self.radar_semaphores = []

        self.radar_combined_index_map = None
        self.radar_combined_hit_count = None
        self.radar_combined_origin = None
        self.radar_combined_metrics = None
        self.radar_combined_cell_count_cpu = None        

        self.height_map = None
        self.inferred_height_map = None
        self.roughness_map = None
        self.guessed_height_delta = None
        self.voxels_eigenvalues = None

        self.radar_map = None
        self.radar_height_map = None

        self.buffer_index = 0
        self.last_buffer_index = 0

        self.radar_buffer_index = 0
        self.radar_last_buffer_index = 0

        self.voxel_count = self.xy_size*self.xy_size*self.z_size

        for i in range(self.buffer_size):
            self.index_buffer.append(None)
            self.hit_count_buffer.append(None)
            self.total_count_buffer.append(None)
            self.metrics_buffer.append(None)
            self.origin_buffer.append(None)
            self.min_height_buffer.append(None)
            self.semaphores.append(threading.Semaphore())

        for i in range(self.radar_buffer_size):
            self.radar_index_buffer.append(None)
            self.radar_hit_count_buffer.append(None)
            self.radar_metrics_buffer.append(None)
            self.radar_origin_buffer.append(None)
            self.radar_semaphores.append(threading.Semaphore())

        self.last_combined_index_map = None
        self.last_combined_hit_count = None
        self.last_combined_total_count = None
        self.last_combined_min_height = None
        self.last_combined_origin = None
        self.last_combined_metrics = None
        self.last_combined_cell_count_cpu = None

        self.radar_last_combined_index_map = None
        self.radar_last_combined_hit_count = None
        self.radar_last_combined_origin = None
        self.radar_last_combined_metrics = None
        self.radar_last_combined_cell_count_cpu = None

        self.threads_per_block = 256
        self.threads_per_block_3D = (8, 8, 4)
        self.threads_per_block_2D = (16, 16)

        self.blocks = math.ceil(self.xy_size*self.xy_size*self.z_size / self.threads_per_block)

        self.ego_semaphore = threading.Semaphore()
        self.ego_position = [0,0,0]

    def Process_radar_pointcloud(self, pointcloud, ego_position, transform=None):
        """ 
        Imports a pointcloud from radar and processes it into a voxel map then adds the map to the buffer\n
        Assumes pointcloud is an array with shape [N,3] with x,y,z
        """

        # Import pointcloud
        self.ego_semaphore.acquire()
        print("acquire ego")
        self.ego_position = ego_position
        self.ego_semaphore.release()
        print("release ego")

        point_count = pointcloud.shape[0]
        print(pointcloud.shape)
        pointcloud = cuda.to_device(pointcloud)

        # Initilize arrays on GPU
        cell_count = cuda.to_device(np.zeros([1], dtype=np.int32))
        # -1 = unknown, -1 < free space, >= 0 point index in shorter arrays
        
        
        index_map = cuda.device_array([self.xy_size*self.xy_size*self.z_size], dtype=np.int32)
        self.__init_1D_array[self.blocks,self.threads_per_block](index_map,-1,self.xy_size*self.xy_size*self.z_size)
        
        tmp_hit_count = cuda.device_array([self.xy_size*self.xy_size*self.z_size], dtype=np.int32)
        self.__init_1D_array[self.blocks,self.threads_per_block](tmp_hit_count,0,self.xy_size*self.xy_size*self.z_size)

        tmp_total_count = cuda.device_array([self.xy_size*self.xy_size*self.z_size], dtype=np.int32)
        self.__init_1D_array[self.blocks,self.threads_per_block](tmp_total_count,0,self.xy_size*self.xy_size*self.z_size)


        origin = np.zeros([3])
        origin[0] = math.floor((ego_position[0]/self.xy_resolution) - self.xy_size/2)
        origin[1] = math.floor((ego_position[1]/self.xy_resolution) - self.xy_size/2)
        origin[2] = math.floor((ego_position[2]/self.z_resolution) - self.z_size/2)

        ego_position = cuda.to_device(ego_position)
        origin = cuda.to_device(origin)


        blocks_pointcloud = int(np.ceil(point_count/self.threads_per_block))
        blocks_map = int(np.ceil(self.xy_size*self.xy_size * self.z_size/self.threads_per_block))


        # Transform pointcloud
        if not transform is None:
            self.__transform_pointcloud[blocks_pointcloud, self.threads_per_block](
                pointcloud, transform, point_count)


        # Count points in each voxel
        self.__point_2_map[blocks_pointcloud, self.threads_per_block](
            self.xy_resolution, self.z_resolution, self.xy_size, self.z_size, self.min_distance, pointcloud, tmp_hit_count, tmp_total_count, point_count, ego_position, origin, False)


        # Make index map so we only need to store data on non-empty voxels
        self.__assign_indices[blocks_map, self.threads_per_block](tmp_hit_count,tmp_total_count, index_map, cell_count, self.voxel_count)

        cell_count_cpu = cell_count.copy_to_host()[0]
        hit_count = cuda.device_array([cell_count_cpu], dtype=np.int32)
        #total_count = cuda.device_array([cell_count_cpu], dtype=np.int32)

        # Move count to smaller array
        self.__move_data[blocks_map, self.threads_per_block](
            tmp_hit_count, hit_count, index_map, self.voxel_count)

        #self.__move_data[blocks_map, self.threads_per_block](
        #    tmp_total_count, total_count, index_map, self.voxel_count)


        # Calculate metrics
        metrics = self.__calculate_metrics_radar_master(pointcloud, point_count, hit_count, index_map, cell_count_cpu, origin)
 
 
        # Assign data to buffer
        # Block the main thread from accessing this buffer index wile we write to it
        self.radar_semaphores[self.radar_buffer_index].acquire()
        print("acquire radar " + str(self.radar_buffer_index))

        self.radar_index_buffer[self.radar_buffer_index] = index_map
        self.radar_hit_count_buffer[self.radar_buffer_index] = hit_count
        #self.total_count_buffer[self.buffer_index] = total_count
        self.radar_metrics_buffer[self.radar_buffer_index] = metrics
        #self.min_height_buffer[self.buffer_index] = min_height
        self.radar_origin_buffer[self.radar_buffer_index] = origin  

        print(self.radar_origin_buffer)    
 
        self.radar_last_buffer_index = self.radar_buffer_index
        print("added data to " + str(self.radar_buffer_index))
        

        #release this buffer index
        self.radar_semaphores[self.radar_buffer_index].release()
        print("release radar " + str(self.radar_buffer_index))
        self.radar_buffer_index += 1
        if(self.radar_buffer_index >= self.radar_buffer_size):
            self.radar_buffer_index = 0
        
    def Process_pointcloud(self, pointcloud, ego_position, transform=None):
        """ 
        Imports a pointcloud from radar and processes it into a voxel map then adds the map to the buffer\n
        Assumes pointcloud is an array with shape [N,3] with x,y,z for lidar
        """
        
        # Import pointcloud
        self.ego_semaphore.acquire()
        print("acquire ego")
        
        self.ego_position = ego_position
        self.ego_semaphore.release()
        print("release ego")

        point_count = pointcloud.shape[0]
        pointcloud = cuda.to_device(pointcloud)

        # Initilize arrays on GPU
        cell_count = cuda.to_device(np.zeros([1], dtype=np.int32))
        # -1 = unknown, -1 < free space, >= 0 point index in shorter arrays
        
        
        index_map = cuda.device_array([self.xy_size*self.xy_size*self.z_size], dtype=np.int32)
        self.__init_1D_array[self.blocks,self.threads_per_block](index_map,-1,self.xy_size*self.xy_size*self.z_size)
        
        tmp_hit_count = cuda.device_array([self.xy_size*self.xy_size*self.z_size], dtype=np.int32)
        self.__init_1D_array[self.blocks,self.threads_per_block](tmp_hit_count,0,self.xy_size*self.xy_size*self.z_size)

        tmp_total_count = cuda.device_array([self.xy_size*self.xy_size*self.z_size], dtype=np.int32)
        self.__init_1D_array[self.blocks,self.threads_per_block](tmp_total_count,0,self.xy_size*self.xy_size*self.z_size)


        origin = np.zeros([3])
        origin[0] = math.floor((ego_position[0]/self.xy_resolution) - self.xy_size/2)
        origin[1] = math.floor((ego_position[1]/self.xy_resolution) - self.xy_size/2)
        origin[2] = math.floor((ego_position[2]/self.z_resolution) - self.z_size/2)

        ego_position = cuda.to_device(ego_position)
        origin = cuda.to_device(origin)


        blocks_pointcloud = int(np.ceil(point_count/self.threads_per_block))
        blocks_map = int(np.ceil(self.xy_size*self.xy_size * self.z_size/self.threads_per_block))


        # Transform pointcloud
        if not transform is None:
            self.__transform_pointcloud[blocks_pointcloud, self.threads_per_block](
                pointcloud, transform, point_count)


        # Count points in each voxel and number of rays through each voxel
        self.__point_2_map[blocks_pointcloud, self.threads_per_block](
            self.xy_resolution, self.z_resolution, self.xy_size, self.z_size, self.min_distance, pointcloud, tmp_hit_count, tmp_total_count, point_count, ego_position, origin, True)


        # Make index map so we only need to store data on non-empty voxels
        self.__assign_indices[blocks_map, self.threads_per_block](tmp_hit_count,tmp_total_count, index_map, cell_count, self.voxel_count)

        cell_count_cpu = cell_count.copy_to_host()[0]
        hit_count = cuda.device_array([cell_count_cpu], dtype=np.int32)
        total_count = cuda.device_array([cell_count_cpu], dtype=np.int32)

        # Move count to smaller array
        self.__move_data[blocks_map, self.threads_per_block](
            tmp_hit_count, hit_count, index_map, self.voxel_count)

        self.__move_data[blocks_map, self.threads_per_block](
            tmp_total_count, total_count, index_map, self.voxel_count)


        # Calculate metrics
        metrics, min_height = self.__calculate_metrics_master(pointcloud, point_count, hit_count, index_map, cell_count_cpu, origin)
 
 
        # Assign data to buffer
        # Block the main thread from accessing this buffer index wile we write to it
        print("wating for "+ str(self.buffer_index))
        self.semaphores[self.buffer_index].acquire()
        print("acquire " + str(self.buffer_index))

        self.index_buffer[self.buffer_index] = index_map
        self.hit_count_buffer[self.buffer_index] = hit_count
        self.total_count_buffer[self.buffer_index] = total_count
        self.metrics_buffer[self.buffer_index] = metrics
        self.min_height_buffer[self.buffer_index] = min_height
        self.origin_buffer[self.buffer_index] = origin      
        
        #release this buffer index
        self.semaphores[self.buffer_index].release()
        print("release" + str(self.buffer_index))
        
        self.last_buffer_index = self.buffer_index

        self.buffer_index += 1

        if(self.buffer_index >= self.buffer_size):
            self.buffer_index = 0

    def combine_radar_maps(self):
        if(self.radar_origin_buffer[self.radar_last_buffer_index] is None):
            print("ERROR: No data in radar buffer")
            print(self.radar_origin_buffer)   
            print(self.radar_last_buffer_index)
            return
        
        self.radar_combined_origin = cuda.to_device(self.radar_origin_buffer[self.radar_last_buffer_index].copy_to_host())

        combined_cell_count = np.zeros([1], dtype=np.int64)
        self.radar_combined_index_map = cuda.device_array([self.xy_size*self.xy_size*self.z_size], dtype=np.int32)
        self.__init_1D_array[self.blocks,self.threads_per_block](self.radar_combined_index_map,-1,self.xy_size*self.xy_size*self.z_size)

        blockspergrid_xy = math.ceil(self.xy_size / self.threads_per_block_3D[0])
        blockspergrid_z = math.ceil(self.z_size / self.threads_per_block_3D[2])
        blockspergrid = (blockspergrid_xy, blockspergrid_xy, blockspergrid_z)

        # Combines the index maps and calculates the nessisary size for the combined map

        for i in range(0, self.radar_buffer_size):
            self.radar_semaphores[i].acquire()
            print("acquire radar " + str(i))
            if(self.radar_origin_buffer[i] is None):
                self.radar_semaphores[i].release()
                print("release radar " + str(i))
                print("no data in " + str(i))
                continue
            print("combine indices")
            
            self.__combine_indices[blockspergrid, self.threads_per_block_3D](
                combined_cell_count, self.radar_combined_index_map, self.radar_combined_origin, self.radar_index_buffer[i], self.voxel_count, self.radar_origin_buffer[i], self.xy_size, self.z_size)
            self.radar_semaphores[i].release()
            print("release radar " + str(i))

        if not (self.radar_last_combined_origin is None):
            print("combine_old_indices")
             #__combine_old_indices
            self.__combine_old_indices[blockspergrid, self.threads_per_block_3D](
                 combined_cell_count, self.radar_combined_index_map, self.radar_combined_origin, self.radar_last_combined_index_map, self.voxel_count, self.radar_last_combined_origin, self.xy_size, self.z_size)

        self.radar_combined_cell_count_cpu = combined_cell_count[0]

        blockspergrid_cell = math.ceil(self.radar_combined_cell_count_cpu/self.threads_per_block)
        self.radar_combined_hit_count = cuda.device_array([self.radar_combined_cell_count_cpu], dtype=np.int32)
        self.__init_1D_array[blockspergrid_cell,self.threads_per_block](self.radar_combined_hit_count,0,self.radar_combined_cell_count_cpu)
                
        blockspergrid_cell_2D = math.ceil(self.radar_combined_cell_count_cpu / self.threads_per_block_2D[0])
        blockspergrid_metric_2D = math.ceil(self.radar_metrics_count / self.threads_per_block_2D[1])
        blockspergrid_2D = (blockspergrid_cell_2D, blockspergrid_metric_2D)

        self.radar_combined_metrics = cuda.device_array([self.radar_combined_cell_count_cpu,self.radar_metrics_count], dtype=np.float32)
        self.__init_2D_array[blockspergrid_2D,self.threads_per_block_2D](self.radar_combined_metrics,0,self.radar_combined_cell_count_cpu, self.radar_metrics_count)

        # Combines the data in the buffer
        for i in range(0, self.radar_buffer_size):
            
            self.radar_semaphores[i].acquire()
            print("acquire radar " + str(i))

            if(self.radar_origin_buffer[i] is None):
                self.radar_semaphores[i].release()
                print("release radar " + str(i))
                continue
            
            print("combine metrics")
            self.__radar_combine_metrics[blockspergrid, self.threads_per_block_3D](
                self.radar_combined_metrics, self.radar_combined_hit_count, self.radar_combined_index_map, self.radar_combined_origin, self.radar_metrics_buffer[i],
                self.radar_hit_count_buffer[i], self.radar_index_buffer[i], self.radar_origin_buffer[i], self.xy_size, self.z_size)
            
            self.radar_semaphores[i].release()
            print("release radar " + str(i))

        # fill unknown cells with data from the last combined map
        if not (self.radar_last_combined_origin is None):
                #__combine_old_metrics
            self.__radar_combine_metrics[blockspergrid, self.threads_per_block_3D](self.radar_combined_metrics, self.radar_combined_hit_count, self.radar_combined_index_map, self.radar_combined_origin, self.radar_last_combined_metrics,
                                                                                  self.radar_last_combined_hit_count, self.radar_last_combined_index_map, self.radar_last_combined_origin, self.xy_size, self.z_size)

        # set the last combined map

        self.radar_last_combined_cell_count_cpu = self.radar_combined_cell_count_cpu
        self.radar_last_combined_hit_count = self.radar_combined_hit_count
        self.radar_last_combined_index_map = self.radar_combined_index_map
        self.radar_last_combined_metrics = self.radar_combined_metrics
        self.radar_last_combined_origin = self.radar_combined_origin

        # make output maps
        tmp_height_map = cuda.device_array([self.xy_size,self.xy_size]) 
        self.__init_2D_array[blockspergrid, self.threads_per_block_2D](tmp_height_map,-1000.0,self.xy_size,self.xy_size)
        self.radar_height_map = cuda.device_array([self.xy_size,self.xy_size])
        self.__init_2D_array[blockspergrid, self.threads_per_block_2D](self.radar_height_map,-1000.0,self.xy_size,self.xy_size)

        self.__make_radar_height_map[blockspergrid,self.threads_per_block_2D](
            self.radar_combined_origin, self.radar_combined_index_map, self.radar_combined_metrics, self.xy_size, self.z_size, self.xy_resolution, 
            self.z_resolution, self.radar_ground_threshold, tmp_height_map)

        self.__smooth_height_map[blockspergrid,self.threads_per_block_2D](tmp_height_map,self.radar_height_map,1,self.xy_size)

        self.radar_obs_map = cuda.device_array([self.xy_size,self.xy_size])
        self.__init_2D_array[blockspergrid, self.threads_per_block_2D](self.radar_obs_map,0,self.xy_size,self.xy_size)

        # make obstacle map
        self.__make_radar_positive_obstacle_map[blockspergrid,self.threads_per_block_2D](self.radar_combined_index_map, self.radar_height_map, self.xy_size, self.z_size, self.z_resolution, self.radar_positive_obstacle_threshold, self.robot_height, self.radar_combined_origin, self.radar_obs_map)
        
        # make ground visability map
        visability_map = cuda.device_array([self.xy_size,self.xy_size],dtype=np.int32)

        self.__make_visability_map[blockspergrid, self.threads_per_block_2D](visability_map,self.radar_height_map,self.xy_size)


        #format output data
        combined_origin_world = self.radar_combined_origin.copy_to_host()
        combined_origin_world[0] = combined_origin_world[0] * self.xy_resolution
        combined_origin_world[1] = combined_origin_world[1] * self.xy_resolution
        combined_origin_world[2] = combined_origin_world[2] * self.z_resolution

        return (combined_origin_world,self.radar_obs_map.copy_to_host(), visability_map.copy_to_host())


    def combine_maps(self):
        """ Combines all maps in the buffer and processes into 2D maps """
        #voxel_start_time = time.time()
        if(self.origin_buffer[self.last_buffer_index] is None):
            print("ERROR: No data in buffer")
            return

        self.combined_origin = cuda.to_device(self.origin_buffer[self.last_buffer_index].copy_to_host())

        combined_cell_count = np.zeros([1], dtype=np.int64)
        self.combined_index_map = cuda.device_array([self.xy_size*self.xy_size*self.z_size], dtype=np.int32)
        self.__init_1D_array[self.blocks,self.threads_per_block](self.combined_index_map,-1,self.xy_size*self.xy_size*self.z_size)


        blockspergrid_xy = math.ceil(self.xy_size / self.threads_per_block_3D[0])
        blockspergrid_z = math.ceil(self.z_size / self.threads_per_block_3D[2])
        blockspergrid = (blockspergrid_xy, blockspergrid_xy, blockspergrid_z)

        # Combines the index maps and calculates the nessisary size for the combined map
        print("combine index")
        for i in range(0, self.buffer_size):
            self.semaphores[i].acquire()
            print("acquire " + str(i))
            if(self.origin_buffer[i] is None):
                self.semaphores[i].release()
                print("release " + str(i))
                continue

            
            self.__combine_indices[blockspergrid, self.threads_per_block_3D](
                combined_cell_count, self.combined_index_map, self.combined_origin, self.index_buffer[i], self.voxel_count, self.origin_buffer[i], self.xy_size, self.z_size)
            self.semaphores[i].release()
            print("release " + str(i))

        if not (self.last_combined_origin is None):
             #print("combine_old_indices")
             #__combine_old_indices
            self.__combine_old_indices[blockspergrid, self.threads_per_block_3D](
                 combined_cell_count, self.combined_index_map, self.combined_origin, self.last_combined_index_map, self.voxel_count, self.last_combined_origin, self.xy_size, self.z_size)

        self.combined_cell_count_cpu = combined_cell_count[0]
        print(self.combined_cell_count_cpu)

        blockspergrid_cell = math.ceil(self.combined_cell_count_cpu/self.threads_per_block)
        self.combined_hit_count = cuda.device_array([self.combined_cell_count_cpu], dtype=np.int32)
        self.__init_1D_array[blockspergrid_cell,self.threads_per_block](self.combined_hit_count,0,self.combined_cell_count_cpu)
                
        self.combined_total_count = cuda.device_array([self.combined_cell_count_cpu], dtype=np.int32)
        self.__init_1D_array[blockspergrid_cell,self.threads_per_block](self.combined_total_count,0,self.combined_cell_count_cpu)

        self.combined_min_height = cuda.device_array([self.combined_cell_count_cpu], dtype=np.float32)
        self.__init_1D_array[blockspergrid_cell,self.threads_per_block](self.combined_min_height,1,self.combined_cell_count_cpu)

        blockspergrid_cell_2D = math.ceil(self.combined_cell_count_cpu / self.threads_per_block_2D[0])
        blockspergrid_metric_2D = math.ceil(self.metrics_count / self.threads_per_block_2D[1])
        blockspergrid_2D = (blockspergrid_cell_2D, blockspergrid_metric_2D)

        self.combined_metrics = cuda.device_array([self.combined_cell_count_cpu,self.metrics_count], dtype=np.float32)
        self.__init_2D_array[blockspergrid_2D,self.threads_per_block_2D](self.combined_metrics,0,self.combined_cell_count_cpu, self.metrics_count)
        print("combine data")
        # Combines the data in the buffer
        for i in range(0, self.buffer_size):
            
            print(self.semaphores[i].acquire())
            print("acquire c " + str(i))

            if(self.origin_buffer[i] is None):
                self.semaphores[i].release()
                print("release  c " + str(i))
                continue
            
            print(i)
            self.__combine_metrics[blockspergrid, self.threads_per_block_3D](self.combined_metrics, self.combined_hit_count,self.combined_total_count,self.combined_min_height, self.combined_index_map, self.combined_origin, self.metrics_buffer[
                                                                             i], self.hit_count_buffer[i],self.total_count_buffer[i], self.min_height_buffer[i],self.index_buffer[i], self.origin_buffer[i], self.voxel_count, self.metrics, self.xy_size, self.z_size, len(self.metrics))
            
            self.semaphores[i].release()
            print("release c " + str(i))

        # fill unknown cells with data from the last combined map
        if not (self.last_combined_origin is None):
                #__combine_old_metrics
            self.__combine_metrics[blockspergrid, self.threads_per_block_3D](self.combined_metrics, self.combined_hit_count,self.combined_total_count,self.combined_min_height, self.combined_index_map, self.combined_origin, self.last_combined_metrics,
                                                                                  self.last_combined_hit_count,self.last_combined_total_count,self.last_combined_min_height, self.last_combined_index_map, self.last_combined_origin, self.voxel_count, self.metrics, self.xy_size, self.z_size, len(self.metrics))

        # set the last combined map

        self.last_combined_cell_count_cpu = self.combined_cell_count_cpu
        self.last_combined_hit_count = self.combined_hit_count
        self.last_combined_total_count = self.combined_total_count
        self.last_combined_index_map = self.combined_index_map
        self.last_combined_metrics = self.combined_metrics
        self.last_combined_min_height = self.combined_min_height
        self.last_combined_origin = self.combined_origin

        # Compute eigenvalues for each voxel
        print("eigen")
        blockspergrid_cell_2D = math.ceil(self.combined_cell_count_cpu / self.threads_per_block_2D[0])
        blockspergrid_eigenvalue_2D = math.ceil(3 / self.threads_per_block_2D[1])
        blockspergrid_2D = (blockspergrid_cell_2D, blockspergrid_eigenvalue_2D)

        self.voxels_eigenvalues = cuda.device_array([self.combined_cell_count_cpu,3], dtype=np.float32)
        self.__init_2D_array[blockspergrid_2D,self.threads_per_block_2D](self.voxels_eigenvalues,0,self.combined_cell_count_cpu, 3)

        self.__calculate_eigenvalues[blockspergrid_cell,self.threads_per_block](self.voxels_eigenvalues,self.combined_metrics,self.combined_cell_count_cpu)


        #print("voxel map rate = " + str(1.0 / (time.time() - voxel_start_time)))
        #map_start_time = time.time()

        # Make 2d maps from combined map
        print("maps")
        # Make height map from minimum height in lowest cell
        self.height_map = cuda.device_array([self.xy_size,self.xy_size])
        self.__init_2D_array[blockspergrid, self.threads_per_block_2D](self.height_map,-1000.0,self.xy_size,self.xy_size)
        
        self.inferred_height_map = cuda.device_array([self.xy_size,self.xy_size])
        self.__init_2D_array[blockspergrid, self.threads_per_block_2D](self.inferred_height_map,-1000.0,self.xy_size,self.xy_size)

        self.ego_semaphore.acquire()
        print("acquire ego")
        self.__make_height_map[blockspergrid, self.threads_per_block_2D](
            self.combined_origin, self.combined_index_map, self.combined_min_height, self.xy_size, self.z_size, self.xy_resolution, self.z_resolution,self.ego_position,self.robot_radius,self.ground_to_lidar_height, self.height_map)
        self.ego_semaphore.release()
        print("release ego")

        self.__make_inferred_height_map[blockspergrid, self.threads_per_block_2D](
            self.combined_origin, self.combined_index_map, self.xy_size, self.z_size, self.z_resolution, self.inferred_height_map)
        # Estimate ground slope

        self.roughness_map = cuda.device_array([self.xy_size,self.xy_size])
        self.__init_2D_array[blockspergrid, self.threads_per_block_2D](self.roughness_map,-1.0,self.xy_size,self.xy_size)

        self.x_slope_map = cuda.device_array([self.xy_size,self.xy_size])
        self.__init_2D_array[blockspergrid, self.threads_per_block_2D](self.x_slope_map,0.0,self.xy_size,self.xy_size)

        self.y_slope_map = cuda.device_array([self.xy_size,self.xy_size])
        self.__init_2D_array[blockspergrid, self.threads_per_block_2D](self.y_slope_map,0.0,self.xy_size,self.xy_size)

        self.__calculate_slope[blockspergrid, self.threads_per_block_2D](
            self.height_map, self.xy_size, self.xy_resolution, self.x_slope_map, self.y_slope_map, self.roughness_map)

        # Guess what the height is in unobserved cells
        self.guessed_height_delta = cuda.device_array([self.xy_size,self.xy_size])
        self.__init_2D_array[blockspergrid, self.threads_per_block_2D](self.guessed_height_delta,0.0,self.xy_size,self.xy_size)
        
        self.__guess_height[blockspergrid, self.threads_per_block_2D](
            self.height_map,self.inferred_height_map,self.xy_size,self.xy_resolution,self.x_slope_map,self.y_slope_map,self.guessed_height_delta)


        # Check for positive obstacles. Any cell where the max height is more than "threshold" above the height map and less than "threshold + robot height" is marked as an obstacle
        # Obstacle type can be determined from cell metrics
        positive_obstacle_map = cuda.device_array([self.xy_size,self.xy_size],dtype=np.int32)
        self.__init_2D_array[blockspergrid, self.threads_per_block_2D](positive_obstacle_map,0,self.xy_size,self.xy_size)

        self.__make_positive_obstacle_map[blockspergrid, self.threads_per_block_2D](
            self.combined_index_map, self.height_map, self.xy_size, self.z_size, self.z_resolution, self.positive_obstacle_threshold,self.combined_hit_count,self.combined_total_count, self.robot_height, self.combined_origin,self.x_slope_map,self.y_slope_map,self.slope_obsacle_threshold, positive_obstacle_map)


        # Check for negative obstacles. 
        negative_obstacle_map = cuda.device_array([self.xy_size,self.xy_size],dtype=np.int32)
        self.__init_2D_array[blockspergrid, self.threads_per_block_2D](negative_obstacle_map,0,self.xy_size,self.xy_size)

        self.__make_negative_obstacle_map[blockspergrid, self.threads_per_block_2D](self.guessed_height_delta,negative_obstacle_map,self.negative_obstacle_threshold,self.xy_size)

        # make ground visability map
        visability_map = cuda.device_array([self.xy_size,self.xy_size],dtype=np.int32)

        self.__make_visability_map[blockspergrid, self.threads_per_block_2D](visability_map,self.height_map,self.xy_size)


        # format output data

        combined_origin_world = self.combined_origin.copy_to_host()
        combined_origin_world[0] = combined_origin_world[0] * self.xy_resolution
        combined_origin_world[1] = combined_origin_world[1] * self.xy_resolution
        combined_origin_world[2] = combined_origin_world[2] * self.z_resolution

        #print("2d map rate = " + str(1.0 / (time.time() - map_start_time)))

        # return all maps as cpu arrays
        return (combined_origin_world, positive_obstacle_map.copy_to_host(),negative_obstacle_map.copy_to_host(),self.roughness_map.copy_to_host(),visability_map.copy_to_host() )

    def make_debug_voxel_map(self):
        if(self.combined_cell_count_cpu is None):
            print("No data")
            return None
        blockspergrid_xy = math.ceil(
            self.xy_size / self.threads_per_block_3D[0])
        blockspergrid_z = math.ceil(self.z_size / self.threads_per_block_3D[2])
        blockspergrid = (blockspergrid_xy, blockspergrid_xy, blockspergrid_z)

        output_voxel_map = np.zeros(
            [self.combined_cell_count_cpu, 8], np.float32)

        self.__make_voxel_pointcloud[blockspergrid, self.threads_per_block_3D](
            self.combined_index_map,self.combined_hit_count,self.combined_total_count,self.voxels_eigenvalues, self.combined_origin, output_voxel_map, self.xy_size, self.z_size, self.xy_resolution, self.z_resolution)

        return output_voxel_map

    def make_debug_radar_map(self, threshold = 22000):
        if(self.radar_combined_cell_count_cpu is None):
            print("No data")
            return None
        blockspergrid_xy = math.ceil(
            self.xy_size / self.threads_per_block_3D[0])
        blockspergrid_z = math.ceil(self.z_size / self.threads_per_block_3D[2])
        blockspergrid = (blockspergrid_xy, blockspergrid_xy, blockspergrid_z)

        output_voxel_map = np.zeros(
            [self.radar_combined_cell_count_cpu, 9], np.float32)


        self.__make_radar_pointcloud[blockspergrid, self.threads_per_block_3D](
            self.radar_combined_index_map,self.radar_combined_hit_count,self.radar_combined_metrics, self.radar_combined_origin, output_voxel_map,threshold, self.xy_size, self.z_size, self.xy_resolution, self.z_resolution)

        return output_voxel_map


    def make_debug_height_map(self):
        if(self.height_map is None):
            print("No data")
            return None

        output_height_map_voxel = np.zeros(
            [self.xy_size*self.xy_size, 7], np.float32)

        blockspergrid_xy = math.ceil(
            self.xy_size / self.threads_per_block_2D[0])
        blockspergrid = (blockspergrid_xy, blockspergrid_xy)
        self.__make_height_map_pointcloud[blockspergrid, self.threads_per_block_2D](
            self.height_map,self.roughness_map,self.x_slope_map,self.y_slope_map, self.combined_origin, output_height_map_voxel, self.xy_size, self.xy_resolution,self.z_resolution)

        return output_height_map_voxel

    def make_debug_radar_height_map(self):
        if(self.radar_height_map is None):
            print("No data")
            return None

        output_height_map_voxel = np.zeros(
            [self.xy_size*self.xy_size, 3], np.float32)

        blockspergrid_xy = math.ceil(
            self.xy_size / self.threads_per_block_2D[0])
        blockspergrid = (blockspergrid_xy, blockspergrid_xy)
        self.__make_radar_height_map_pointcloud[blockspergrid, self.threads_per_block_2D](
            self.radar_height_map,self.radar_combined_origin, output_height_map_voxel, self.xy_size, self.xy_resolution,self.z_resolution)

        return output_height_map_voxel

    def make_debug_inferred_height_map(self):
        if(self.height_map is None):
            print("No data")
            return None

        output_height_map_voxel = np.zeros(
            [self.xy_size*self.xy_size, 3], np.float32)

        blockspergrid_xy = math.ceil(
            self.xy_size / self.threads_per_block_2D[0])
        blockspergrid = (blockspergrid_xy, blockspergrid_xy)
        self.__make_infered_height_map_pointcloud[blockspergrid, self.threads_per_block_2D](
            self.guessed_height_delta, self.combined_origin, output_height_map_voxel, self.xy_size, self.xy_resolution,self.z_resolution)

        return output_height_map_voxel
    
    @cuda.jit
    def __make_visability_map(visability,height_map,xy_size):
        x, y = cuda.grid(2)
        if(x >= xy_size or y >= xy_size):
            return
        if(height_map[x,y] > - 1000):
            visability[x,y] = 1.0
        else:
            visability[x,y] = 0.0

    @cuda.jit
    def __make_radar_height_map_pointcloud(height_map, origin, output_voxel_map, xy_size, xy_resolution,z_resolution):
        x, y = cuda.grid(2)
        if(x >= xy_size or y >= xy_size):
            return
        index = x + y*xy_size
        if(index >= 0):
            output_voxel_map[index, 0] = (x + origin[0]) * xy_resolution
            output_voxel_map[index, 1] = (y + origin[1]) * xy_resolution
            output_voxel_map[index, 2] = height_map[x, y] - z_resolution

    @cuda.jit
    def __make_height_map_pointcloud(height_map,roughness,x_slope,y_slope, origin, output_voxel_map, xy_size, xy_resolution,z_resolution):
        x, y = cuda.grid(2)
        if(x >= xy_size or y >= xy_size):
            return
        index = x + y*xy_size
        if(index >= 0):
            output_voxel_map[index, 0] = (x + origin[0]) * xy_resolution
            output_voxel_map[index, 1] = (y + origin[1]) * xy_resolution
            output_voxel_map[index, 2] = height_map[x, y] - z_resolution
            output_voxel_map[index, 3] = roughness[x,y]
            output_voxel_map[index, 4] = x_slope[x,y]
            output_voxel_map[index, 5] = y_slope[x,y]
            output_voxel_map[index, 6] = math.sqrt(x_slope[x,y] * x_slope[x,y] + y_slope[x,y]*y_slope[x,y])

    @cuda.jit
    def __make_infered_height_map_pointcloud(height_map, origin, output_voxel_map, xy_size, xy_resolution,z_resolution):
        x, y = cuda.grid(2)
        if(x >= xy_size or y >= xy_size):
            return
        index = x + y*xy_size
        if(index >= 0):
            output_voxel_map[index, 0] = (x + origin[0]) * xy_resolution
            output_voxel_map[index, 1] = (y + origin[1]) * xy_resolution
            output_voxel_map[index, 2] = height_map[x, y] - z_resolution

    @cuda.jit
    def __make_voxel_pointcloud(combined_index_map, combined_hit_count,combined_total_count, eigenvalues, origin, output_voxel_map, xy_size, z_size, xy_resolution, z_resolution):
        x, y, z = cuda.grid(3)
        if(x >= xy_size or y >= xy_size or z > z_size):
            return

        index = int(combined_index_map[int(
            x + y * xy_size + z * xy_size * xy_size)])
        if(index >= 0):
            output_voxel_map[index, 0] = (x + origin[0]) * xy_resolution
            output_voxel_map[index, 1] = (y + origin[1]) * xy_resolution
            output_voxel_map[index, 2] = (z + origin[2]) * z_resolution
            output_voxel_map[index, 3] = float(combined_hit_count[index]) / float(combined_total_count[index])
            output_voxel_map[index, 4] = combined_hit_count[index]

            d1 = eigenvalues[index,0] - eigenvalues[index,1]
            d2 = eigenvalues[index,1] - eigenvalues[index,2]

            output_voxel_map[index, 5] = d1
            output_voxel_map[index, 6] = d2
            output_voxel_map[index, 7] = eigenvalues[index,2]
            #if(d1 > 0.0) and (d2 > 0.0):
            #    output_voxel_map[index, 7] = math.log10(d1/d2)

    @cuda.jit
    def __make_radar_pointcloud(combined_index_map, combined_hit_count, metrics, origin, output_voxel_map,threshold, xy_size, z_size, xy_resolution, z_resolution):
        x, y, z = cuda.grid(3)
        if(x >= xy_size or y >= xy_size or z > z_size):
            return

        index = int(combined_index_map[int(
            x + y * xy_size + z * xy_size * xy_size)])
        if(index >= 0):
            if(metrics[index,0] >= threshold):
                output_voxel_map[index, 0] = (x + origin[0]) * xy_resolution
                output_voxel_map[index, 1] = (y + origin[1]) * xy_resolution
                output_voxel_map[index, 2] = (z + origin[2]) * z_resolution
                output_voxel_map[index, 3] = combined_hit_count[index]
                output_voxel_map[index, 4] = metrics[index,0] # intensity
                output_voxel_map[index, 5] = metrics[index,2] # grad_x
                output_voxel_map[index, 6] = metrics[index,3] # grad_y
                output_voxel_map[index, 7] = abs(metrics[index,4]) # grad_z
                output_voxel_map[index, 8] = math.sqrt(metrics[index,2] * metrics[index,2] + metrics[index,1]*metrics[index,1]) # abs xy grad
            else:
                output_voxel_map[index, 0] = np.NAN
                output_voxel_map[index, 1] = np.NAN
                output_voxel_map[index, 2] = np.NAN
                output_voxel_map[index, 3] = np.NAN
                output_voxel_map[index, 4] = np.NAN

    @cuda.jit
    def __make_negative_obstacle_map(guessed_height_delta,negative_obstacle_map,negative_obstacle_threshold,xy_size):
        x, y = cuda.grid(2)
        if(x >= xy_size or y >= xy_size):
            return

        if(guessed_height_delta[x,y] > negative_obstacle_threshold):
            negative_obstacle_map[x,y] = 100
    
    @cuda.jit
    def __make_radar_positive_obstacle_map(combined_index_map, radar_height_map, xy_size, z_size, z_resolution, radar_positive_obstacle_threshold, robot_height, origin, radar_obstacle_map):
        x, y = cuda.grid(2)
        if(x >= xy_size or y >= xy_size):
            return
        
        min_obs_height = radar_height_map[x,y] + radar_positive_obstacle_threshold
        max_obs_height = radar_height_map[x,y] + robot_height

        min_height_index = int(math.floor((min_obs_height/z_resolution) - origin[2])) + 1
        max_height_index = int(math.floor((max_obs_height/z_resolution) - origin[2]))

        if not (min_height_index >= 0 and min_height_index < z_size):
            return
        
        if not (max_height_index >= 0 and max_height_index < z_size):
            return

        density = 0.0
        n = 0.0
        for z in range(min_height_index,max_height_index+1):
            index = int(combined_index_map[int(x + y * xy_size + z * xy_size * xy_size)])
            
            if(index >= 0):
                radar_obstacle_map[x, y] = 100

    @cuda.jit
    def __make_positive_obstacle_map(combined_index_map, height_map, xy_size, z_size, z_resolution, positive_obstacle_threshold,hit_count,total_count, robot_height, origin,x_slope,y_slope,slope_threshold,  obstacle_map):
        """
        Obstacle map reports the average density of occpied voxels within the obstacle range
        """
        x, y = cuda.grid(2)
        if(x >= xy_size or y >= xy_size):
            return

        if(math.sqrt(x_slope[x,y] * x_slope[x,y] + y_slope[x,y] * y_slope[x,y]) >= slope_threshold):
            obstacle_map[x,y] = 100
            return


        min_obs_height = height_map[x,y] + positive_obstacle_threshold
        max_obs_height = height_map[x,y] + robot_height

        min_height_index = int(math.floor((min_obs_height/z_resolution) - origin[2])) + 1
        max_height_index = int(math.floor((max_obs_height/z_resolution) - origin[2]))

        if not (min_height_index >= 0 and min_height_index < z_size):
            return
        
        if not (max_height_index >= 0 and max_height_index < z_size):
            return

        density = 0.0
        n = 0.0
        for z in range(min_height_index,max_height_index+1):
            index = int(combined_index_map[int(x + y * xy_size + z * xy_size * xy_size)])
            
            if(index >= 0):
                if(hit_count[index] > 10):
                    n += float(total_count[index])
                    density += float(hit_count[index])

        
        if(n>0.0):
            density /= n

        obstacle_map[x, y] = int(density * 100)


    @cuda.jit                   
    def __make_radar_height_map(combined_origin, combined_index_map, metrics, xy_size, z_size,xy_resolution, z_resolution,radar_ground_threshold, output_height_map):
        x, y = cuda.grid(2)
        if(x >= xy_size or y >= xy_size):
            return
        
        #xp = (((combined_origin[0] + x) * xy_resolution)  - ego_position[0])
        #yp = (((combined_origin[1] + y) * xy_resolution) - ego_position[1])

        #if(xp*xp + yp*yp <= radius*radius):
        #    output_height_map[x, y] = ego_position[2] - ground_to_lidar_height

        for z in range(z_size):
            index = combined_index_map[int(x + y * xy_size + z * xy_size * xy_size)]
            if(index >= 0):
                if(metrics[index,0]>= radar_ground_threshold):
                    output_height_map[x, y] = (z + combined_origin[2]) * z_resolution
                    return

    @cuda.jit
    def __smooth_height_map(input_map, output_map, smooth_size, xy_size):
        x,y = cuda.grid(2)
        if(x >= xy_size or y >= xy_size):
            return

        pts = 0.0
        h = 0.0
        for xp in range(max(x - smooth_size,0),min(x + smooth_size + 1,xy_size)):
            for yp in range(max(y - smooth_size,0),min(y + smooth_size + 1,xy_size)):
                if(input_map[xp,yp] >-1000):
                    pts += 1.0
                    h += input_map[xp,yp]
        
        if(pts > 0.0):
            output_map[x,y] = h / pts
        if(input_map[x,y] == -1000):
            output_map[x,y] = input_map[x,y]



    @cuda.jit                   
    def __make_height_map(combined_origin, combined_index_map, min_height, xy_size, z_size,xy_resolution, z_resolution,ego_position,radius,ground_to_lidar_height, output_height_map):
        x, y = cuda.grid(2)
        if(x >= xy_size or y >= xy_size):
            return
        
        xp = (((combined_origin[0] + x) * xy_resolution)  - ego_position[0])
        yp = (((combined_origin[1] + y) * xy_resolution) - ego_position[1])

        if(xp*xp + yp*yp <= radius*radius):
            output_height_map[x, y] = ego_position[2] - ground_to_lidar_height

        for z in range(z_size):
            index = combined_index_map[int(x + y * xy_size + z * xy_size * xy_size)]
            if(index >= 0):
                output_height_map[x, y] = ( min_height[index] + z + combined_origin[2]) * z_resolution
                return


    @cuda.jit
    def __make_inferred_height_map(combined_origin, combined_index_map, xy_size, z_size, z_resolution, output_inferred_height_map):
        x, y = cuda.grid(2)
        if(x >= xy_size or y >= xy_size):
            return

        for z in range(z_size):
            index = combined_index_map[int(x + y * xy_size + z * xy_size * xy_size)]
            if(index < -1):
                inferred_height = (z + combined_origin[2]) * z_resolution
                output_inferred_height_map[x, y] = inferred_height
                return

    @cuda.jit
    def __guess_height(height_map,inferred_height_map,xy_size,xy_resolution,slope_map_x,slope_map_y,output_guessed_height_delta):
        x0, y0 = cuda.grid(2)
        if(x0 >= xy_size or y0 >= xy_size):
            return
        if( height_map[x0,y0] > -1000 ):
            return
        if(inferred_height_map[x0,y0] == -1000.0):
            return

        x_p_done = False
        x_n_done = False
        y_p_done = False
        y_n_done = False


        x_p = x0
        x_ph = -1000
        x_n = x0
        x_nh = -1000
        y_p = y0
        y_ph = -1000
        y_n = y0
        y_nh = -1000


        i = 0
        while (i < 15 ) and (not (x_n_done and x_n_done and y_p_done and y_n_done)): 

            x_p += 1
            x_n -= 1
            y_p += 1
            y_n -= 1

            i += 1

            if not x_p_done:
                if(x_p < xy_size):

                    for dy in range(-i,i):
                        if(y0 + dy >= xy_size or y0+dy <0):
                            continue

                        if( height_map[x_p,y0 + dy] > -1000 ):
                        
                            x_ph = height_map[x_p,y0 + dy]
                            x_p_done = True
                            break
                else:
                    x_p_done = True

            if not x_n_done:
                if(x_n >= 0):

                    for dy in range(-i + 1 ,i + 1):
                        if(y0 + dy >= xy_size or y0+dy <0):
                            continue

                        if( height_map[x_n,y0 + dy] > -1000 ):
                        
                            x_nh = height_map[x_n,y0 + dy]
                            x_n_done = True
                            break
                else:
                    x_n_done = True
            
            if not y_p_done:
                if(y_p < xy_size):

                    for dx in range(-i+1,i+1):
                        if(x0 + dx >= xy_size or x0+dx <0):
                            continue

                        if( height_map[x0+dx,y_p] > -1000 ):
                        
                            y_ph = height_map[x0+dx,y_p]
                            y_p_done = True
                            break
                else:
                    y_p_done = True
            
            if not y_n_done:
                if(y_n >= 0):

                    for dx in range(-i ,i ):
                        if(x0 + dx >= xy_size or x0+dx <0):
                            continue

                        if( height_map[x0 + dx,y_n] > -1000 ):
                        
                            y_nh = height_map[x0 + dx,y_n]
                            y_n_done = True
                            break
                else:
                    y_n_done = True


        min_h = 1000.0
        max_h = inferred_height_map[x0,y0]

        if(x_ph > -1000):
            min_h = min(x_ph,min_h)
            max_h = max(x_ph,max_h)

        if(x_nh > -1000):
            min_h = min(x_nh,min_h)
            max_h = max(x_nh,max_h)

        if(y_ph > -1000):
            min_h = min(y_ph,min_h)
            max_h = max(y_ph,max_h)
        
        if(x_nh > -1000):
            min_h = min(y_nh,min_h)
            max_h = max(y_nh,max_h)
        


        dh = max_h - min_h 

        if(dh > 0):
            output_guessed_height_delta[x0,y0] = dh



    @cuda.jit
    def __calculate_slope(height_map,xy_size,xy_resolution,output_slope_map_x,output_slope_map_y, output_roughness_map):
        x0, y0 = cuda.grid(2)
        if(x0 >= xy_size or y0 >= xy_size):
            return

        n_good_pts = 0

        radius = 1
        for x in range(max(0,x0 - radius), min(xy_size, x0 + radius + 1)):
            for y in range(max(0,y0 - radius), min(xy_size, y0 + radius + 1)):
                if( height_map[x,y] > -1000 ):
                    n_good_pts += 1
        
        if(n_good_pts <3):
            return

        pts = numba.cuda.local.array((3,9),np.float64)
        
        i=0
        mean_x = 0.0
        mean_y = 0.0
        mean_z = 0.0
        for x in range(max(0,x0 - radius), min(xy_size, x0 + radius + 1)):
            for y in range(max(0,y0 - radius), min(xy_size, y0 + radius + 1)):
                if( height_map[x,y] > -1000 ):
                    pts[0,i] = x * xy_resolution
                    pts[1,i] = y * xy_resolution
                    pts[2,i] = height_map[x,y]

                    mean_x += pts[0,i]
                    mean_y += pts[1,i]
                    mean_z += pts[2,i]

                    i+=1
        
        mean_x /= float(i)
        mean_y /= float(i)
        mean_z /= float(i)
        
        xx=0.0
        xy=0.0
        xz=0.0
        yy=0.0
        yz=0.0  
        for i in range(0,n_good_pts):
            xx += (pts[0,i] - mean_x)*(pts[0,i] - mean_x)
            xy += (pts[0,i] - mean_x)*(pts[1,i] - mean_y)
            xz += (pts[0,i] - mean_x)*(pts[2,i] - mean_z)
            yy += (pts[1,i] - mean_y)*(pts[1,i] - mean_y)
            yz += (pts[1,i] - mean_y)*(pts[2,i] - mean_z)

        det = xx*yy - xy*xy
        if(det == 0.0):
            return

        a0 = (yy*xz - xy*yz) / det
        a1 = (xx*yz - xy*xz) / det

        error = 0.0

        # A*x + B*y + C*z = D 
        # n = [A,B,C]
        # z = a0 * x + a1 * y
        # 0 = a0 * x + a1 * y - z
        # D = 0, A = a0, B = a1, C = -1
        # n = [-a0,-a1,1]
        # theta_ = atan2(1,-a0)

        m = math.sqrt(a0*a0 + a1*a1 + 1)
        a0/=m
        a1/=m

        for i in range(0,n_good_pts):
            e = (pts[2,i] - mean_z) - (a0 * (pts[0,i] - mean_x) + a1 * (pts[1,i] - mean_y))
            error += e*e

        error /= float(n_good_pts)
        #if(error >0):
        #    error = math.log(error)
        output_roughness_map[x0,y0] = math.sqrt(error)

        x_angle = math.atan2(a0,1.0/m)
        y_angle = math.atan2(a1,1.0/m)

        output_slope_map_x[x0,y0] = x_angle
        output_slope_map_y[x0,y0] = y_angle

        pass

    @cuda.jit
    def __expand_binary(input_img, output_img, xy_size, r):
        """
            Assumes that both the input and output images contain only 0s and 1s
        """
        x, y = cuda.grid(2)
        if(x >= xy_size or y >= xy_size):
            return

        tmp_val = 0.0
        tmp_count = 0.0

        r_int = int(math.floor(r))

        for i in range(-r_int, r_int + 1):
            dy = int(math.floor(math.sqrt(r*r - i*i)))
            if(x + i < 0 or x+i >= xy_size):
                continue
            for j in range(-dy, dy + 1):
                if(y + j < 0 or y+j >= xy_size):
                    continue

                if(input_img[x+i, y+j] == 1):
                    output_img[x, y] = 1
                    return

        output_img[x, y] = 0

    @cuda.jit
    def __lowpass_binary(input_img, output_img, xy_size, filter_size, filter_fraction):
        """
            Assumes that both the input and output images contain only 0s and 1s
        """
        x, y = cuda.grid(2)
        if(x >= xy_size or y >= xy_size):
            return

        tmp_val = 0.0
        tmp_count = 0.0

        if(input_img[x, y] == 0):
            output_img[x, y] = 0
            return

        for i in range(-filter_size, filter_size + 1):
            if(x + i < 0 or x+i >= xy_size):
                continue
            for j in range(-filter_size, filter_size + 1):
                if(y + j < 0 or y+j >= xy_size):
                    continue
                tmp_val += input_img[x+i, y+j]*1.0
                tmp_count += 1.0

        if((tmp_val/tmp_count) >= filter_fraction):
            output_img[x, y] = 1
        else:
            output_img[x, y] = 0

    @cuda.jit
    def __convolve(input_img, output_img, kernel, xy_size, kernel_size):
        x, y = cuda.grid(2)
        if(x >= xy_size or y >= xy_size):
            return

        r = (kernel_size-1)/2

        tmp_val = 0.0

        for i in range(-r, r+1):
            x2 = i + r
            if(x + r < 0 or x+r >= xy_size):
                continue
            for j in range(-r, r+1):
                y2 = j + r
                if(y + r < 0 or y+r >= xy_size):
                    continue
                tmp_val += input_img[x+r, y+r] * kernel[x2, y2]

        output_img[x, y] = tmp_val

    @cuda.jit
    def __radar_combine_metrics(combined_metrics, combined_hit_count, combined_index_map, combined_origin, old_metrics, old_hit_count, old_index_map, old_origin, xy_size, z_size):
        x, y, z = cuda.grid(3)

        if(x >= xy_size or y >= xy_size or z >= z_size): # Check kernel bounds
            return

        # metrics: Average return strength, radar_point_count, grad x,y,z

        # Calculate offset between old and new maps

        dx = combined_origin[0] - old_origin[0]
        dy = combined_origin[1] - old_origin[1]
        dz = combined_origin[2] - old_origin[2]

        if((x + dx) >= xy_size or (y + dy) >= xy_size or (z + dz) >= z_size or (x+dx) < 0 or (y+dy) < 0 or (z+dz) < 0):
            return

        index = combined_index_map[int(
            x + y * xy_size + z * xy_size * xy_size)]
        index_old = old_index_map[int(
            (x + dx) + (y + dy) * xy_size + (z + dz) * xy_size * xy_size)]

        if(index < 0 or index_old < 0):
            return

        # combine average return strength

        # (S1 * N1 + S2 * N2) / (N1 + N2)

        mean_return = (combined_metrics[index,0] * combined_metrics[index,1] + old_metrics[index_old, 0] * old_metrics[index_old, 1]) / (combined_metrics[index,1] + old_metrics[index_old, 1])

        # combine gradients

        grad_x = (combined_metrics[index,2] * combined_metrics[index,1] + old_metrics[index_old, 2] * old_metrics[index_old, 1]) / (combined_metrics[index,1] + old_metrics[index_old, 1])
        grad_y = (combined_metrics[index,3] * combined_metrics[index,1] + old_metrics[index_old, 3] * old_metrics[index_old, 1]) / (combined_metrics[index,1] + old_metrics[index_old, 1])
        grad_z = (combined_metrics[index,4] * combined_metrics[index,1] + old_metrics[index_old, 4] * old_metrics[index_old, 1]) / (combined_metrics[index,1] + old_metrics[index_old, 1])

        # assign new values

        combined_metrics[index,0] = mean_return
        combined_metrics[index,1] = combined_metrics[index,1] + old_metrics[index_old, 1]
        combined_metrics[index,2] = grad_x
        combined_metrics[index,3] = grad_y
        combined_metrics[index,4] = grad_z

        combined_hit_count[index] = combined_hit_count[index] + old_hit_count[index_old]

        
        

    @cuda.jit
    def __combine_metrics(combined_metrics, combined_hit_count,combined_total_count,combined_min_height, combined_index_map, combined_origin, old_metrics, old_hit_count,old_total_count,old_min_height, old_index_map, old_origin, voxel_count, metrics_list, xy_size, z_size, num_metrics):
        x, y, z = cuda.grid(3)

        if(x >= xy_size or y >= xy_size or z >= z_size): # Check kernel bounds
            return

        # Calculate offset between old and new maps

        dx = combined_origin[0] - old_origin[0]
        dy = combined_origin[1] - old_origin[1]
        dz = combined_origin[2] - old_origin[2]

        if((x + dx) >= xy_size or (y + dy) >= xy_size or (z + dz) >= z_size or (x+dx) < 0 or (y+dy) < 0 or (z+dz) < 0):
            return

        index = combined_index_map[int(
            x + y * xy_size + z * xy_size * xy_size)]
        index_old = old_index_map[int(
            (x + dx) + (y + dy) * xy_size + (z + dz) * xy_size * xy_size)]

        if(index < 0 or index_old < 0):
            return

       

        
        ## Combine covariance
        
        #self.metrics_count = 10 # Mean: x, y, z; Covariance: xx, xy, xz, yy, yz, zz; Count
        #                               0  1  2               3   4   5   6   7   8

        
        # C = (n1 * C1 + n2 * C2 + 
        #   n1 * (mean_x1 - mean_x_combined) * (mean_y1 - mean_y_combined) + 
        #   n2 * (mean_x2 - mean_x_combined) * (mean_y2 - mean_y_combined)
        #   ) / (n1 + n2)

        mean_x_combined = (combined_metrics[index,0] * combined_metrics[index,9] + old_metrics[index_old, 0] * old_metrics[index_old, 9]) / (combined_metrics[index,9] + old_metrics[index_old, 9])

        mean_y_combined = (combined_metrics[index,1] * combined_metrics[index,9] + old_metrics[index_old, 1] * old_metrics[index_old, 9]) / (combined_metrics[index,9] + old_metrics[index_old, 9])

        mean_z_combined = (combined_metrics[index,2] * combined_metrics[index,9] + old_metrics[index_old, 2] * old_metrics[index_old, 9]) / (combined_metrics[index,9] + old_metrics[index_old, 9])

        # xx
        combined_metrics[index,3] =( combined_metrics[index,9] * combined_metrics[index,3] + old_metrics[index_old, 9] * old_metrics[index_old, 3] + 
            combined_metrics[index,9] * (combined_metrics[index,0] - mean_x_combined) * (combined_metrics[index,0] - mean_x_combined) + 
            old_metrics[index_old, 9] *  (old_metrics[index_old,0] - mean_x_combined) *  (old_metrics[index_old,0] - mean_x_combined)
            ) / (combined_metrics[index,9] + old_metrics[index_old, 9]) 

        # xy
        combined_metrics[index,4] =( combined_metrics[index,9] * combined_metrics[index,4] + old_metrics[index_old, 9] * old_metrics[index_old, 4] + 
            combined_metrics[index,9] * (combined_metrics[index,0] - mean_x_combined) * (combined_metrics[index,1] - mean_y_combined) + 
            old_metrics[index_old, 9] *  (old_metrics[index_old,0] - mean_x_combined) *  (old_metrics[index_old,1] - mean_y_combined)
            ) / (combined_metrics[index,9] + old_metrics[index_old, 9]) 
        # xz
        combined_metrics[index,5] =( combined_metrics[index,9] * combined_metrics[index,5] + old_metrics[index_old, 9] * old_metrics[index_old, 5] + 
            combined_metrics[index,9] * (combined_metrics[index,0] - mean_x_combined) * (combined_metrics[index,2] - mean_z_combined) + 
            old_metrics[index_old, 9] *  (old_metrics[index_old,0] - mean_x_combined) *  (old_metrics[index_old,2] - mean_z_combined)
            ) / (combined_metrics[index,9] + old_metrics[index_old, 9]) 

        # yy
        combined_metrics[index,6] =( combined_metrics[index,9] * combined_metrics[index,6] + old_metrics[index_old, 9] * old_metrics[index_old, 6] + 
            combined_metrics[index,9] * (combined_metrics[index,1] - mean_y_combined) * (combined_metrics[index,1] - mean_y_combined) + 
            old_metrics[index_old, 9] *  (old_metrics[index_old,1] - mean_y_combined) *  (old_metrics[index_old,1] - mean_y_combined)
            ) / (combined_metrics[index,9] + old_metrics[index_old, 9]) 

        # yz
        combined_metrics[index,7] =( combined_metrics[index,9] * combined_metrics[index,7] + old_metrics[index_old, 9] * old_metrics[index_old, 7] + 
            combined_metrics[index,9] * (combined_metrics[index,1] - mean_y_combined) * (combined_metrics[index,2] - mean_z_combined) + 
            old_metrics[index_old, 9] *  (old_metrics[index_old,1] - mean_y_combined) *  (old_metrics[index_old,2] - mean_z_combined)
            ) / (combined_metrics[index,9] + old_metrics[index_old, 9]) 
        
        # zz
        combined_metrics[index,8] =( combined_metrics[index,9] * combined_metrics[index,8] + old_metrics[index_old, 9] * old_metrics[index_old, 8] + 
            combined_metrics[index,9] * (combined_metrics[index,2] - mean_z_combined) * (combined_metrics[index,2] - mean_z_combined) + 
            old_metrics[index_old, 9] *  (old_metrics[index_old,2] - mean_z_combined) *  (old_metrics[index_old,2] - mean_z_combined)
            ) / (combined_metrics[index,9] + old_metrics[index_old, 9]) 

        ## Combine mean

        #x
        combined_metrics[index,0] = mean_x_combined
        #y
        combined_metrics[index,1] = mean_y_combined
        #z
        combined_metrics[index,2] = mean_z_combined

        ## Combine other metrics
        combined_metrics[index,9] = combined_metrics[index,9] + old_metrics[index_old, 9]
        combined_hit_count[index] = combined_hit_count[index] + old_hit_count[index_old]
        combined_total_count[index] = combined_total_count[index] + old_total_count[index_old]
        combined_min_height[index] = min(combined_min_height[index],old_min_height[index_old])

    @cuda.jit
    def __combine_old_metrics(combined_metrics, combined_hit_count,combined_total_count,combined_min_height, combined_index_map, combined_origin, old_metrics, old_hit_count,old_total_count,old_min_height, old_index_map, old_origin, voxel_count, metrics_list, xy_size, z_size, num_metrics):
        x, y, z = cuda.grid(3)

        if(x >= xy_size or y >= xy_size or z >= z_size):
            return

        dx = combined_origin[0] - old_origin[0]
        dy = combined_origin[1] - old_origin[1]
        dz = combined_origin[2] - old_origin[2]

        if((x + dx) >= xy_size or (y + dy) >= xy_size or (z + dz) >= z_size or (x+dx) < 0 or (y+dy) < 0 or (z+dz) < 0):
            return

        index = combined_index_map[int(
            x + y * xy_size + z * xy_size * xy_size)]
        index_old = old_index_map[int(
            (x + dx) + (y + dy) * xy_size + (z + dz) * xy_size * xy_size)]

        if(index < 0 or index_old < 0):
            return

        combined_hit_count[index] = combined_hit_count[index] + old_hit_count[index_old]
        combined_total_count[index] = combined_total_count[index] + old_total_count[index_old]
        combined_min_height[index] = min(combined_min_height[index],old_min_height[index_old])


    @cuda.jit
    def __combine_indices(combined_cell_count, combined_index_map, combined_origin, old_index_map, voxel_count, old_origin, xy_size, z_size):
        x, y, z = cuda.grid(3)

        if(x >= xy_size or y >= xy_size or z >= z_size):
            #print("bad index")
            return

        dx = combined_origin[0] - old_origin[0]
        dy = combined_origin[1] - old_origin[1]
        dz = combined_origin[2] - old_origin[2]

        if((x + dx) >= xy_size or (y + dy) >= xy_size or (z + dz) >= z_size or (x+dx) < 0 or (y+dy) < 0 or (z+dz) < 0):
            # print("oob")
            return

        index = int(x + y * xy_size + z * xy_size * xy_size)
        index_old = int((x + dx) + (y + dy) * xy_size +
                        (z + dz) * xy_size * xy_size)

        # If there is no data or empty data in the combined map and an occpuied voexl in the new map
        if(old_index_map[index_old] >= 0 and combined_index_map[index] <= -1):
            combined_index_map[index] = cuda.atomic.add(combined_cell_count, 0, 1)

        # if there is an empty cell in the old map and no data or empty data in the new map
        elif(old_index_map[index_old] < -1 and combined_index_map[index] <= -1):
            combined_index_map[index] += old_index_map[index_old] + 1

    @cuda.jit
    def __combine_old_indices(combined_cell_count, combined_index_map, combined_origin, old_index_map, voxel_count, old_origin, xy_size, z_size):
        x, y, z = cuda.grid(3)

        if(x >= xy_size or y >= xy_size or z >= z_size):
            #print("bad index")
            return

        dx = combined_origin[0] - old_origin[0]
        dy = combined_origin[1] - old_origin[1]
        dz = combined_origin[2] - old_origin[2]

        if((x + dx) >= xy_size or (y + dy) >= xy_size or (z + dz) >= z_size or (x+dx) < 0 or (y+dy) < 0 or (z+dz) < 0):
            # print("oob")
            return

        index = int(x + y * xy_size + z * xy_size * xy_size)
        index_old = int((x + dx) + (y + dy) * xy_size +
                        (z + dz) * xy_size * xy_size)

        # If there is no data in the combined map and an occpuied voexl in the new map
        if((old_index_map[index_old]) >= 0 and (combined_index_map[index] <= -1) and (combined_index_map[index] >= -11)):
            combined_index_map[index] = cuda.atomic.add(combined_cell_count, 0, 1)

        # if there is an empty cell in the old map and no data or empty data in the new map
        elif(old_index_map[index_old] < -1 and combined_index_map[index] <= -1):
            combined_index_map[index] += old_index_map[index_old] + 1

    @cuda.jit
    def __combine_2_maps(map1, map2):
        pass

    def __calculate_metrics_radar_master(self, pointcloud, point_count, hit_count, index_map, cell_count_cpu, origin):
        # Calculate the average radar return strength
        metric_blocks = self.blocks = math.ceil(self.xy_size*self.xy_size*self.z_size / self.threads_per_block)

        blockspergrid_cell = math.ceil(cell_count_cpu / self.threads_per_block_2D[0])
        blockspergrid_metric = math.ceil(metric_blocks / self.threads_per_block_2D[1])
        blockspergrid = (blockspergrid_cell, blockspergrid_metric)

        metrics = cuda.device_array([cell_count_cpu,self.radar_metrics_count])
        self.__init_2D_array[blockspergrid, self.threads_per_block_2D](metrics,0.0,cell_count_cpu,self.radar_metrics_count)

        calculate_blocks = ( int(np.ceil(point_count/self.threads_per_block)))
        self.__calculate_return_intensity[calculate_blocks, self.threads_per_block](
        self.xy_resolution, self.z_resolution, self.xy_size, self.z_size, self.min_distance, index_map, pointcloud, metrics, point_count, origin)





        blockspergrid_cell = math.ceil(cell_count_cpu / self.threads_per_block)
        self.__normalize_radar[blockspergrid_cell,self.threads_per_block](hit_count,metrics,cell_count_cpu)

        blockspergrid_xy = math.ceil(
            self.xy_size / self.threads_per_block_3D[0])
        blockspergrid_z = math.ceil(self.z_size / self.threads_per_block_3D[2])
        blockspergrid = (blockspergrid_xy, blockspergrid_xy, blockspergrid_z)

        self.__calculate_radar_gradients[blockspergrid, self.threads_per_block_3D](
            self.xy_resolution, self.z_resolution, self.xy_size, self.z_size, cell_count_cpu, index_map, metrics
        )

        return metrics



    def __calculate_metrics_master(self, pointcloud, point_count, count, index_map, cell_count_cpu, origin):
        # print("mean")
        #self.metrics_count = 10 # Mean: x, y, z; Covariance: xx, xy, xz, yy, yz, zz; Count

        metric_blocks = self.blocks = math.ceil(self.xy_size*self.xy_size*self.z_size / self.threads_per_block)

        blockspergrid_cell = math.ceil(cell_count_cpu / self.threads_per_block_2D[0])
        blockspergrid_metric = math.ceil(metric_blocks / self.threads_per_block_2D[1])
        blockspergrid = (blockspergrid_cell, blockspergrid_metric)

        metrics = cuda.device_array([cell_count_cpu,self.metrics_count])
        self.__init_2D_array[blockspergrid, self.threads_per_block_2D](metrics,0.0,cell_count_cpu,self.metrics_count)


        #print("min height")
        min_height = cuda.device_array([cell_count_cpu*3], dtype=np.float32)
        self.__init_1D_array[math.ceil(cell_count_cpu*3/self.threads_per_block),self.threads_per_block](min_height,1,cell_count_cpu*3)


        #print("calc blocks")
        calculate_blocks = ( int(np.ceil(point_count/self.threads_per_block)))
        
        #print("calc mean")
        
        self.__calculate_mean[calculate_blocks, self.threads_per_block](
            self.xy_resolution, self.z_resolution, self.xy_size, self.z_size, self.min_distance, index_map, pointcloud, metrics, point_count, origin, self.xy_eigen_dist, self.z_eigen_dist)
        
        #print("norm")
        normalize_blocks = ( int(np.ceil(cell_count_cpu/self.threads_per_block_2D[0])), int(np.ceil(3/self.threads_per_block_2D[0])) )

        self.__normalize_mean[normalize_blocks,self.threads_per_block_2D](metrics,cell_count_cpu)
        #print("other")
        
        
        self.__calculate_covariance[calculate_blocks,self.threads_per_block](

            self.xy_resolution, self.z_resolution, self.xy_size, self.z_size, self.min_distance, index_map, pointcloud, count, metrics, point_count, origin, self.xy_eigen_dist, self.z_eigen_dist
            
                )
        
        normalize_blocks = ( int(np.ceil(cell_count_cpu/self.threads_per_block_2D[0])), int(np.ceil(6/self.threads_per_block_2D[0])) )


        self.__normalize_covariance[normalize_blocks,self.threads_per_block_2D](metrics,cell_count_cpu)

        self.__calculate_min_height[calculate_blocks, self.threads_per_block](
            self.xy_resolution, self.z_resolution, self.xy_size, self.z_size, self.min_distance, index_map, pointcloud, min_height, point_count, origin)
        
        #print("return")

        return metrics, min_height

    @cuda.jit
    def __transform_pointcloud(points, transform, point_count):
        i = cuda.grid(1)
        if(i < point_count):
            pt = numba.cuda.local.array(3, "f8")
            pt[0] = points[i, 0] * transform[0, 0] + points[i, 1] * \
                transform[0, 1] + points[i, 2] * \
                transform[0, 2] + transform[0, 3]
            pt[1] = points[i, 0] * transform[1, 0] + points[i, 1] * \
                transform[1, 1] + points[i, 2] * \
                transform[1, 2] + transform[1, 3]
            pt[2] = points[i, 0] * transform[2, 0] + points[i, 1] * \
                transform[2, 1] + points[i, 2] * \
                transform[2, 2] + transform[2, 3]

            points[i, 0] = pt[0]
            points[i, 1] = pt[1]
            points[i, 2] = pt[2]

    @cuda.jit
    def __point_2_map(xy_resolution, z_resolution, xy_size, z_size, min_distance, points, hit_count, total_count, point_count, ego_position, origin, ray_trace):
        i = cuda.grid(1)
        if(i < point_count):

            if math.isnan(points[i, 0]):
                return

            d2 = points[i, 0]*points[i, 0] + points[i, 1] * \
                points[i, 1] + points[i, 2]*points[i, 2]

            if(d2 < min_distance*min_distance):
                return

            oob = False

            x_index = math.floor((points[i, 0] / xy_resolution) - origin[0])
            if(x_index < 0 or x_index >= xy_size):
                oob = True

            y_index = math.floor((points[i, 1] / xy_resolution) - origin[1])
            if(y_index < 0 or y_index >= xy_size):
                oob = True

            z_index = math.floor((points[i, 2] / z_resolution) - origin[2])
            if(z_index < 0 or z_index >= z_size):
                oob = True

            if not oob:
                # get the index of the hit
                index = x_index + y_index*xy_size + z_index*xy_size*xy_size
                # update the hit count for the index
                cuda.atomic.add(hit_count, index, 1)
                cuda.atomic.add(total_count, index, 1)


            # Trace the ray
            if(ray_trace):
                pt = numba.cuda.local.array(3, numba.float32)
                end = numba.cuda.local.array(3, numba.float32)
                slope = numba.cuda.local.array(3, numba.float32)

                pt[0] = ego_position[0] / xy_resolution
                pt[1] = ego_position[1] / xy_resolution
                pt[2] = ego_position[2] / z_resolution

                end[0] = points[i, 0] / xy_resolution
                end[1] = points[i, 1] / xy_resolution
                end[2] = points[i, 2] / z_resolution

                slope[0] = end[0] - pt[0]
                slope[1] = end[1] - pt[1]
                slope[2] = end[2] - pt[2]

                ray_length = math.sqrt(
                    slope[0]*slope[0] + slope[1]*slope[1] + slope[2]*slope[2])

                slope[0] = slope[0] / ray_length
                slope[1] = slope[1] / ray_length
                slope[2] = slope[2] / ray_length

                slope_max = max(abs(slope[0]), max(abs(slope[1]), abs(slope[2])))

                slope_index = 0

                if(slope_max == abs(slope[1])):
                    slope_index = 1
                if(slope_max == abs(slope[2])):
                    slope_index = 2

                length = 0
                direction = slope[slope_index]/abs(slope[slope_index])
                while (length < ray_length - 1):
                    pt[slope_index] += direction
                    pt[(slope_index + 1) % 3] += slope[(slope_index + 1) %
                                                    3] / abs(slope[slope_index])
                    pt[(slope_index + 2) % 3] += slope[(slope_index + 2) %
                                                    3] / abs(slope[slope_index])

                    x_index = math.floor(pt[0] - origin[0])
                    if(x_index < 0 or x_index >= xy_size):
                        return

                    y_index = math.floor(pt[1] - origin[1])
                    if(y_index < 0 or y_index >= xy_size):
                        return

                    z_index = math.floor(pt[2] - origin[2])
                    if(z_index < 0 or z_index >= z_size):
                        return
                        
                    index = x_index + y_index*xy_size + z_index*xy_size*xy_size

                    cuda.atomic.add(total_count, index, 1)

                    length += abs(1.0/slope[slope_index])

    @cuda.jit
    def __assign_indices(hit_count, miss_count, index_map, cell_count, voxel_count):
        i = cuda.grid(1)
        if(i < voxel_count):
            if(hit_count[i] > 0):
                index_map[i] = cuda.atomic.add(cell_count, 0, 1)
            else:
                index_map[i] = - miss_count[i] - 1

    @cuda.jit
    def __move_data(old, new, index_map, voxel_count):
        i = cuda.grid(1)
        if(i < voxel_count):
            if(index_map[i] >= 0):
                new[index_map[i]] = old[i]
    @cuda.jit
    def __calculate_radar_gradients(xy_resolution, z_resolution, xy_size, z_size, cell_count, index_map, metrics):
        # Average return strength, radar_point_count, grad x,y,z
        x,y,z = cuda.grid(3)
        if(x >= xy_size or y >= xy_size or z >= z_size):
            return
        
        i = index_map[x + y*xy_size + z*xy_size*xy_size ]

        if( i<0 ):
            return
        
        # grad x
        left_i = -1
        left = 0.0
        if(x > 0):
            left_i = index_map[(x-1) + y*xy_size + z*xy_size*xy_size ]
            if (left_i >= 0):
                left = metrics[left_i,0]

        right_i = -1
        right = 0.0
        if(x < xy_size -1):
            right_i = index_map[(x+1) + y*xy_size + z*xy_size*xy_size ]
            if (right_i >= 0):
                right = metrics[right_i,0]

        
        metrics[i,2] = right - left
    
        # grad y
        back_i = -1
        back = 0.0
        if(y > 0):
            back_i = index_map[x + (y-1)*xy_size + z*xy_size*xy_size ]
            if (back_i >= 0):
                back = metrics[back_i,0]

        front_i = -1
        front = 0.0
        if(y < xy_size -1):
            front_i = index_map[x + (y+1)*xy_size + z*xy_size*xy_size ]
            if (front_i >= 0):
                front = metrics[front_i,0]

        
        metrics[i,3] = front - back

        # grad z
        down_i = -1
        down = 0.0
        if(z > 0):
            down_i = index_map[x + y*xy_size + (z-1)*xy_size*xy_size ]
            if (down_i >= 0):
                down = metrics[down_i,0]

        up_i = -1
        up = 0.0
        if(z < z_size -1):
            up_i = index_map[x + y*xy_size + (z+1)*xy_size*xy_size ]
            if (up_i >= 0):
                up = metrics[up_i,0]

        
        metrics[i,4] = up - down


    @cuda.jit
    def __calculate_return_intensity(xy_resolution, z_resolution, xy_size, z_size, min_distance, index_map, points, metrics, point_count, origin):
        i = cuda.grid(1)
        if(i < point_count):
            
            if math.isnan(points[i, 0]):
                return

            d2 = points[i, 0]*points[i, 0] + points[i, 1] * \
                points[i, 1] + points[i, 2]*points[i, 2]

            if(d2 < min_distance*min_distance):
                return

            x_index = math.floor((points[i, 0]/xy_resolution) - origin[0])
            y_index = math.floor((points[i, 1]/xy_resolution) - origin[1])
            z_index = math.floor((points[i, 2]/z_resolution) - origin[2])


            if(x_index < 0 or x_index >= xy_size):
                return


            if(y_index < 0 or y_index >= xy_size):
                return


            if(z_index < 0 or z_index >= z_size):
                return

            index = index_map[int( x_index + y_index*xy_size + z_index*xy_size*xy_size )]

            if index <0 :
                    return

            cuda.atomic.add(metrics, (index,0), points[i, 3])


    @cuda.jit
    def __calculate_mean(xy_resolution, z_resolution, xy_size, z_size, min_distance, index_map, points, metrics, point_count, origin, xy_eigen_dist, z_eigen_dist):
        i = cuda.grid(1)
        if(i < point_count):

            d2 = points[i, 0]*points[i, 0] + points[i, 1] * \
                points[i, 1] + points[i, 2]*points[i, 2]

            if(d2 < min_distance*min_distance):
                return

            local_point = cuda.local.array(shape=3, dtype=numba.float64)

            x_index_base = math.floor((points[i, 0]/xy_resolution) - origin[0])
            y_index_base = math.floor((points[i, 1]/xy_resolution) - origin[1])
            z_index_base = math.floor((points[i, 2]/z_resolution) - origin[2])

            for x_index in range(x_index_base - xy_eigen_dist,  x_index_base + 1 + xy_eigen_dist):

                if(x_index < 0 or x_index >= xy_size):
                    continue

                for y_index in range(y_index_base - xy_eigen_dist, y_index_base + 1 + xy_eigen_dist ):

                    if(y_index < 0 or y_index >= xy_size):
                        continue

                    for z_index in range(z_index_base - z_eigen_dist, z_index_base + 1 + z_eigen_dist):

                        if(z_index < 0 or z_index >= z_size):
                            continue

                        

                        local_point[0] = (points[i, 0]/xy_resolution) - origin[0] - x_index
                        local_point[1] = (points[i, 1]/xy_resolution) - origin[1] - y_index
                        local_point[2] = (points[i, 2]/z_resolution) - origin[2] - z_index


                        index = index_map[int( x_index + y_index*xy_size + z_index*xy_size*xy_size )]

                        if index <0 :
                                continue


                        cuda.atomic.add(metrics, (index,0), local_point[0])
                        cuda.atomic.add(metrics, (index,1), local_point[1])
                        cuda.atomic.add(metrics, (index,2), local_point[2])

                        cuda.atomic.add(metrics,(index,9),1.0) # update count for this voxel

    @cuda.jit
    def __normalize_mean(metrics, cell_count):
        i, j = cuda.grid(2)
        if(i>=cell_count):
            return
        if(j>=3):
            return

        metrics[i,j] = metrics[i,j]/metrics[i,9]

    @cuda.jit
    def __normalize_radar(hit_count, metrics, cell_count):
        i = cuda.grid(1)
        if(i>=cell_count):
            return

        metrics[i,0] = metrics[i,0]/hit_count[i]
        metrics[i,1] = hit_count[i]

    @cuda.jit
    def __calculate_covariance(xy_resolution, z_resolution, xy_size, z_size, min_distance, index_map, points, count, metrics, point_count, origin, xy_eigen_dist, z_eigen_dist):
        i = cuda.grid(1)
        if(i < point_count):

            d2 = points[i, 0]*points[i, 0] + points[i, 1] * \
                points[i, 1] + points[i, 2]*points[i, 2]

            if(d2 < min_distance*min_distance):
                return

            local_point = cuda.local.array(shape=3, dtype=numba.float64)

            x_index_base = math.floor((points[i, 0]/xy_resolution) - origin[0])
            y_index_base = math.floor((points[i, 1]/xy_resolution) - origin[1])
            z_index_base = math.floor((points[i, 2]/z_resolution) - origin[2])

            for x_index in range(x_index_base - xy_eigen_dist,  x_index_base + 1 + xy_eigen_dist):

                if(x_index < 0 or x_index >= xy_size):
                    continue

                for y_index in range(y_index_base - xy_eigen_dist, y_index_base + 1 + xy_eigen_dist ):

                    if(y_index < 0 or y_index >= xy_size):
                        continue

                    for z_index in range(z_index_base - z_eigen_dist, z_index_base + 1 + z_eigen_dist):

                        if(z_index < 0 or z_index >= z_size):
                            continue

                        

                        local_point[0] = (points[i, 0]/xy_resolution) - origin[0] - x_index
                        local_point[1] = (points[i, 1]/xy_resolution) - origin[1] - y_index
                        local_point[2] = (points[i, 2]/z_resolution) - origin[2] - z_index


                        index = index_map[int( x_index + y_index*xy_size + z_index*xy_size*xy_size )]

                        if index <0 :
                                continue

                        # xx
                        cov_xx = (local_point[0] - metrics[index,0])*(local_point[0] - metrics[index,0])
                        cuda.atomic.add(metrics,(index,3),cov_xx)
                        # xy
                        cov_xy = (local_point[0] - metrics[index,0])*(local_point[1] - metrics[index,1])
                        cuda.atomic.add(metrics,(index,4),cov_xy)
                        # xz
                        cov_xz = (local_point[0] - metrics[index,0])*(local_point[2] - metrics[index,2])
                        cuda.atomic.add(metrics,(index,5),cov_xz)
                        # yy
                        cov_yy = (local_point[1] - metrics[index,1])*(local_point[1] - metrics[index,1])
                        cuda.atomic.add(metrics,(index,6),cov_yy)
                        # yz
                        cov_yz = (local_point[1] - metrics[index,1])*(local_point[2] - metrics[index,2])
                        cuda.atomic.add(metrics,(index,7),cov_yz)
                        # zz
                        cov_zz = (local_point[2] - metrics[index,2])*(local_point[2] - metrics[index,2])
                        cuda.atomic.add(metrics,(index,8),cov_zz)

                        
            
    @cuda.jit
    def __normalize_covariance(metrics, cell_count):
        i, j = cuda.grid(2)
        if(i>=cell_count):
            return
        if(j>=6):
            return

        if(metrics[i,9] <= 0):
            metrics[i,j+3] = 0
            return

        metrics[i,j+3] = metrics[i,j+3]/metrics[i,9]

    @cuda.jit
    def __calculate_min_height(xy_resolution, z_resolution, xy_size, z_size, min_distance, index_map, points, min_height, point_count, origin):
        i = cuda.grid(1)
        if(i < point_count):

            d2 = points[i, 0]*points[i, 0] + points[i, 1] * points[i, 1] + points[i, 2]*points[i, 2]

            if(d2 < min_distance*min_distance):
                return

            x_index = math.floor((points[i, 0]/xy_resolution) - origin[0])
            if(x_index < 0 or x_index >= xy_size):
                return

            y_index = math.floor((points[i, 1]/xy_resolution) - origin[1])
            if(y_index < 0 or y_index >= xy_size):
                return

            z_index = math.floor((points[i, 2]/z_resolution) - origin[2])
            if(z_index < 0 or z_index >= z_size):
                return

            local_point = cuda.local.array(shape=3, dtype=numba.float64)
            

            local_point[0] = (points[i, 0]/xy_resolution) - origin[0] - x_index
            local_point[1] = (points[i, 1]/xy_resolution) - origin[1] - y_index
            local_point[2] = (points[i, 2]/z_resolution) - origin[2] - z_index

            
            index = index_map[int(x_index + y_index*xy_size + z_index*xy_size*xy_size)]

            cuda.atomic.min(min_height, index, local_point[2])

    @cuda.jit
    def __calculate_eigenvalues(voxels_eigenvalues,metrics,cell_count):
        i = cuda.grid(1)

        if(i >= cell_count):
            return

        # Mean: x, y, z; Covariance: xx, xy, xz, yy, yz, zz
        #       0  1  2               3   4   5   6   7   8

        # [xx   xy   xz]
        # [xy   yy   yz]
        # [xz   yz   zz]

        xx = metrics[i,3]
        xy = metrics[i,4]
        xz = metrics[i,5]
        yy = metrics[i,6]
        yz = metrics[i,7]
        zz = metrics[i,8]

        p1 = xy*xy + xz*xz + yz*yz  
        q = (xx + yy + zz ) / 3.0
        
        if (p1 == 0): # diagonal matrix

            voxels_eigenvalues[i,0] = max(xx,max(yy,zz))
            
            voxels_eigenvalues[i,2] = min(xx,min(yy,zz))

            voxels_eigenvalues[i,1] = 3.0 * q - voxels_eigenvalues[i,0] - voxels_eigenvalues[i,2]


        else:

            
            p2 = (xx - q)*(xx - q) + (yy - q)*(yy - q) + (zz - q)*(zz - q) + 2.0 * p1
            p = math.sqrt(p2 / 6.0)
            
            B = numba.cuda.local.array(shape=6, dtype=numba.float64)

            B[0] = (xx - q)/p
            B[1] = xy / p
            B[2] = xz / p
            B[3] = (yy - q)/p
            B[4] = yz / p
            B[5] = (zz - q)/p


            r =  B[0] * ( B[3] * B[5] - B[4] * B[4] ) - B[1] * ( B[1] * B[5] - B[4] * B[2] ) + B[2] * ( B[1] * B[4] - B[3] * B[2] )
            r = r / 2

            phi = 0.0
            if (r <= -1):
                phi = math.pi / 3.0
            elif (r >= 1):
                phi = 0.0
            else:
                phi = math.acos(r) / 3.0

            voxels_eigenvalues[i,0] = q + 2.0 * p * math.cos(phi)
            voxels_eigenvalues[i,2] = q + 2.0 * p * math.cos(phi + (2.0*math.pi/3.0))
            voxels_eigenvalues[i,1] = 3.0 * q - voxels_eigenvalues[i,0] - voxels_eigenvalues[i,2] 



    @cuda.jit
    def __init_1D_array(array,value,length):
        i = cuda.grid(1)
        if(i>=length):
            return
        array[i] = value
    
    @cuda.jit
    def __init_2D_array(array,value,width,height):
        x,y = cuda.grid(2)
        if(x>=width or y>=height):
            return
        array[x,y] = value
