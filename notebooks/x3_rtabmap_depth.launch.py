"""
x3_rtabmap_depth.launch.py
---------------------------
Yahboom ROSMASTER X3 PLUS — ROS2 Humble
 
RTAB-Map RGB-D SLAM launch file.
Brings up the four-node RTAB-Map pipeline that consumes synchronized
color and depth frames from the Astra Pro Plus camera and produces
visual odometry, a dense 3D point-cloud map, and a 2D occupancy grid
(via Grid/3D + RGBD/CreateOccupancyGrid). Provides the 3D-reconstruction
half of the SLAM layer described in slam_module.py / lidar_slam_planner.py.
 
Nodes launched (in order):
  1. rgbd_sync       — time-aligns color + depth streams (20ms window)
  2. rgbd_odometry   — visual odometry from synchronized RGB-D frames
  3. rtabmap          — SLAM backend; builds the 3D map + occupancy grid
  4. rtabmap_viz       — RViz-like visualization window for the 3D map
 
All four nodes use approx_sync=True with a 20ms window, which was
found necessary because the color and depth streams on this hardware
are not perfectly synchronized.
 
Run order (must be launched after the camera and base are up):
  Terminal 1: ros2 launch yahboomcar_description display_X3.launch.py
  Terminal 2: ros2 run yahboomcar_bringup Mcnamu_driver_X3
  Terminal 3: ros2 run yahboomcar_base_node base_node_X3
  Terminal 4: ros2 launch astra_camera astro_pro_plus.launch.xml
  Terminal 5: ros2 launch ~/x3_rtabmap_depth.launch.py
 
Topics:
  Subscribed : /camera/color/image_raw    (Astra Pro Plus — color stream)
               /camera/depth/image_raw    (Astra Pro Plus — depth stream)
               /camera/color/camera_info  (Astra Pro Plus — color calibration)
               /odom                      (rtabmap, rtabmap_viz — wheel odometry)
  Published  : /rgbd_image                (rgbd_sync — synced RGB-D pair)
               tf: odom -> base_footprint (rgbd_odometry, publish_tf=True)
               3D point-cloud map + 2D occupancy grid (rtabmap)
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        Node(
            package='rtabmap_sync',
            executable='rgbd_sync',
            name='rgbd_sync',
            output='screen',
            parameters=[{
                'approx_sync': True,
                'approx_sync_max_interval': 0.02,
                'queue_size': 50
            }],
            remappings=[
                ('rgb/image',       '/camera/color/image_raw'),
                ('depth/image',     '/camera/depth/image_raw'),
                ('rgb/camera_info', '/camera/color/camera_info'),
                ('rgbd_image', '/rgbd_image')
            ]
        ),

        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            name='rgbd_odometry',
            output='screen',
            parameters=[{
                'frame_id': 'base_footprint',
                'odom_frame_id': 'odom',
                'publish_tf': True,
                'approx_sync': True,
                'approx_sync_max_interval': 0.02,
                'queue_size': 50,
                'Vis/MinInliers': '8',
                'Odom/ResetCountdown': '0'
            }],
            remappings=[
                ('rgb/image',       '/camera/color/image_raw'),
                ('depth/image',     '/camera/depth/image_raw'),
                ('rgb/camera_info', '/camera/color/camera_info'),
            ],
        ),

        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=[{
                'frame_id': 'base_footprint',
                'subscribe_rgbd': True,
                'subscribe_depth': False,
                'subscribe_rgb': False,
                'subscribe_scan': False,
                'subscribe_scan_cloud': False,
                'approx_sync': True,
                'queue_size': 50,
                'Grid/3D': True,
                'Grid/FromDepth': True,
                'RGBD/CreateOccupancyGrid': True,
                'Reg/Force3DoF': 'false'
            }],
            remappings=[
                ('rgbd_image', '/rgbd_image'),
                ('odom', '/odom')
            ],
            arguments=['-d']
        ),

        Node(
            package='rtabmap_viz',
            executable='rtabmap_viz',
            name='rtabmap_viz',
            output='screen',
            parameters=[{
                'frame_id': 'base_footprint',
                'subscribe_rgbd': True,
                'subscribe_depth': False,
                'subscribe_rgb': False,
                'subscribe_scan': False,
                'approx_sync': True
            }],
            remappings=[
                ('rgbd_image', '/rgbd_image'),
                ('odom', '/odom')
            ]
        ),
    ])