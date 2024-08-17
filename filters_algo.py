import numpy
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor, LinearRegression
import pptk

import open3d as o3d


class LidarFilter:
    def __init__(self, ground_model=RANSACRegressor(max_trials=200, residual_threshold=10),
                 cluster_model=DBSCAN(eps=20, min_samples=15)):
        self.ground_model = ground_model
        self.cluster_model = cluster_model

        self.filter_tests = [self.length_test, self.width_test, self.height_test, self.min_height_test,
                             self.number_of_pts_test, self.cone_shape_egv_test]
        self.points = None
        self.ground_points = None
        self.clusters_list = None
        self.height_ground = None

        self.rot_matrix = None

    def length_test(self, points, min_val=5, max_val=60):
        x_coords = points[:, 0]
        if min_val < np.abs(np.max(x_coords) - np.min(x_coords)) < max_val:
            return True
        return False

    def width_test(self, points, min_val=5, max_val=70):
        y_coords = points[:, 1]
        if min_val < np.abs(np.max(y_coords) - np.min(y_coords)) < max_val:
            return True
        return False  # didn't pass the filter, then return False

    def height_test(self, points, min_val=10, max_val=80):
        z_coords = points[:, 2]
        if min_val < np.abs(np.max(z_coords) - np.min(z_coords)) < max_val:
            return True
        return False

    def min_height_test(self, points, min_diff=20):
        x_coords = points[:, 0]
        z_coords = points[:, 2]
        if np.min(z_coords) < (self.height_ground + min_diff) or \
                (np.min(x_coords) > 1250 and np.min(z_coords) < (self.height_ground + (min_diff * 2))):
            return True
        return False

    def number_of_pts_test(self, points, min_pts=10, max_pts=1500):
        if min_pts < points.shape[0] < max_pts:
            return True
        return False

    def cone_shape_egv_test(self, points):
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        if eigenvalues[0] > eigenvalues[1] * 1.5 and eigenvalues[0] > eigenvalues[2] * 2.5:
            return True
        else:
            return False

    def rotation_matrix_from_plane(self, plane_coeffs):
        """
        Generates a 3D rotation matrix based on the coefficients of a plane.

        Args:
            plane_coeffs (tuple): Coefficients of the plane equation (a, b, c, d).

        Returns:
            np.ndarray: 3x3 rotation matrix.
        """
        # Extract plane coefficients
        a, b, c, d = plane_coeffs

        # Normalize the normal vector
        n_norm = np.linalg.norm([a, b, c])
        n_normalized = np.array([a, b, c]) / n_norm

        # Construct skew-symmetric matrix K
        K = np.array([[0, -n_normalized[2], n_normalized[1]],
                      [n_normalized[2], 0, -n_normalized[0]],
                      [-n_normalized[1], n_normalized[0], 0]])

        # Rotation matrix
        R = np.eye(3) + K + 0.5 * np.dot(K, K)
        self.rot_matrix = R

    def filter_fov(self):
        # Filter the points that are out of the FOV
        # filter x axis (how far the car can see front and back)
        # in Centimeters
        # self.points = self.points[self.points[:, 0] > 200]
        # self.points = self.points[self.points[:, 0] < 3500]
        #
        # # filter y axis (how far the car can see left and right)
        # self.points = self.points[self.points[:, 1] > -2000]
        # self.points = self.points[self.points[:, 1] < 2000]

        mask1 = (self.points[:, 0] > 100) & (self.points[:, 0] < 1500) & (self.points[:, 1] > -1000) & (self.points[:, 1] < 1000)
        mask2 = (self.points[:, 0] >= 1000) & (self.points[:, 0] < 3500) & (self.points[:, 1] > -1000) & (self.points[:, 1] < 1000)
        combined_mask = mask1 | mask2
        self.points = self.points[combined_mask]
        # filter z axis (how far the car can see up and down)

        self.points = self.points[self.points[:, 2] > -100]
        self.points = self.points[self.points[:, 2] < 80]


    def filter_ground(self, pointcloud, height_threshold=15, model_option="RANSAC"):
        # Extract the x, y, and z coordinates of the points
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

        self.ground_model.fit(np.column_stack((x, y)), z)
        self.height_ground = self.ground_model.estimator_.intercept_ + height_threshold

        # Calculate the height of each point relative to the ground plane
        z_pred = self.ground_model.predict(np.column_stack((x, y)))
        dz = z - z_pred

        # Filter out points that are below the height threshold
        ground_points = pointcloud[dz < np.min(dz) + height_threshold]
        non_ground_points = pointcloud[dz > np.min(dz) + height_threshold]
        self.points, self.ground_points = non_ground_points, ground_points

    def filter_clusters(self):
        self.points = np.empty((1, 3))  # initialize the final points list.
        filtered_clusters = []
        vaild_cluster = True
        for cluster_num, cluster in enumerate(self.clusters_list):
            for filter_test in self.filter_tests:
                if not filter_test(cluster):  # if the cluster didn't pass the test then continue to next cluster
                    # print("cluster number ", cluster_num, " didn't pass the filter test ", filter_test)
                    vaild_cluster = False
                    break
            if vaild_cluster:
                # if the cluster passed all the filter tests then add it to the final points list and to the clusters
                self.points = np.concatenate((self.points, cluster))
                filtered_clusters.append(cluster)
            vaild_cluster = True
        # self.clusters_list = np.array(filtered_clusters)

    def points_to_clusters(self):
        cluster_labels = self.cluster_model.fit_predict(self.points)
        # Create a list to hold the points in each cluster
        unique_labels = np.unique(cluster_labels)
        cluster_point_lists = []
        # Iterate over the unique cluster labels
        for label in unique_labels:
            # Find the indices of the points in the current cluster
            cluster_indices = np.where(cluster_labels == label)[0]
            # Get the points in the current cluster
            cluster_points = self.points[cluster_indices]
            # Add the points to the list of cluster points
            cluster_point_lists.append(cluster_points)
        self.clusters_list = cluster_point_lists

    def run(self, points):
        self.points = points
        self.filter_fov()
        ground_plane, non_ground_points = ground_plane_finder(self.points, visual=False)
        self.rotation_matrix_from_plane(ground_plane)

        #TEST
        po = np.asarray(non_ground_points.points)
        temp = np.matmul(po , self.rot_matrix)
        v = pptk.viewer(temp)
        v.set(point_size=0.03, show_grid=True)
        ###
        self.points_to_clusters()
        # self.filter_clusters()

        # self.points are the final points after filtering
        # self.clusters_list are the final clusters after filtering (contains all the self.points)
        # self.ground_points are the ground points after filtering only the ground for debugging.
        return self.points, self.ground_points, self.clusters_list


def plane_finder_point_cloud(point_cloud_data, thresh=0.1, ransac_n=10):
    # Convert numpy array to open3d point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)

    # Downsample the point cloud (optional)
    # voxel_size = 3 #0.05  # adjust this parameter as needed
    # downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size)

    # Estimate normals
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))  # radius = 0.1
    # Segment ground plane
    ground_plane_model, inliers = point_cloud.segment_plane(
        distance_threshold=thresh,
        ransac_n=ransac_n,
        num_iterations=2000
    )
    # Extract ground points and non-ground points
    non_ground_points = point_cloud.select_by_index(inliers, invert=True)
    ground_points = point_cloud.select_by_index(inliers)
    return inliers, ground_plane_model, non_ground_points, ground_points


def visual_pcd(ground_points, non_ground_points):
    ground_points_np = np.asarray(ground_points.points)
    non_ground_points_np = np.asarray(non_ground_points.points)

    # Combine ground and non-ground points
    all_points_np = np.vstack((ground_points_np, non_ground_points_np))

    # Create a colors array with red for ground points and green for non-ground points
    colors = np.vstack((np.repeat([[0, 1, 0]], len(ground_points_np), axis=0),  # Red for ground points
                        np.repeat([[0.7, 0, 0]], len(non_ground_points_np), axis=0)))  # Green for non-ground points
    v = pptk.viewer(all_points_np)
    v.attributes(colors)
    v.set(point_size=0.03, show_grid=False)
    return


def is_ground_plane(plane_model, angle_threshold=10):
    # The normal vector of the XY plane pointing upwards is (0, 0, 1)
    xy_plane_normal = np.array([0, 0, 1])

    # Extract the normal vector of the detected plane
    detected_plane_normal = plane_model[:3]

    # Calculate the angle between the two vectors in degrees
    angle = np.arccos(np.dot(detected_plane_normal, xy_plane_normal) /
                      (np.linalg.norm(detected_plane_normal) * np.linalg.norm(xy_plane_normal)))
    angle_degrees = np.degrees(angle)

    # Check if the angle is within the threshold
    return angle_degrees <= angle_threshold


def remove_close_points(plane, point_cloud, distance):
    """
    Removes points from the point cloud that are closer than 'distance' to the 'plane'.

    Parameters:
    plane (array-like): The plane coefficients [a, b, c, d] for the equation ax + by + cz + d = 0.
    point_cloud (open3d.geometry.PointCloud): The point cloud object.
    distance (float): The distance threshold.

    Returns:
    open3d.cpu.pybind.geometry.PointCloud: The filtered point cloud.
    """
    # Extract the plane coefficients
    a, b, c, d = plane

    # Convert the PointCloud to a NumPy array
    points = np.asarray(point_cloud.points)

    # Calculate the distances of all points from the plane
    distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)

    # Create a mask for points that are further than 'distance' from the plane
    mask = distances > distance

    # Use the mask to filter the point cloud
    filtered_points = points[mask]

    # Create a new PointCloud object for the filtered points
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_point_cloud


def ground_plane_finder(point_cloud_data, visual=False, thresh=0.1, ransac_n=10):
    i = 0
    while True:
        i += 1
        inliers, ground_plane_model, non_ground_points, ground_points = plane_finder_point_cloud(point_cloud_data,
                                                                                                 thresh,
                                                                                                 ransac_n)
        if is_ground_plane(ground_plane_model):
            non_ground_points2 = remove_close_points(ground_plane_model, non_ground_points, distance=8)
            if visual:
                visual_pcd(ground_points, non_ground_points2)
            temp = len(ground_points.points) + len(non_ground_points.points) - len(non_ground_points2.points)
            print("ground found after " + str(i) + " tries, filtered " + str(temp) + " ground points")
            return ground_plane_model, non_ground_points2
