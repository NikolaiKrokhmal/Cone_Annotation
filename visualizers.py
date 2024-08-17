import open3d as o3d
import pptk


class Visualizer:
    def __init__(self,visualizer="pptk"):
        print("Visualizer: ",visualizer)
        if not visualizer == "pptk" and not visualizer=="open3d":
            raise Exception("Visualizer not supported")
        self.visualizer = visualizer
        self.v = None

    def show(self,points):
        if self.visualizer == "pptk":
            self.v = pptk.viewer(points)
        elif self.visualizer == "open3d":
            self.v = o3d.geometry.PointCloud()
            self.v.points = o3d.utility.Vector3dVector(points)
            o3d.visualization.draw_geometries([self.v])

    def close(self):
        if self.visualizer == "pptk":
            self.v.close()
        elif self.visualizer == "open3d":
            self.v.close()
