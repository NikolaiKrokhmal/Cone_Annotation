import pickle
from pcl_reader import PCL_Reader
from filters_algo import LidarFilter
from visualizers import Visualizer
import keyboard
import mouse
import numpy


def get_points(indice_list: list, pcl: numpy.ndarray):
    cone_points = pcl[indice_list]
    return cone_points


def main():
    # create all relevant objects for annotation
    pcl_filter = LidarFilter()
    vis1 = Visualizer(visualizer="pptk")
    rec_path = r"D:\Trimmed Recordings\27_11_2022\Lidar Recording_Cut_640-2940"
    reader = PCL_Reader(rec_path)
    print('In this recording there are ' + str(reader.number_of_frames) + ' frames.')
    start_frame = int(input('Enter starting frame: '))
    reader.frame_skip = int(input('Enter skip size: '))
    end_frame = int(input('Enter last frame: '))
    cone_dic = {0: "any cones!"}
    # cone_dic = {0: "BLUE cones!",
    #             1: "YELLOW cones!",
    #             2: "ORANGE cones!",
    #             3: "BIG cones!"}
    cone_list = [{}, {}, {}, {}]
    break_var = False

    for cone_idx in range(len(cone_dic)):
        if break_var:
            break
        reader.current_frame_num = start_frame
        cone_cnt = 0
        print(f"Now we annotate {cone_dic[cone_idx]}")
        frame_idx = reader.current_frame_num
        print(f"this is frame {frame_idx}")
        cone_list[cone_idx][frame_idx] = {}
        pcl_filter.points = reader.read_points_cloud()
        reader.points = []
        pcl_filter.filter_fov()
        vis1.show(pcl_filter.points)
        while True:
            if keyboard.read_key() == 'q':
                print('Exiting now.')
                break_var = True
                break
            elif keyboard.read_key() == 'u':
                vis1.show(pcl_filter.points)
            elif keyboard.read_key() == 'r':
                cone_cnt -= 1
                del cone_list[cone_idx][frame_idx][cone_cnt]
                print(f'Deleted cone {cone_cnt}!')
            elif keyboard.read_key() == 'n':
                vis1.close()
                frame_idx = reader.current_frame_num
                if frame_idx > end_frame:
                    print(f'done with {cone_dic[cone_idx]}!')
                    break
                pcl_filter.points = reader.read_points_cloud()
                reader.points = []
                pcl_filter.filter_fov()
                vis1.show(pcl_filter.points)
                cone_cnt = 0
                cone_list[cone_idx][frame_idx] = {}
                print(f"this is frame {frame_idx}")
            elif keyboard.read_key() == 'z':
                cone_points = vis1.v.get('selected')
                if len(cone_points) == 0:
                    pass
                else:
                    mouse.right_click()
                    cone_list[cone_idx][frame_idx][cone_cnt] = cone_points
                    print(f"cone {cone_cnt} recorded.")
                    cone_cnt += 1
    print('Done annotating batch! YIPPI!')
    with open(f"{rec_path}_frames_{start_frame}_{end_frame}.bin", 'wb') as file:
        pickle.dump(cone_list, file)


if __name__ == "__main__":
    main()
