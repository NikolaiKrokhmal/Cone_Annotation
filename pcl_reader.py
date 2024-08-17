import numpy as np
from innopy.api import FileReader, FrameDataAttributes, GrabType


class PCL_Reader:
    def __init__(self, recording_path,frame_skip=1, start_frame_num=0):
        self.recording_path = "../ut/" + recording_path
        print("recording path: ", self.recording_path)
        self.attr = [FrameDataAttributes(GrabType.GRAB_TYPE_MEASURMENTS_REFLECTION0)]
        self.fr = FileReader(recording_path)
        self.number_of_frames = self.fr.num_of_frames
        self.current_frame_num = start_frame_num
        self.data_frame = None
        self.points = []
        self.frame_skip = frame_skip

    def get_frame(self):
        if self.current_frame_num > self.number_of_frames:
            raise Exception("No more frames to read, end of recording")
        res = self.fr.get_frame(self.current_frame_num, self.attr)
        while res.success is False:
            print("Couldn't get frame number: ", self.current_frame_num)
            self.current_frame_num += 1
            print("Try to get frame number: ", self.current_frame_num)
            res = self.fr.get_frame(self.current_frame_num, self.attr)
        self.current_frame_num += self.frame_skip
        self.data_frame = res.results['GrabType.GRAB_TYPE_MEASURMENTS_REFLECTION0']

    def basic_reader_filter(self):
        for pcl_num in range(len(self.data_frame)):
            curr_pcl = self.data_frame[pcl_num]
            if curr_pcl['confidence'] > 61:
                self.points.append([curr_pcl['x'], curr_pcl['y'], curr_pcl['z']])

    def read_points_cloud(self):
        self.get_frame()
        self.basic_reader_filter()
        return np.array(self.points)
