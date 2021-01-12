import pandas as pd
import numpy as np
import scipy

try:
    import os
    print(os.getcwd())
    os.chdir("exam_ds")
    print(os.getcwd())
except:
    pass

try:
    from exam_ds.ex_ply_output import ex_write_ply_to_file
except:
    from ex_ply_output import ex_write_ply_to_file


def interpolate_vector_linear(input, input_timestamp, output_timestamp):
    """
    This function interpolate n-d vectors (despite the '3d' in the function name) into the output time stamps.
    
    Args:
        input: Nxd array containing N d-dimensional vectors.
        input_timestamp: N-sized array containing time stamps for each of the input quaternion.
        output_timestamp: M-sized array containing output time stamps.
    Return:
        quat_inter: Mxd array containing M vectors.
    """
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated

def get_synced_data(df_data, output_timestamp):
    data = df_data.to_numpy()
    input = data.transpose(1, 0)[1:].transpose(1, 0)
    input_timestamp = data.transpose(1, 0)[0]
    synced_data = interpolate_vector_linear(input, input_timestamp, output_timestamp)
    return synced_data


class DataSyncAgent(object):
    def __init__(self, root_data_dir):
        self.data_name = root_data_dir
        self.root_dir = os.path.abspath(root_data_dir)
        self.df_gyro = None
        self.df_acce = None
        self.df_magn = None

        self.df_pose = None        
        self.df_arkit = None
        self.df_arcore = None
        self.df_tango = None

        self.df_loaded = False
        self.construct_paths()

    def construct_paths(self):
        self.output_dir = self.root_dir + "/sync"
        self.ply_raw_arkit = self.output_dir + "/trj_raw_arkit.ply"
        self.ply_raw_arcore = self.output_dir + "/trj_raw_arcore.ply"
        self.ply_raw_pose = self.output_dir + "/trj_raw_pose.ply"
        self.ply_raw_tango = self.output_dir + "/trj_raw_tango.ply"

        self.csv_synced_data = self.output_dir + "/synced_data.csv"

    def load_raw_df(self, path, names, vobose=1):
        if vobose >= 1:
            print("\n\nRaw Data from {}".format(path))
        df_data= pd.read_csv(path, names=names)
        if vobose >= 1:
            print(df_data)
        if vobose >= 2:
            print(df_data.describe())
        return df_data

    def load_raw_dfs(self, vobose=1):
        if self.df_loaded:
            return True

        raw_data_dir = self.root_dir
        print("Load Raw Data from {}...".format(raw_data_dir))

        path= raw_data_dir + '/iphone/gyro.csv'
        names=["sec", "w_x", "w_y", "w_z"]
        self.df_gyro = self.load_raw_df(path, names, vobose=vobose)

        path= raw_data_dir + '/iphone/accelerometer.csv'
        names=["sec", "a_x", "a_y", "a_z"]
        self.df_acce = self.load_raw_df(path, names, vobose=vobose)

        path= raw_data_dir + '/iphone/magnetometer.csv'
        names=["sec", "m_x", "m_y", "m_z"]
        self.df_magn = self.load_raw_df(path, names, vobose=vobose)

        names=["sec", "pos_x", "pos_y", "pos_z", "rw", "rx", "ry", "rz"]
        path= raw_data_dir + '/ground-truth/pose.csv'
        self.df_pose = self.load_raw_df(path, names, vobose=vobose)

        path= raw_data_dir + '/iphone/arkit.csv'
        self.df_arkit = self.load_raw_df(path, names, vobose=vobose)

        path= raw_data_dir + '/pixel/arcore.csv'
        self.df_arcore = self.load_raw_df(path, names, vobose=vobose)

        path= raw_data_dir + '/tango/area-learning.csv'
        self.df_tango = self.load_raw_df(path, names, vobose=vobose)

        self.df_loaded = True
        print("Load Raw Data from {} done".format(raw_data_dir))
        return True 

    def _output_raw_ply(self, ply_path, df_data, kpoints):
        print("generate ply file ", ply_path)
        position = df_data[["pos_x", "pos_y", "pos_z"]].to_numpy()
        orientation = df_data[["rw", "rx", "ry", "rz"]].to_numpy()
        ex_write_ply_to_file(ply_path, 
                            position,
                            orientation,
                            kpoints=kpoints)

    def output_raw_ply(self, re_gen=False):
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        
        kpoints = 6
        ply_path = self.ply_raw_arkit
        if not os.path.isfile(ply_path) or re_gen: 
            self._output_raw_ply(ply_path, self.df_arkit, kpoints)

        ply_path = self.ply_raw_arcore
        if not os.path.isfile(ply_path) or re_gen: 
            self._output_raw_ply(ply_path, self.df_arcore, kpoints)

        ply_path = self.ply_raw_tango
        if not os.path.isfile(ply_path) or re_gen: 
            self._output_raw_ply(ply_path, self.df_tango, kpoints)

        ply_path = self.ply_raw_pose
        if not os.path.isfile(ply_path) or re_gen: 
            self._output_raw_ply(ply_path, self.df_pose, kpoints)
   
        return True

    def _view_raw_ply(self, path, name):
        import open3d as o3d
        if os.path.isfile(path):
            pcd = o3d.io.read_point_cloud(path)        
            o3d.visualization.draw_geometries([pcd], window_name=name)

    def view_raw_ply(self, plys):
        if "arkit" in plys:
            self._view_raw_ply(self.ply_raw_arkit, "arkit (iphone) - {}".format(self.data_name))

        if "arcore" in plys:
            self._view_raw_ply(self.ply_raw_arcore, "arcore (andorid) - {}".format(self.data_name))

        if "tango" in plys:
            self._view_raw_ply(self.ply_raw_tango, "tango (andorid) - {}".format(self.data_name))

        if "pose" in plys:
            self._view_raw_ply(self.ply_raw_pose, "pose (unknown) - {}".format(self.data_name))

        return True

    def output_sync(self, re_gen=False):
        if not(os.path.isfile(self.csv_synced_data) or re_gen):
            return
        
        secs_pose = self.df_pose["sec"].to_numpy()
        secs_gyro = self.df_gyro["sec"].to_numpy()
        secs_acce = self.df_acce["sec"].to_numpy()
        secs_magn = self.df_magn["sec"].to_numpy()

        start_ndx = -1
        end_ndx = -1
        secs_start, secs_end = -1, -1
        for i in range(1000):
            secs_start = secs_pose[i]
            if secs_start >= secs_gyro[0] and \
               secs_start >= secs_acce[0] and \
               secs_start >= secs_magn[0]:
               start_ndx = i
               break

        for i in range(len(secs_pose) - 1, len(secs_pose)-1000, -1):
            secs_end = secs_pose[i]
            if secs_end <= secs_gyro[-1] and \
               secs_end <= secs_acce[-1] and \
               secs_end <= secs_magn[-1]:
               end_ndx = i
               break        

        if start_ndx == -1 or end_ndx == -1:
            raise ValueError("no sec range")
        print("range of the sec: {:.4f} to {:.4f}".format(secs_pose[start_ndx], secs_pose[end_ndx]))

        sel_pose = self.df_pose.to_numpy()[start_ndx:end_ndx]
        output_timestamp = sel_pose.transpose(1, 0)[0]

        synced_data_gypo = get_synced_data(self.df_gyro, output_timestamp).transpose(0,1)
        synced_data_acce = get_synced_data(self.df_acce, output_timestamp).transpose(0,1)
        synced_data_magn = get_synced_data(self.df_magn, output_timestamp).transpose(0,1)
        synced_pose = sel_pose.transpose(0, 1)

        synced_full = np.column_stack((synced_pose, synced_data_gypo, synced_data_acce, synced_data_magn))
        columns = ["sec", "pos_x", "pos_y", "pos_z", "rw", "rx", "ry", "rz"] + \
                  ["w_x", "w_y", "w_z"] + ["a_x", "a_y", "a_z"] + ["m_x", "m_y", "m_z"]
                    
        df_synced_full = pd.DataFrame(synced_full, columns=columns)
        print(df_synced_full)
        #print(df_synced_full.describe())

        df_sync_ordered = df_synced_full[["sec"] + ["w_x", "w_y", "w_z"] + ["a_x", "a_y", "a_z"] + ["m_x", "m_y", "m_z"] + ["pos_x", "pos_y", "pos_z", "rw", "rx", "ry", "rz"]]
        print(df_sync_ordered)
        #print(df_sync_ordered.describe())

        df_sync_ordered.to_csv(self.csv_synced_data, float_format="%.6f")
        return True
        

        
