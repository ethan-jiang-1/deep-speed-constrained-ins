# Synchronize the accelerometer and gyroscope data  
#
# Description:
#   Create data structure containing IMU and ARKit data.
#
# Copyright (C) 2018 Santiago Cortes
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.

import pandas as pd
import numpy as np

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


class DataSyncAgent(object):
    def __init__(self, root_data_dir):
        self.root_dir = os.path.abspath(root_data_dir)
        self.df_gyro = None
        self.df_acce = None
        self.df_magn = None
        self.df_arkit = None
        self.df_pose = None
        self.df_loaded = False
        self.construct_paths()

    def construct_paths(self):
        self.output_dir = self.root_dir + "/sync"
        self.ply_raw_arkit = self.output_dir + "/trj_raw_arkit.ply"
        self.ply_raw_pose = self.output_dir + "/trj_raw_pose.ply"
        self.ply_synced_arkit = self.output_dir + "/trj_synced_arkit.ply"
        self.ply_synced_pose = self.output_dir + "/trj_synced_pose.ply"

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

        path= raw_data_dir + '/iphone/arkit.csv'
        names=["sec", "pos_x", "pos_y", "pos_z", "rw", "rx", "ry", "rz"]
        self.df_arkit = self.load_raw_df(path, names, vobose=vobose)

        path= raw_data_dir + '/ground-truth/pose.csv'
        names=["sec", "pos_x", "pos_y", "pos_z", "rw", "rx", "ry", "rz"]
        self.df_pose = self.load_raw_df(path, names, vobose=vobose)

        self.df_loaded = True
        print("Load Raw Data from {} done".format(raw_data_dir))
        return True 

    def output_raw_ply(self):
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        
        ply_path = self.ply_raw_arkit
        if not os.path.isfile(ply_path):
            print("generate ply file ", ply_path)
            position = self.df_arkit[["pos_x", "pos_y", "pos_z"]].to_numpy()
            orientation = self.df_arkit[["rw", "rx", "ry", "rz"]].to_numpy()
            ex_write_ply_to_file(ply_path, 
                                position,
                                orientation)

        
        ply_path = self.ply_raw_pose
        if not os.path.isfile(ply_path):
            print("generate ply file ", ply_path)
            position = self.df_pose[["pos_x", "pos_y", "pos_z"]].to_numpy()
            orientation = self.df_pose[["rw", "rx", "ry", "rz"]].to_numpy()
            ex_write_ply_to_file(ply_path, 
                                position,
                                orientation)
        return True

    def view_raw_ply(self):
        import open3d as o3d
        pcd1, pcd2 = None, None

        if os.path.isfile(self.ply_raw_arkit):
            pcd1 = o3d.io.read_point_cloud(self.ply_raw_arkit)
        if os.path.isfile(self.ply_raw_pose):
            pcd2 = o3d.io.read_point_cloud(self.ply_raw_pose)

        if pcd1:
            o3d.visualization.draw_geometries([pcd1], window_name="arkit")
        if pcd2:
            o3d.visualization.draw_geometries([pcd2], window_name="pose")
        return True


def make_sync_data_on(ndx):
    root_path = '../data_ds/advio-'+str(ndx).zfill(2)

    dsa = DataSyncAgent(root_path)
    dsa.load_raw_dfs(vobose=0)
    dsa.output_raw_ply()
    dsa.view_raw_ply()



'''
    g=[]
    a=[]
    #t=np.array((map(float,acc[list('t')].values)))
    t=np.array(acc[list('t')].values)
    zer=t*0
    # Make imu
    for c in 'abc':
        #g.append(np.interp( np.array((map(float,acc[list('t')].values))), np.array((map(float,gyro[list('t')].values))), np.array((map(float,gyro[list(c)].values)))))
        g_aligned = np.interp(acc[list('t')].values.ravel(), gyro[list('t')].values.ravel(), gyro[list(c)].values.ravel())
        g.append(g_aligned)
        #a.append(np.array((map(float,acc[list(c)].values))))
        a.append(acc[list(c)].values.ravel())
    M=np.column_stack((t,zer+34,g[0],g[1],g[2],a[0],a[1],a[2],zer,zer))


    v=[]
    #t=np.array((map(float,arkit[list('t')].values)))
    t=np.array((arkit[list('t')].values))
    zer=t*0
    #Make arkit data
    for c in 'abcdefg':
        v.append(arkit[list(c)].values.ravel())
    Mkit=np.column_stack((t,zer+7,zer,v[0],v[1],v[2],v[3],v[4],v[5],v[6]))


    full=np.concatenate((M,Mkit))


    #sort to time vector
    #full = full[full[:,0].argsort()]
    #path= '../data_ds/advio-'+str(i).zfill(2)+'/iphone/imu-gyro.csv'
    #np.savetxt(path, full, delimiter=",",fmt='%.7f')
'''

def make_sync_data_for_all():
    for i in range(1,24):
        make_sync_data_on(i)  


if __name__ == "__main__":
    make_sync_data_for_all()