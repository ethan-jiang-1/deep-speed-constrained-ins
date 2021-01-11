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
    from exam_ds.ex_data_sync_agent import DataSyncAgent
except:
    from ex_data_sync_agent import DataSyncAgent


def make_sync_data_on(root_path, todos=["gen_ply", "view_ply"]):
    dsa = DataSyncAgent(root_path)
    dsa.load_raw_dfs(vobose=0)

    if "gen_ply" in todos:
        dsa.output_raw_ply(re_gen=True)
    if "view_ply" in todos:
        dsa.view_raw_ply(plys=["arkit", "tango", "pose"])


def make_sync_data_for_all(todos):
    root_paths = []
    for i in range(1,24):
        root_path = '../data_ds/advio-'+str(i).zfill(2)
        root_paths.append(root_path)

    #for i in range(1, 5):
    #    root_path = '../static/dataset-{:02d}'.format(i)        
    #    root_paths.append(root_path)            
    #for i in range(1, 3):
    #    root_path = '../swing/dataset-{:02d}'.format(i)        
    #    root_paths.append(root_path)
      
    for root_path in root_paths:
        make_sync_data_on(root_path, todos=todos)  


if __name__ == "__main__":
    todos = []
    make_sync_data_for_all(todos)