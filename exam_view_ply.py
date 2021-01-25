from exam_ds.ex_data_sync_agent import DataSyncAgent


def make_sync_data_on(root_path, todos=["gen_ply", "view_ply"], vobose=0, re_gen=True):
    dsa = DataSyncAgent(root_path)
    dsa.load_raw_dfs(vobose=vobose)

    #if "gen_ply" in todos:
    #    dsa.output_raw_ply(re_gen=re_gen)
    if "view_ply" in todos:
        dsa.view_raw_ply(plys=["pose"])  # ["arkit", "tango", "pose"])
    #if "sync" in todos:
    #    dsa.output_sync(re_gen=re_gen)

def get_all_data_paths():
    root_paths = []
    for i in range(1,24):
        root_path = 'data_ds/advio-'+str(i).zfill(2)
        root_paths.append(root_path)

    #for i in range(1, 5):
    #    root_path = '../static/dataset-{:02d}'.format(i)        
    #    root_paths.append(root_path)            
    #for i in range(1, 3):
    #    root_path = '../swing/dataset-{:02d}'.format(i)        
    #    root_paths.append(root_path)
    return root_paths

def make_sync_data_for_all(todos):
    root_paths = get_all_data_paths()
      
    for root_path in root_paths:
        make_sync_data_on(root_path, todos=todos, vobose=2)


if __name__ == "__main__":
    todos = ["view_ply"]
    make_sync_data_for_all(todos)
