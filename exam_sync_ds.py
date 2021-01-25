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


def make_sync_data_for_ds():
    for i in range(1,24):  
        #Read data
        path= 'data_ds/advio-'+str(i).zfill(2)+'/iphone/arkit.csv'
        arkit=pd.read_csv(path,names=list('tabcdefg'))
        path= 'data_ds/advio-'+str(i).zfill(2)+'/iphone/accelerometer.csv'
        acc= pd.read_csv(path,names=list('tabc'))
        path= 'data_ds/advio-'+str(i).zfill(2)+'/iphone/gyro.csv'
        gyro= pd.read_csv(path,names=list('tabc'))


        print(acc)
        print(gyro)
        print(arkit)

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


        #full=np.concatenate((M,Mkit))


        #sort to time vector
        #full = full[full[:,0].argsort()]
        #path= '../data_ds/advio-'+str(i).zfill(2)+'/iphone/imu-gyro.csv'
        #np.savetxt(path, full, delimiter=",",fmt='%.7f')
        return M, Mkit


if __name__ == "__main__":
    make_sync_data_for_ds()
