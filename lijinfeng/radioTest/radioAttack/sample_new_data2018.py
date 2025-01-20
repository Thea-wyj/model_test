import numpy as np
import h5py

def saveData(dir,id,x,y,z):
	filename = dir  + '/part' + str(id) + '.h5'
	fw = h5py.File(filename, 'w')
	fw['X'] = np.vstack(x)
	fw['Y'] = np.vstack(y)
	fw['Z'] = np.vstack(z)
	print('X shape:', fw['X'].shape)
	print('Y shape:', fw['Y'].shape)
	print('Z shape:', fw['Z'].shape)
	fw.close()


##############################全局参数#######################################
f = h5py.File('E:/2018.01.OSC.0001_1024x2M.h5.tar/2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5','r')
dir_path = 'datas'
# modu_snr_size 代表每个part的数据数量
modu_snr_size = 68
############################################################################

for modu in range(24):
	X_list = []
	Y_list = []
	Z_list = []
	tX_list = []
	tY_list = []
	tZ_list = []
	print('part ',modu)
	start_modu = modu*106496
	for snr in range(11,26):
		start_snr = start_modu + snr*4096
		idx_list = np.random.choice(range(0,4096),size=modu_snr_size,replace=False)
		X = f['X'][start_snr:start_snr+4096][idx_list]
		#X = X[:,0:768,:]
		train = X[:60]
		test = X[60:]
		X_list.append(train)
		Y_list.append(f['Y'][start_snr:start_snr+4096][idx_list][:60])
		Z_list.append(f['Z'][start_snr:start_snr+4096][idx_list][:60])
		tX_list.append(test)
		tY_list.append(f['Y'][start_snr:start_snr + 4096][idx_list][60:])
		tZ_list.append(f['Z'][start_snr:start_snr + 4096][idx_list][60:])

	dir1 = dir_path+'/traindatas'
	dir2 = dir_path+'/testdatas'
	saveData(dir1,modu,X_list,Y_list,Z_list)
	saveData(dir2,modu,tX_list,tY_list,tZ_list)

f.close()