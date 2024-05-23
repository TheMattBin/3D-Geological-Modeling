import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt

def CorrFun(dx,dy,dz,SOFH,SOFV):
	return np.exp(-2*np.sqrt(dx**2+dy**2)/SOFH-2*dz/SOFV)

excel_file = r'D:\pythonProject\BH Data Processing\CentralWater_20211230.csv'
df = pd.read_csv(excel_file)

for i, row in df.iterrows():
	if row['Legend Code'] == 0:
		df.loc[i, 'P(I=0)'] = 1
	if row['Legend Code'] == 1:
		df.loc[i, 'P(I=1)'] = 1
	if row['Legend Code'] == 2:
		df.loc[i, 'P(I=2)'] = 1
	if row['Legend Code'] == 3:
		df.loc[i, 'P(I=3)'] = 1
	if row['Legend Code'] == 4:
		df.loc[i, 'P(I=4)'] = 1
	if row['Legend Code'] == 5:
		df.loc[i, 'P(I=5)'] = 1
	if row['Legend Code'] == 6:
		df.loc[i, 'P(I=6)'] = 1
	if row['Legend Code'] == 7:
		df.loc[i, 'P(I=7)'] = 1
	if row['Legend Code'] == 8:
		df.loc[i, 'P(I=8)'] = 1
	if row['Legend Code'] == 9:
		df.loc[i, 'P(I=9)'] = 1
	if row['Legend Code'] == 10:
		df.loc[i, 'P(I=10)'] = 1
	if row['Legend Code'] == 11:
		df.loc[i, 'P(I=11)'] = 1

# get average of each soil layer
df[['P(I=0)','P(I=1)','P(I=2)','P(I=3)','P(I=4)','P(I=5)','P(I=6)','P(I=7)','P(I=8)','P(I=9)','P(I=10)','P(I=11)']] = \
	df[['P(I=0)','P(I=1)','P(I=2)','P(I=3)','P(I=4)','P(I=5)','P(I=6)','P(I=7)','P(I=8)','P(I=9)','P(I=10)','P(I=11)']].fillna(0)
prob_mean = df.mean(axis=0)
Im = []
for i in range(12):
	Im.append(prob_mean['P(I={})'.format(i)])

# transform into indicator
I = df[['P(I=0)','P(I=1)','P(I=2)','P(I=3)','P(I=4)','P(I=5)','P(I=6)','P(I=7)','P(I=8)','P(I=9)','P(I=10)','P(I=11)']]
I = np.array(I.values.tolist())
I = I - Im

SOFH = 100
SOFV = 0.0001

# coordinates of each point
coordinates = df[['Easting', 'Northing', 'ElevMid']]
coordinates = np.array(coordinates.values.tolist())
X = np.reshape(coordinates[:, 0], (-1, 1))
Y = np.reshape(coordinates[:, 1], (-1, 1))
Z = np.reshape(coordinates[:, 2], (-1, 1))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = coordinates[:, 0].min() - 1, coordinates[:, 0].max() + 1
y_min, y_max = coordinates[:, 1].min() - 1, coordinates[:, 1].max() + 1
z_min, z_max = coordinates[:, 2].min() - 1, coordinates[:, 2].max() + 1

#spatial martix X, Y, Z by meshgrid
#corrFunc for known matrix
x1, x2= np.meshgrid(X, X)
y1, y2= np.meshgrid(Y, Y)
z1, z2= np.meshgrid(Z, Z)
dx = (abs(x1-x2))
dy = (abs(y1-y2))
dz = (abs(z1-z2))
Rkk = CorrFun(dx, dy, dz, SOFH, SOFV)
print(Rkk)
print(dy)
print(dz)
print(I.shape)

# create grid for 3D model
x = np.linspace(x_min, x_max, 20)
y = np.linspace(y_min, y_max, 20)
z = np.linspace(z_max, z_min, 10)
Zu, Yu, Xu = np.meshgrid(z, y, x, indexing='ij')

#correlation between unknown and known locations
x1u, x2u = np.meshgrid(X, Xu)
y1u, y2u = np.meshgrid(Y, Yu)
z1u, z2u = np.meshgrid(Z, Zu)
dxu = (abs(x1u-x2u))
dyu = (abs(y1u-y2u))
dzu = (abs(z1u-z2u))
Ruk = CorrFun(dxu, dyu, dzu, SOFH, SOFV)
print(Ruk.shape)
Rkkinv = np.linalg.inv(Rkk)
Predict = np.matmul(np.matmul(Ruk, Rkkinv), I) + Im
print(Predict)
#df.to_csv('test2.csv', index=False)




'''
n = 3
SOF = 20
xmin, xmax = 0, 3
ymin, ymax = 0, 2
zmin, zmax = 0, 4

# create grid for 3D model
x = np.linspace(xmin, xmax, n)
y = np.linspace(ymin, ymax, n)
z = np.linspace(zmax, zmin, 4)
Zu, Yu, Xu = np.meshgrid(z, y, x, indexing='ij')

# corr. coefficient of the grid ()
x1,x2= np.meshgrid(Xu, Xu)
y1,y2= np.meshgrid(Yu, Yu)
z1,z2= np.meshgrid(Zu, Zu)
dx = (abs(x1-x2))
dy = (abs(y1-y2))
dz = (abs(z1-z2))
Rkk = CorrFun(dx,dy,dz,SOF)
Rkkinv = np.linalg.inv(Rkk)
print(Zu.shape)
print(Rkk.shape)
print(Rkkinv.shape)

fig3d = plt.figure(figsize=(15,15))
plt3d = fig3d.add_subplot(111, projection='3d')
plt3d.scatter(Xu, Yu, Zu)
plt.show()

SOF = 20

f, ax = plt.subplots(1, 2, figsize=(13, 5.5))
'''
