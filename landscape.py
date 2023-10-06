import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


r = 1
n = 21
search_r = 1
# higher-order function parameter
l = 4
lbd = 0

# generate Yalcin's example with given epsilon
eps = 0.1
delta = (1-eps)/(1+eps)
m = 1
A = np.zeros((n, n, m))
for i in range(n):
  for j in range(n):
    if i==j or (i+1) == 2*(j+1) or 2*(i+1) == (j+1):
      A[i,j,0] = 1
    else:
      A[i,j,0] = eps
#for specific dimension
z = np.zeros(n)
for k in range(int(np.ceil(n/2))):
    z[2*k]=1

M_star = np.outer(z,z)
X_true = z.reshape(n, r)
M_true = X_true @ X_true.T

def grad(A, M_true, U, l, lbd):
  return jax.grad(func_vec)(jnp.array(V.T.reshape(n*search_r))).reshape(n,search_r)

def func(A, M_true, U, l, lbd):
  obj = 0
  num_samples = A.shape[-1]
  for i in range(num_samples):
    diftr = (A[:,:,i]* (U@U.T - M_true))
    obj += np.sum(diftr ** 2) / 2 + lbd * np.sum(diftr ** l) / l
  return obj/num_samples

def func_vec(Uvec):
  U = Uvec.reshape(search_r, n).T
  obj = 0
  num_samples = A.shape[-1]
  for i in range(num_samples):
    diftr = (A[:,:,i]* (U@U.T - M_true))
    obj += jnp.sum(diftr ** 2) / 2 + lbd * jnp.sum(diftr ** l) / l
  return obj/num_samples


# generate initial point close to spurious local minima
v = np.zeros(n)
for k in range(int(np.ceil(n/2))):
    v[2*k]=0.5*(-1)**(k%3)
V = v.reshape(n, r)


max_iter = 10000

# find saddle points
eta = 0.02
for j in range(max_iter):
  g = grad(A, M_true, V, l, lbd)
  if np.linalg.norm(g)<0.001:
    break
  V = V - eta * g

# generate two orthogonal directions from the critical point to the ground truth
d = X_true -V
d1 = np.zeros((n,1))
for k in range(int(np.ceil(n/2))):
    d1[k,0]=d[k,0]
d2 = (d-d1).reshape(n,search_r)

H = jax.hessian(func_vec)
def plot_hessian(alpha, beta):
  pos = (V+alpha*d1+beta*d2).T.reshape(n*search_r)
  HV = H(jnp.array(pos))
  ev = np.linalg.eigvals(HV)
  return np.real(min(ev))
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

x = arange(-0.3,1.3,0.05)
y = arange(-0.3,1.3,0.05)

grid = len(x)

# evaluation of the function on the grid
lbd = 0
Z0= np.zeros((grid,grid))
for i in range(grid):
  for j in range(grid):
    Z0[i,j] = plot_hessian(x[i],y[j])

lbd = 0.5
Z1 = np.zeros((grid, grid))
for i in range(grid):
    for j in range(grid):
        Z1[i, j] = plot_hessian(x[i], y[j])

lbd = 5
Z2= np.zeros((grid,grid))
for i in range(grid):
  for j in range(grid):
    Z2[i,j] = plot_hessian(x[i],y[j])

## plot the value of the minimum eigenvalue of hessian matrix around saddles
colmax = np.max([np.max(Z0),np.max(Z1),np.max(Z2)])
colmin = np.min([np.min(Z0),np.min(Z1),np.min(Z2)])
import matplotlib
norm = matplotlib.colors.Normalize(vmin=colmin, vmax=colmax)


fig = plt.figure()
fig.set_figheight(3.5)
fig.set_figwidth(15)
ax = fig.add_subplot(131)
bx = fig.add_subplot(132)
cx = fig.add_subplot(133)
a = ax.pcolormesh(Z0, norm=norm, cmap=plt.get_cmap('plasma'))
b = bx.pcolormesh(Z1, norm=norm, cmap=plt.get_cmap('plasma'))
c = cx.pcolormesh(Z2, norm=norm, cmap=plt.get_cmap('plasma'))
fig.colorbar(a, ax=[ax, bx, cx], shrink=0.7)


ax.scatter([6],[6], s=15, c='r')
ax.text(6,6,'saddle',c ='r', ha='center',va='bottom',fontsize=8.5)
ax.scatter([26],[26], s=15, c='b')
ax.text(26,26,'ground\n truth',c ='b', ha='center',va='bottom',fontsize=8.5)
ax.xaxis.set_ticks(arange(0,32,4),np.round(x[arange(0,32,4)],2))
ax.yaxis.set_ticks(arange(0,32,4),np.round(y[arange(0,32,4)],2))


bx.scatter([6],[6], s=15, c='r')
bx.text(6,6,'saddle',c ='r', ha='center',va='bottom',fontsize=8.5)
bx.scatter([26],[26], s=15, c='b')
bx.text(26,26,'ground\n truth',c ='b', ha='center',va='bottom',fontsize=8.5)
bx.xaxis.set_ticks(arange(0,32,4),np.round(x[arange(0,32,4)],2))
bx.yaxis.set_ticks(arange(0,32,4),np.round(y[arange(0,32,4)],2))

cx.scatter([6],[6], s=15, c='r')
cx.text(6,6,'saddle',c ='r', ha='center',va='bottom',fontsize=8.5)
cx.scatter([26],[26], s=15, c='b')
cx.text(26,26,'ground\n truth',c ='b', ha='center',va='bottom',fontsize=8.5)
cx.xaxis.set_ticks(arange(0,32,4),np.round(x[arange(0,32,4)],2))
cx.yaxis.set_ticks(arange(0,32,4),np.round(y[arange(0,32,4)],2))

plt.show()


# validating theorem 1/4
def neg_hessian_value(V,delta,l,lbd):
  cl = m**((2-l)/2) * ((2**l-1)/l-1)
  svr = np.linalg.svd(V)[1][search_r-1]
  D = np.linalg.norm(V@V.T - M_true, ord='fro')
  trmstar = np.trace(M_true)
  bd = np.sqrt(trmstar*svr*svr*((1+delta)+lbd*(l-1)*(1+delta)**(l/2)*D**(l-2))/((1-delta)/2+lbd*cl*(1-delta)**(l/2)*D**(l-2)))
  neg = (2*(1+delta)*svr**2-D*D*(1-delta)/trmstar)+lbd*D**(l-2)*(2*(l-1)*(1+delta)**(l/2)*svr**2-2*cl*D*D*(1-delta)**(l/2)/trmstar)
  return (D, bd, neg)
# print the theoretical bound for V
print(neg_hessian_value(V,delta,l,lbd))
# print the least eigenvalue value of Hessian at V
print(min(np.linalg.eigvals(jax.hessian(func_vec)(V.T.reshape(n*search_r)))))