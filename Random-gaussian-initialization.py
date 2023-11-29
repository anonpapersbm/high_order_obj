import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


########################################################

########## Generate sensing matrix A for Random Gaussian initialization
# parameters for matrix sensing problem
# (n,r) is the dimension for ground truth matrix M*
# (n,search_r) is the dimension for solution matrix X, search_r is the rank of X
r = 1
n = 20
search_r = 1

# higher-order function parameter
# l is the order of added loss function
# lbd is the value of regularization paramater lambda
l = 4
lbd = 0

# randomly generate sensing matrix A and ground truth M
# m is the number of sensing matrix
m = 20
A = np.random.normal(size=(n, n, m))/5
for i in range(m): #for symmetry
  A[:,:,i] = (A[:,:,i]+A[:,:,i].T)/2

# X_true is the BM factorization ground truth matrix X*
# M_true is the ground truth matrix M*
X_true = np.random.normal(size=(n, r))
M_true = X_true @ X_true.T


# gradient function of the objective function
def grad(A, M_true, U, l, lbd):
  num_samples = A.shape[-1]
  n, search_r = U.shape
  g = np.zeros((n, search_r))
  for i in range(num_samples):
    diftr = np.trace(A[:,:,i].T @ (U@U.T - M_true))
    g += (diftr + lbd * diftr ** (l-1)) * ((A[:,:,i] + A[:,:,i].T) @ U)
  return g/num_samples


# objective function f
def func(A, M_true, U, l, lbd):
  obj = 0
  num_samples = A.shape[-1]
  for i in range(num_samples):
    diftr = np.trace(A[:,:,i]* (U@U.T - M_true))
    obj += (diftr ** 2) / 2 + lbd * (diftr ** l) / l
  return obj/num_samples

# objective function f for vectorized input
def func_vec(Uvec):
  U = Uvec.reshape(search_r, n).T
  obj = 0
  num_samples = A.shape[-1]
  for i in range(num_samples):
    diftr = jnp.trace(A[:,:,i]* (U@U.T - M_true))
    obj += (diftr ** 2) / 2 + lbd * (diftr ** l) / l
  return obj/num_samples

########################################################




###### Plot figure 1
# The evolution of the objective function and the error
# with preturbed gradient descent
eta = 0.002
r = 3
search_r = 3
max_iter = 10000
n = 5
m = 20

# record the objective function
# record the error between the obtained solution and the ground truth
traj_normdif = np.zeros((max_iter, 1,5))
obj_func = np.zeros((max_iter, 1,5))
traj_normdif2 = np.zeros((max_iter, 1,5))
obj_func2 = np.zeros((max_iter, 1,5))

for i in range(5):
  # generate random gaussian sensing matrix and ground truth
  A = np.random.normal(size=(n, n, m))/5
  for ii in range(m):
    A[:,:,ii] = (A[:,:,ii]+A[:,:,ii].T)/2
  X_true = np.random.normal(size=(n, r))
  M_true = X_true @ X_true.T

  # lambda = 0, without high-order loss function
  lbd = 0
  V = np.random.normal(size=(n, search_r))/10
  M = np.copy(V)
  for j in range(max_iter):
    g = grad(A, M_true, V, l, lbd)
    if np.linalg.norm(g)<0.001:#perturbed GD
      g += np.random.normal(size=(n, search_r))/1000
    V = V - eta * g
    traj_normdif[j, 0,i] = np.linalg.norm(V@V.T - M_true)
    obj_func[j, 0,i] = func(A, M_true, V, l, lbd)


  # lambda = 1, with high-order loss function
  lbd = 1
  V = np.copy(M)
  for j in range(max_iter):
    g = grad(A, M_true, V, l, lbd)
    if np.linalg.norm(g)<0.001:#perturbed GD
      g += np.random.normal(size=(n, search_r))/1000
    V = V - eta * g
    traj_normdif2[j, 0,i] = np.linalg.norm(V@V.T - M_true)
    obj_func2[j, 0,i] = func(A, M_true, V, l, lbd)

def min3(x):
  return max(x,1e-3)
def min8(x):
  return max(x,1e-8)

figure, axis = plt.subplots(2, figsize = (5,5))
corbar = ["#0072BD","#D95319","#EDB120", "#7E2F8E", "#77AC30"]
for i in range(5):
  axis[0].semilogy(list(map(min3,traj_normdif2[0:max_iter,:,i])),'--', color = corbar[i])
  axis[0].semilogy(list(map(min3,traj_normdif[0:max_iter,:,i])), color =corbar[i])
  axis[0].legend([r'$\lambda=1$',r'$\lambda=0$'])
  axis[0].set_ylabel(r'$||\hat{X}\hat{X}^T - M^*||_F$')


  axis[1].semilogy(list(map(min8,obj_func2[0:max_iter,:,i])),'--',color = corbar[i])
  axis[1].semilogy(list(map(min8,obj_func[0:max_iter,:,i])), color =corbar[i])
  axis[1].set_ylabel(r'$f^l(\hat{X})$')
  axis[1].set_xlabel('Iteration Steps')
  axis[1].legend([r'$\lambda=1$',r'$\lambda=0$'])
plt.show()