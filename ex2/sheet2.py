#!/usr/bin/env python
# coding: utf-8

# ## Before submitting
# 1. Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# 2. Make sure that no assertions fail or exceptions occur, otherwise points will be subtracted.
# 
# 3. After you submit the notebook more tests will be run on your code. The fact that no assertions fail on your computer localy does not guarantee that completed the exercise correctly.
# 
# 4. Please submit only the `*.ipynb` file.
# 
# 5. Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE". Edit only between `YOUR CODE HERE` and `END YOUR CODE`.
# 
# 6. Make sure to use Python 3, not Python 2.
# 
# Fill your group name and collaborators below:

# In[1]:


GROUPNAME = "Group 1"
COLLABORATORS = "Kjetsaa, Tor Istvan Stadler; Lassner, David; Schulte-Mattler, Jonas Prissian"


# ---

# # Exercise Sheet 2: Timing, Numpy, Plotting
# 
# The previous exercise sheet introduced several methods for classification: decision trees, nearest neighbors, and nearest means. Of those, the one that could learn from the data, and that also offered enough complexity to produce an accurate decision function was k-nearest neighbors. However, nearest neighbors can be slow when implemented in pure Python (i.e. with loops). This is especially the case when the number of data points or input dimensions is large.
# 
# In this exercise sheet, we will speed up nearest neighbors by utilizing `numpy` and `scipy` packages. Your task will be to **replace list-based operations by vector-based operations** between numpy arrays. The speed and correctness of the implementations will then be tested. In particular, performance graphs will be drawn using the library `matplotlib`.
# 
# Make sure to have installed all the required packages (e.g. sklearn, scipy). For this you can e.g. use `conda install <package>` or `pip install <package>`.

# ## Python Nearest Neighbor
# 
# The most basic element of computation of nearest neighbors is its distance function relating two arbitrary data points `x1` and `x2`. We assume that these points are iterable (i.e. we can use a loop over their dimensions). One way among others to compute the square Euclidean distance between two points is by computing the sum of the component-wise distances.

# In[2]:


def pydistance(x1, x2):
    return sum([(x1d - x2d) ** 2 for x1d, x2d in zip(x1, x2)])


# where we use the prefix "`py-`" of the function to indicate that the latter makes use of pure `Python` instead of `numpy`. Once the distance matrix has been implemented, the nearest neighbor for a given unlabeled point `u` that we would like to classify is obtained by iterating over all points in the training set `(X, Y)`, selecting the point with smallest distance to `u`, and returning its corresponding label. Here `X` denotes the list of inputs in the training set and `Y` denotes the list of labels.

# In[3]:


def pynearest(u, X, Y, distance=pydistance):
    xbest = None
    ybest = None
    dbest = float('inf')
    
    for x, y in zip(X, Y):
        d = distance(u, x)
        if d < dbest:
            ybest = y
            xbest = x
            dbest = d
            
    return ybest


# Note that this function either uses function `pydistance` (given as default if the argument distance is not specified). Or one could specify as argument a more optimized function for distance compuation, for example, one that uses `numpy`. Finally, one might not be interested in classifying a single point, but many of them. The method below receives a collection of such unlabeled test points stored in the variable `U`. The function returns a list of predictions associated to each test point.

# In[4]:


def pybatch(U, X, Y, nearest=pynearest, distance=pydistance):
    return [nearest(u, X, Y, distance=distance) for u in U]


# Again, such function uses by default the Python nearest neighbor search (with a specified distance function). However, we can also specified a more optimized nearest neighbor function, for example, based on `numpy`. Finally, one could consider an alternative function to `pybatch` that would use `numpy` from the beginning to the end. The implementation of such more optimized functions, and the testing of their correct behavior and higher performance will be the object of this exercise sheet.

# ## Testing and correctness
# 
# As a starting point, the code below tests the output of the nearest neighbor algorithm for some toy dataset with fixed parameters. In particular, the function `data.toy(M,N,d)` generates a problem with `M` unlabeled test points stored in a matrix `U` of size `(M x d)`, then `N` labeled training points stored in a matrix `X` of size `(N x d)` and the output label is stored in a vector `Y` of size `N` composed of zeros and ones encoding the two possible classes. The variable `d` denotes the number of dimensions of each point. The toy dataset is pseudo-random, that is, for fixed parameters, it produce a random-looking dataset, but every time the method is called with the same parameters, the dataset is the same. The pseudo-randomness property will be useful to verify that each nearest neighbor implementation performs the same overall computation. Please check the `data.py` file within the exercise folder for the implementation details. 

# In[5]:


import data
U, X, Y = data.toy(20, 100, 50)
print(pybatch(U, X, Y))


# In particular, the output of this function will help us to verify that the more optimized `numpy`-based versions of nearest neighbor are still valid.

# ## Plotting and performance
# 
# We now describe how to build a plot that relates a certain parameter of the dataset (e.g. the number of input dimensions `d` to the time required for the computation. We first initialize the basic plotting environment.

# In[6]:


import matplotlib
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 90


# The command "`%matplotlib inline`" tells IPython notebook that the plots should be rendered inside the notebook. 
# 
# The following code plots the computation time of predicting `100` points from the test set using a training set of size `100`, and where we vary the number of input dimensions.

# In[7]:


import timeit
from statistics import mean
# Values for the number of dimensions d to test
dlist = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# Measure the computation time for each choice of number of dimensions d
tlist = []
for d in dlist:
    U, X, Y = data.toy(100, 100, d)  
    # get the average of three runs
    delta = mean(timeit.repeat(lambda : pybatch(U,X,Y), number=1, repeat=3))
    tlist.append(delta)

# Plot the results in a graph
fig = plt.figure(figsize=(5, 3))
plt.plot(dlist, tlist, '-o')
plt.xscale('log'); plt.yscale('log'); plt.xlabel('d'); plt.ylabel('time'); plt.grid(True)


# The time on the vertical axis is in seconds. Note that the exact computation time depends on the speed of your computer. As expected, the computation time increases with the number of input dimensions. Unfortunately, for the small dataset considered here (`100` training and test points of `100` dimensions each), the algorithm already takes more than one second to execute. Thus, it is necessary for practical applications (e.g. the digit recognition task that we will consider at the end of this exercise sheet) to accelerate this nearest neighbor algorithm.

# ## 1. Accelerating the distance computation (25 P)
# 
# In this first exercise, we would like to accelerate the function that compute pairwise distances.
# 
# **a)** Create a new function `npdistance(x1,x2)` with the same output as `pydistance(x1,x2)`, but that computes the squared Euclidean distance using `numpy` operations. Verify that in both cases (i.e. using either `npdistance` or `pydistance` in the function `pybatch`) the output for the above toy example with parameters `M=20`, `N=100`, `d=50` (i.e. `data.toy(20,100,50)`) remains the same.
# 
# **Note**: The input vectors will always be `Numpy Arrays` or a `Lists of floats`.

# In[8]:


import numpy as np
def npdistance(x1, x2):
    '''
    Calculates the eucledian distance between two data points x1, x2
    
    Args:
        x1, x2 (vector-like): Two vectors (ndim=1) for which we want to calculate the distance
        
    Returns: 
        float: The distance between the two vectors x1, x2
    '''
    # >>>>> YOUR CODE HERE
    return np.sum((np.array(x1) - np.array(x2))**2)
    # <<<<< END YOUR CODE


# In[9]:


# Verify your function
assert npdistance([1,2,3], [2,3,4]) == 3
np.testing.assert_allclose(pybatch(U, X, Y, distance=pydistance), pybatch(U, X, Y, distance=npdistance))


# In[ ]:





# **b)** Create a plot similar to the one above, but where the computation time required by both methods are shown in a superposed manner. Here, we fix `M=100`, `N=100`, and we let `d` vary from `1` to `1000`, taking the list of values `[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]`.

# In[ ]:


# >>>>> YOUR CODE HERE
# Values for the number of dimensions d to test
dlistnp = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# Measure the computation time for each choice of number of dimensions d
tlistpy = []
tlistnp = []
for d in dlistnp:
    U, X, Y = data.toy(100, 100, d)  
    # get the average of three runs
    deltapy = mean(timeit.repeat(lambda : pybatch(U,X,Y), number=1, repeat=3))
    deltanp = mean(timeit.repeat(lambda : pybatch(U,X,Y, distance = npdistance), number=1, repeat=3))
    tlistpy.append(deltapy)
    tlistnp.append(deltanp)

# Plot the results in a graph
fig = plt.figure(figsize=(5, 3))
plt.plot(dlistnp, tlistpy, '-o')
plt.plot(dlistnp, tlistnp, '-or')
plt.xscale('log'); plt.yscale('log'); plt.xlabel('d'); plt.ylabel('time'); plt.grid(True)
# <<<<< END YOUR CODE


# **c)** Based on your results, explain what kind of speedup `numpy` provides, and in what regime do you expect the speedup to be the most important:
# 
# **Note**: For this exercise you only need to provide a free text answer

# In[10]:


# Explain the speedup that numpy provides
# >>>>> YOUR CODE HERE
# numpy allows us to avoid zip operators.
# A numpy array is a wrapper aorund a C array, and numpy.sum will therefore be much faster when doing operations
# on C arrays as opposed to python list. (Because C is faster than python)
# The reason for the numpy solution being slow for small values of d is that converting a numpy array from a
# python list is a slow process, but for large values of d it saves us time in total.
# <<<<< END YOUR CODE


# ## 2. Accelerating the nearest neighbor search (25 P)
# 
# Motivated by the success of the `numpy` optimized distance computation, we would like further accelerate the code by performing nearest neighbor search directly in `numpy`. This means that your function cannot use `for`/`while` loops.
# 
# **a)** Create a new function `npnearest(u,X,Y)` as an alternative to the function `pynearest(u,X,Y,distance=npdistance)` that we have used in the previous exercise. Again, verify your function for the same toy example as before (i.e. `data.toy(20,100,50)`).

# In[11]:



def npnearest(u: np.ndarray, X: np.ndarray, Y: np.ndarray, distance: 'callbale'=npdistance):
    '''
    Finds x1 so that x1 is in X and u and x1 have a minimal distance (according to the 
    provided distance function) compared to all other data points in X. Returns the label of x1
    
    Args:
        u (np.ndarray): The vector (ndim=1) we want to classify
        X (np.ndarray): A matrix (ndim=2) with training data points (vectors)
        Y (np.ndarray): A vector containing the label of each data point in X
        distance (callable): A function that receives two inputs and defines the distance function used
        
    Returns:
        int: The label of the data point which is closest to `u`
    '''
    # >>>>> YOUR CODE HERE
    #raise NotImplementedError("Replace this line by your code.")
    #U = np.full(X.shape, u)
    #distances = np.full(X.shape[0], distance(U,X))
    U = np.array([distance(u,x) for x in X])
    return Y[np.argmin(U)]
    

    # <<<<< END YOUR CODE


# In[12]:


# Verify your function
np.testing.assert_allclose(
    pybatch(U, X, Y, nearest=pynearest), 
    pybatch(U, X, Y, nearest=npnearest, distance=npdistance)
)


# **b)** Create a plot similar to the one above, where the new method is compared to the previous one. Here, we fix `M=100`, `d=100`, and we let `N` take different values `[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]`.

# In[13]:


# >>>>> YOUR CODE HERE
Nlistnp = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# Measure the computation time for each choice of number of dimensions d
tlistpy = []
tlistnp = []
for N in Nlistnp:
    U, X, Y = data.toy(100, N, 100)  
    # get the average of three runs
    deltapy = mean(timeit.repeat(lambda : pybatch(U,X,Y, nearest = pynearest), number=1, repeat=3))
    deltanp = mean(timeit.repeat(lambda : pybatch(U,X,Y, nearest = npnearest, distance = npdistance), number=1, repeat=3))
    tlistpy.append(deltapy)
    tlistnp.append(deltanp)

# Plot the results in a graph
fig = plt.figure(figsize=(5, 3))
plt.plot(Nlistnp, tlistpy, '-o')
plt.plot(Nlistnp, tlistnp, '-or')
plt.xscale('log'); plt.yscale('log'); plt.xlabel('N'); plt.ylabel('time'); plt.grid(True)
# <<<<< END YOUR CODE


# ## 3. Accelerating the processing of multiple test points (25 P)
# 
# Not yet fully happy with the performance of the algorithm, we would like to further optimize it by avoiding performing a loop on the test points, and instead, classify them all at once.
# 
# **a)** Create a new function `npbatch(U,X,Y)` as a replacement of the implementation `pybatch(U,X,Y,nearest=npnearest)` that we have built in the previous exercise. Inside this function, use [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) for the actual distance computation. Again, verify your function for the same toy example as before (i.e. `data.toy(20,100,50)`).

# In[14]:


import scipy
from scipy.spatial import distance
def npbatch(U, X, Y, *args, **kwargs):
    '''
    This function has the same functionality as the `pybatch` function.
    HOWEVER, the distance function is fixed (scipy.spatial.distance.cdist)
    
    Args:
        U (np.ndarray): A matrix (ndim=2) containing multiple vectors which we want to classify
        X (np.ndarray): A matrix (ndim=2) that represents the training data
        Y (np.ndarray): A vector (ndim=1) containing the labels for each data point in X
        
        All other arguments are ignored. *args, **kwargs are only there for compatibility 
        with the `pybatch` function
    '''
    # >>>>> YOUR CODE HERE
    
    D = distance.cdist(U,X,'sqeuclidean')
    C = [np.argmin(D[i]) for i in range(U.shape[0])]
    return [Y[c] for c in C]
    
    # <<<<< END YOUR CODE


# In[15]:


np.testing.assert_allclose(pybatch(U, X, Y), npbatch(U, X, Y))


# **b)** Create a plot comparing the computation time of the new implementation compared to the previous one. Here, we fix `N=100`, `d=100`, and we let `M` vary from `1` to `1000` with values `[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]`.

# In[94]:


# >>>>> YOUR CODE HERE
Mlistnp = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
# Measure the computation time for each choice of number of dimensions d
tlistpy = []
tlistnp = []
for M in Mlistnp:
    U, X, Y = data.toy(M, 100, 100)  
    # get the average of three runs
    deltapy = mean(timeit.repeat(lambda : pybatch(U,X,Y, nearest = npnearest, distance = npdistance), number=1, repeat=3))
    deltanp = mean(timeit.repeat(lambda : npbatch(U,X,Y), number=1, repeat=3))
    tlistpy.append(deltapy)
    tlistnp.append(deltanp)

# Plot the results in a graph
fig = plt.figure(figsize=(5, 3))
plt.plot(Mlistnp, tlistpy, '-o')
plt.plot(Mlistnp, tlistnp, '-or')
plt.xscale('log'); plt.yscale('log'); plt.xlabel('M'); plt.ylabel('time'); plt.grid(True)
# <<<<< END YOUR CODE


# ## 4. Application to real-world data (25 P)
# 
# Having now implemented an efficient K-nearest neighbor classifier, we can test it on real problems with many data points and dimensions. We consider a small handwritten digits recognition dataset, that can be directly obtained from the library `scikit-learn`. This dataset consists of handwritten digits of size `8 x 8` flattened into arrays of size `64`, with class between `0` and `9`. We use a function `data.digits()` to load the data and arrange data points in some predefined order.

# In[17]:


X, Y = data.digits()
print('Shape of data: {}'.format(X.shape),
      'Shape of labels: {}'.format(Y.shape),
      sep='\n')


# **a)** Using the function `imshow` of `matplotlib.pyplot (plt)` to visualize the first 16 digits (in a `4x4` grid) of the dataset.
# 
# **Note**: Your solution should output exactly one plot and shouldn't create new figures i.e. call `plt.figure`

# In[76]:


def plot_first_digits():
    '''
    Plots the first 100 digits in the dataset in one image
    '''
    # >>>>> YOUR CODE HERE
    #raise NotImplementedError("Replace this line by your code.")
    s=[1,2]
    t=[3,4]
    print(s+t)
    for i in range(8):
        print(X[0][8*i:8*i+8])
    print(X[1][0:8])
    
    IM0 = [[X[j][i+8*k] for j in range(4) for i in range(8)] for k in range(8)]
    IM1 = [[X[j][i+8*k] for j in range(4,8) for i in range(8)] for k in range(8)]
    IM2 = [[X[j][i+8*k] for j in range(8,12) for i in range(8)] for k in range(8)]
    IM3 = [[X[j][i+8*k] for j in range(12,16) for i in range(8)] for k in range(8)]
    
    #IM = [[X[j][i+8*k]  for h in range(4) for j in range(h*4, h*4 +4) for i in range(8)] for k in range(8)]
    #print(X[1])
    plt.imshow(IM0+IM1+IM2+IM3)
    #plt.imshow(IM)
    
    # <<<<< END YOUR CODE
plot_first_digits()


# In[ ]:





# **b)**
# * Partition the data into a "training" set and "test" set. The training set contains the first 1000 digits of `X`, and the test set contains the remaining ones.
# 
# * Assume that you don't know the labels for the test data and classify the test data using your efficient nearest neighbor implementation.
# 
# * Print the predicted labels for the test set.

# In[80]:


def train_test_split(x: np.ndarray, y: np.ndarray):
    '''
    Splits the data into train and test sets
    The first 1000 samples belong to the training set the rest to the test set
    
    Args:
        x (np.ndarray): A matrix (ndim=2) containing the data
        y (np.ndarray): A vector (ndim=1) containing the label for each datapoint
    
    Returns:
        tuple: A tuple containing 4 elements. The training data, the test data, the training labels 
            and the test labels
    '''
    # >>>>> YOUR CODE HERE
    #raise NotImplementedError("Replace this line by your code.")
    x_train, x_test = x[:1000], x[1000:]
    y_train, y_test = y[:1000], y[1000:]
    # <<<<< END YOUR CODE
    return x_train, x_test, y_train, y_test

def predict(x_train, x_test, y_train):
    '''
    For each x in x_test this function finds the nearest neighbour in x_train and returns thhat label
    Uses the `npbatch` function
    
    Args: 
        x_train (np.ndarray): A matrix (ndim=2) containing all the training data
        x_test (np.ndarray): A matrix (ndim=2) containing all the test data for which we want a prediction
        y_train (np.ndarray): A vector (ndim=1) containing the label of each datapoint in the training set
        
    Returns: 
        np.ndarray: A vector with the prediction for each datapoint/vector in x_test
    '''
    # >>>>> YOUR CODE HERE
    #raise NotImplementedError("Replace this line by your code.")
    y_test = np.array(npbatch(x_test, x_train, y_train))
    # <<<<< END YOUR CODE
    return y_test


# In[81]:


x_train, x_test, y_train, y_test = train_test_split(X, Y)

# check if types match
assert isinstance(x_train, np.ndarray)
assert isinstance(x_test, np.ndarray)
assert isinstance(y_train, np.ndarray)
assert isinstance(y_test, np.ndarray)

# check if shapes match
assert x_train.shape == (1000, 64), x_train.shape
assert x_test.shape == (797, 64), x_test.shape
assert y_train.shape == (1000,), y_train.shape
assert y_test.shape == (797,), y_test.shape

# check if values match
assert y_train[0] == 7
assert y_test[0] == 0


# In[ ]:





# In[82]:


# check if prediction shape matches
predictions = predict(x_train, x_test, y_train)
assert predictions.shape == (797,)
np.testing.assert_allclose(predictions[:3], [0, 7, 3])


# In[ ]:





# **c)** 
# * Finally, in order to determine the accuracy of the classifier, we would like to compare the predictions with the ground truth (i.e. the true labels from the test data).
# 
# * Compute the fraction of the test set where the predictions of the nearest neighbor algorithm and labels disagree. 

# In[87]:


def evaluate(x_train, x_test, y_train, y_test):
    # >>>>> YOUR CODE HERE
    #raise NotImplementedError("Replace this line by your code.")
    y_predict = predict(x_train, x_test, y_train)
    ratio_disagree = 1- (np.sum(y_predict == y_test)/y_test.size)
    # <<<<< END YOUR CODE
    return ratio_disagree


# In[88]:


x_train, x_test, y_train, y_test = train_test_split(X, Y)
ratio_disagree = evaluate(x_train, x_test, y_train, y_test)
assert 0. <= ratio_disagree <= 1.


# In[ ]:




