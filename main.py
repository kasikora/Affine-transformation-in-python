import math

#import griddata as griddata
import numpy
import matplotlib
import matplotlib.pyplot
import numpy as np
import pylab as pl
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
#import scipy.interpolate
#from scipy.interpolate import griddata


def loadingData(): #path
    f = open("szescian.txt", "r") #path
    file = f.read()
    #print(len(file))
    file = file.splitlines()
    #print(file)
    for i in range(len(file)):
        file[i] = file[i].split(",\t")
    return file

def matrix(file):
    matrix = np.matrix(file, float)

    #print(matrix)
    return matrix


def translation(matrix, x, y, z):
    num_rows, num_cols = matrix.shape
    #print(num_rows, num_cols)
    for i in range(num_rows):
        matrix[i] += (x, y, z)
    return matrix

    # tmp = [x,0,0; 0,y,0; 0,0,z]
    #T = np.zeros(shape=(num_rows - 1, num_cols - 1), dtype=float)
    #for i in range(num_rows):
    #   T[i] = (x, y, z)
    #matrixT = np.matrix([[x, 0, 0], [0, y, 0], [0, 0, z]])
    #return np.add(matrix, matrixT)


def scaling(matrix, x, y, z):
    num_rows, num_cols = matrix.shape
    #print(num_rows, num_cols)
    S = np.matrix([[x, 0, 0], [0, y, 0], [0, 0, z]])
    matrix = np.dot(matrix, S)
    return matrix
###############
def rotateNeutral(matrix, alfa, axis):
    tmp1 = []
    tmp2 = []
    if axis=="X":
        for i in matrix:
            tmp1.append(i[0,1])
            tmp2.append(i[0,2])
        offset = [0, (max(tmp1)+min(tmp1))/2, (max(tmp2)+min(tmp2))/2]
    elif axis=="Y":
        for i in matrix:
            tmp1.append(i[0,0])
            tmp2.append(i[0,2])
        offset = [(max(tmp1)+min(tmp1))/2, 0, (max(tmp2)+min(tmp2))/2]
        #offset = [(matrix[120, 0] + matrix[0, 0]) / 2, (matrix[120, 1] + matrix[0, 1]) / 2, 0]
    elif axis=="Z":
        for i in matrix:
            tmp1.append(i[0,0])
            tmp2.append(i[0,1])
        offset = [(max(tmp1)+min(tmp1))/2, (max(tmp2)+min(tmp2))/2, 0]
        #offset = [(matrix[120, 0] + matrix[0, 0]) / 2, (matrix[120, 1] + matrix[0, 1]) / 2, 0]
    else:
        offset=0
    #print(offset)
    return rotate(matrix, alfa, offset, axis)

def chooseRotationMatrix(axis, alpha):
    if axis=="X":
        return np.matrix([[1, 0, 0], [0, math.cos(alpha), math.sin(alpha)], [0, -math.sin(alpha), math.cos(alpha)]])
    elif axis=="Y":
        return np.matrix([[math.cos(alpha), 0, math.sin(alpha)], [0, 1, 0], [-math.sin(alpha), 0, math.cos(alpha)]])
    elif axis=="Z":
        return np.matrix([[math.cos(alpha), -math.sin(alpha), 0], [math.sin(alpha), math.cos(alpha), 0], [0, 0, 1]])
    else:
        return 0

def rotate(matrix, alfa, offset, axis):
    #num_rows, num_cols = matrix.shape
    #print(num_rows, num_cols)
    matrix=translation(matrix, 0-offset[0], 0-offset[1], 0-offset[2])
    r = chooseRotationMatrix(axis, alfa)
    matrix = np.dot(matrix, r)
    matrix=translation(matrix, offset[0], offset[1], offset[2])
    return matrix

def reflection(matrix):
    #num_rows, num_cols = matrix.shape
    #print(num_rows, num_cols)
    R = np.matrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return np.dot(matrix, R)

def shear(matrix, cx, cy):
    num_rows, num_cols = matrix.shape
    print(num_rows, num_cols)
    Sh = np.matrix([[1, cx, 0], [cy, 1, 0], [0, 0, 1]])
    return np.dot(matrix, Sh)
################
def plotDots(matrix):

    num_rows, num_cols = matrix.shape
    ax = plt.axes(projection="3d")
    X = np.zeros(shape=(num_rows, 1))
    Y = np.zeros(shape=(num_rows, 1))
    Z = np.zeros(shape=(num_rows, 1))
    for i in range(num_rows):
        X[i] = matrix[i, 0]
        Y[i] = matrix[i, 1]
        Z[i] = matrix[i, 2]
    #PlotPoints


    plt.show()
    ax.scatter(X, Y, Z, marker=".", alpha=1,)
    plt.show()


def plotLine(matrix):
    num_rows, num_cols = matrix.shape
    ax = plt.axes(projection="3d")
    X = np.zeros(shape=(num_rows, 1))
    Y = np.zeros(shape=(num_rows, 1))
    Z = np.zeros(shape=(num_rows, 1))
    for i in range(num_rows):
        X[i] = matrix[i, 0]
        Y[i] = matrix[i, 1]
        Z[i] = matrix[i, 2]
    #PlotLine?
    ax.plot(X, Y, Z)
    plt.show()

def plot3d(X,Y,Z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('plot3d');
    plt.show()

def plot3dbabiagora(matrix):
    num_rows, num_cols = matrix.shape

    X = np.zeros(shape=(num_rows, 1))
    Y = np.zeros(shape=(num_rows, 1))
    Z = np.zeros(shape=(num_rows, 1))
    for i in range(num_rows):
        X[i] = matrix[i, 0]
        Y[i] = matrix[i, 1]
        Z[i] = matrix[i, 2]


    x, y = np.meshgrid(X, Y)
    xi = np.c_[X.ravel(), Y.ravel()]
    yi = np.c_[x.ravel(), y.ravel()]
    #Z_interp = griddata(xi, Z.ravel(), yi) ??
    #z = griddata(X, Y, Z, x, y, 'cubic')

    fig = plt.figure(figsize=[])
    ax = fig.gca(projection="3d")
    #ax.plot_surface(x, y, z)
    #ax.plot_surface(x, y, Z_interp)
    plt.show()


    #ax = plt.figure().add_subplot(projection='3d')
    #ax.plot(x, y, z, label='jakiś tam label')
    #plt.show()


def plot(matrix):
    #matplotlib.pyplot.plot(matrix)
    #matplotlib.pyplot.show()
    num_rows, num_cols = matrix.shape
    #X = np.zeros(shape=(num_rows, 0))

    X = np.zeros(shape=(num_rows, 1))
    Y = np.zeros(shape=(num_rows, 1))
    Z = np.zeros(shape=(num_rows, 1))
    for i in range(num_rows):
        X[i] = matrix[i, 0]
        Y[i] = matrix[i, 1]
        Z[i] = matrix[i, 2]

    print(X)
    print(Y)
    print(Z)
    ax = plt.axes(projection="3d")

    #vmin = min( min(X), min(Y) );
    #vmax = max(max(X), max(Y));
    #x, y = np.meshgrid(X : (vmax-vmin)/10000 : Y)
    #x = np.linspace(dem['xmin'], dem['xmax'], )
    #z = griddata(X, Y, Z, x, y, "linear")


    #PlotPoints
    #ax.scatter(X, Y, Z, marker="v", alpha=1)
    #plt.show()
    #PlotLine?
    #ax.plot(X, Y, Z)
    #plt.show()
    #SurfacePlot
    #ax.plot_surface(x, y, z, cmap="plasma")
    #ax.plot_surface(X, Y, Z, cmap="plasma")
    plt.show()

def f(x, y):
        return np.sin(np.sqrt(x ** 2 + y ** 2))


def plotSzescian(m):

    C = np.array([[0, 255, 0], [0, 255, 0], [0, 255, 0], [255, 255, 0], [255, 0, 0], [255, 0, 255], [0, 255, 0],
                  [128, 0, 0]])
    X = np.zeros(shape=(len(m), 1))
    Y = np.zeros(shape=(len(m), 1))
    Z = np.zeros(shape=(len(m), 1))
    for i in range(len(m)):
        X[i] = m[i, 0]
        Y[i] = m[i, 1]
        Z[i] = m[i, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=C / 255.0, s=120)
    plt.show()


if __name__ == '__main__':
    m = matrix(loadingData())
    plotSzescian(m)
    m = rotateNeutral(m,math.pi,"Y")
    plotSzescian(m)
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    print (Z)
    print (np.size(Z))
    print(np.ndim(Z))


    plotLine(m)
    #plotLine(rotateNeutral(m, 1 * math.pi, "X"))
    #plotLine(reflection(m))
    #plotLine(shear(m, 0.5, 0))

    #plot funkcji
    plot3d(X,Y,Z)

    #plot babia gora
    #print(translation(m, 50, 50, 50))
    #print(scaling(m, 50, 50, 50))
    plotDots(m)

    #plot3dbabiagora(m)

    plotDots(translation(m, 250, 250, 250))
    plotDots(scaling(m, 50, 50, 50))
    plotDots(rotateNeutral(m, 1 * math.pi, "X"))  # obracanie wokol srodka
    #print(X)
    plotDots(rotate(m, 1 * math.pi, 100, "X"))
    plotDots(reflection(m))
    plotDots(shear(m, 0.5, 0))