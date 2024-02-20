# this module provides code that calculates an "mean" location on the
# earth (or other sphere) given a set of lat/lon points. It also
# returns the standard deviation of the locations of the points.
#
# The "mean" is the point that has the least sum-of-squares distance
# to all the other points. This is not gaurenteed to be unique (think
# of two points at the two poles), but it should be in most practical
# sets of data. The standard deviation is the root-mean-square of all
# the distances from the "mean" point to all the other points.
from numpy import *
from scipy.optimize import minimize


#based on
#https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lon1, lat1, lonVec, latVec,R=6731.0):
    """
    haversine(lon1, lat1, lonVec, latVec,R):

    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)

    lonVec and latVec can be vectors, lon1 and lat1 must be scalers

    R is radius of sphere

    """
    # convert decimal degrees to radians 
    lon1, lat1, lonVec, latVec = map(radians, [lon1, lat1, lonVec, latVec])

    # haversine formula 
    dlon = lonVec - lon1 
    dlat = latVec - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(latVec) * sin(dlon/2)**2
    c = 2 * arcsin(sqrt(a)) 
    #r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * R


def sumSqDiff(lonLat,lonVec,latVec,R):
    #helper function returns sum of square of distances from lonLat=[lon1,lat1] to points in lonVec and latVec
    dist=haversine(lonLat[0],lonLat[1],lonVec,latVec,R)
    #print('in sumSqDiff',lonLat,sum(dist**2))
    return sum(dist**2)

def sumSqDiff_weighted(lonLat,lonVec,latVec,numVec,R):
    #same as function above, but points in lonVec and latVec are weighted by numVec
    dist=haversine(lonLat[0],lonLat[1],lonVec,latVec,R)
    #print('in sumSqDiff_weighted',lonLat,sum((dist**2)*numVec))
    return sum((dist**2)*numVec)

def meanLonLat(lonVec,latVec,R=6372.8):
    '''
    meanLonLat(lonVec,latVec,R=6371.0) given a vector of longitude
    lonVec and latitude latVec, returns the location of the point that has
    the minimum distance to all the other points (the "mean" point) and
    root-mean-square of the distance between the "mean" point and all the
    other points (the "standard deviation" or STD).
    
    The keyword arguement R is the radius of the sphere, and so sets the
    units of the root-mean-square distance. By default, it is the radius
    of the earth in km, R=6372.8
    
    lat and lon are expected in DEGREES

    returns lon,lat,STD
    
    '''

    #starting guess is arythmetic mean of lat and lon
    lat0=mean(latVec)
    lon0=mean(lonVec)

    #call minimization routine
    res=minimize(sumSqDiff,array((lon0,lat0)),args=(lonVec,latVec,R),method = 'Nelder-Mead')

    #unpack results, remembering to normalize STD
    lonMean=res.x[0]
    latMean=res.x[1]
    STD=sqrt(res.fun/(len(latVec)-1))

    #check for failure
    if not res.success:
        print(' ')
        print('Failure to converge')
        print('lon0,lat0',lon0,lat0)
        print('lonVec',lonVec)
        print('latVec',latVec)
        assert False,'oops, meanLonLat did not converge for some reason'
    
    return lonMean,latMean,STD

def meanLonLat_weighted(lonVec,latVec,numVec,R=6372.8):
    '''same as meanLonLat, but takes additional arguement numVec which
    weights the (lon,lat) points such that if its value for a point is
    3, it equivalent to having that point in the list 3 times.

    '''

    #starting guess is arythmetic mean of lat and lon
    lat0=mean(latVec)
    lon0=mean(lonVec)

    #call minimization routine
    Nmax=50000
    res=minimize(sumSqDiff_weighted,array((lon0,lat0)),args=(lonVec,latVec,numVec,R),method = 'Nelder-Mead',
                 options={'maxiter':Nmax,'maxfev':Nmax})

    #unpack results, remembering to normalize STD
    lonMean=res.x[0]
    latMean=res.x[1]
    STD=sqrt(res.fun/(sum(numVec)-1))

    #check for failure
    if not res.success:
        print(' ')
        print('Failure to converge',flush=True)
        print('lon0,lat0',lon0,lat0,flush=True)
        print('lonVec',lonVec,flush=True)
        print('latVec',latVec,flush=True)
        print('numVec',numVec,flush=True)
        print('res',res,flush=True)
        assert False,'oops, meanLonLat did not converge for some reason'
   
    return lonMean,latMean,STD



if __name__ == "__main__":

    #test haversine distance
    lon1=45.0
    lat1=45.0
    lonVec=[44.0,45.0,46.0,45.0]
    latVec=[45.0,45.0,45.0,46.0]
    print('for lon1,lat1 of',lon1,lat1)
    print('For lonVec',lonVec)
    print('and latVec',latVec)
    dist=haversine(lon1,lat1,lonVec,latVec,R=6372.8)
    print('the distances are',dist)
    print(' ')

    #lets run some simple tests of mean distances
    latVec=[-1.0, -1.0,  1.0, 1.0]
    lonVec=[-1.0,  1.0, -1.0, 1.0]

    lonMean,latMean,stdDist=meanLonLat(lonVec,latVec)
    print('For lonVec',lonVec)
    print('and latVec',latVec)
    print('LonMean %f, latMean %f, stdDist %f'%(lonMean,latMean,stdDist))
    print(' ')

    #now lets test the weighted routines by comparing data with
    #duplicated points to data weight weighted points
    latVec=[-1.0, -1.0, -1.0,  1.0, 1.0, 1.0]
    lonVec=[-1.0, -1.0,  1.0, -1.0, 1.0, 1.0]
    lonMean_dup,latMean_dup,stdDist_dup=meanLonLat(lonVec,latVec)

    #now equivalent data with weights
    latVec=[-1.0, -1.0,  1.0, 1.0]
    lonVec=[-1.0,  1.0, -1.0, 1.0]
    numVec=[   2,    1,    1,   2]
    lonMean_weight,latMean_weight,stdDist_weight=meanLonLat_weighted(lonVec,latVec,numVec)

    print('duplicated LonMean %f, latMean %f, stdDist %f'%(lonMean_dup,latMean_dup,stdDist_dup))
    print('weighted   LonMean %f, latMean %f, stdDist %f'%(lonMean_weight,latMean_weight,stdDist_weight))
        
