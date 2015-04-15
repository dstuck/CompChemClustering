
# coding: utf-8

# In[1]:

import numpy as np                                                                     
import pandas as pd                                                                    
import csv                                                                             
import matplotlib.pyplot as plt   
import mpld3
#from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import axes3d

from sklearn import metrics
from sklearn.cluster import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import kneighbors_graph
from sklearn import manifold

from collections import Counter

#get_ipython().magic(u'matplotlib inline')
#mpld3.enable_notebook()


# In[2]:

#Load Data
df = pd.read_csv("data.csv",header=0)
refDat = np.genfromtxt('reference.csv', delimiter=',')
#drop rows with zeros
refDat = refDat[(df!=0.0).min(axis=1).values]
df = df[(df!=0.0).min(axis=1)].dropna(axis=0)


# In[3]:

oneIn20 = np.zeros(len(refDat),dtype=bool)
oneIn20[::20]=True
oneIn10 = np.zeros(len(refDat),dtype=bool)
oneIn10[::10]=True


# In[4]:

#drop 1 in 10 rare gas dimers
refDat = refDat[(np.logical_or((df.Type!='RG').values,oneIn10))]
df = df[(np.logical_or((df.Type!='RG').values,oneIn10))]


# In[5]:

#Subtract Error
numDat = df[df.columns[3:]].values
numDat = numDat - np.outer(refDat,np.ones(numDat.shape[1]))
#plt.plot(numDat)
#plt.show()


# In[6]:

scaling2 = StandardScaler()
numDat = scaling2.fit_transform(numDat.transpose())
numDat = numDat.transpose()
scaling = StandardScaler()
numDat = scaling.fit_transform(numDat)
#plt.plot(numDat)
#plt.show()


# In[7]:

#Set up projection space for plotting
pca = PCA(n_components=40, copy=True, whiten=True)
pcaDat = pca.fit_transform(numDat)
#pca.explained_variance_ratio_


# In[8]:

#Set up type and test set labels
labelOptions = df.Type.unique()
testLabelOptions = df.Dataset.unique()

tempLabels = df.Type.values.copy()
typeLabels = np.zeros(tempLabels.size)
typeLabels.dtype = int
for i in range(labelOptions.size):
    typeLabels[tempLabels==labelOptions[i]]=i
    
tempLabels = df.Dataset.values.copy()
setLabels = np.zeros(tempLabels.size)
setLabels.dtype = int
for i in range(testLabelOptions.size):
    setLabels[tempLabels==testLabelOptions[i]]=i
    
rxnNames = df.Name.values


# In[9]:

#Data dependent functions


# In[10]:

def heatPlots(outs,clust,sil,oMin=0,oMax=2400,cMin=0,cMax=14,sMin=None,sMax=None,penalty=False):
    plt.figure(1,figsize=(14,4))
    plt.subplot(131)
    heatmap = plt.pcolor(outs,vmin=oMin,vmax=oMax)
    plt.colorbar()
    plt.title("Outliers")
    plt.subplot(132)
    #heatmap = plt.pcolor(clustersfull,vmin=1, vmax=16)
    heatmap = plt.pcolor(clust,vmin=cMin,vmax=cMax)
    plt.colorbar()
    plt.title("Number of Clusters")
    plt.subplot(133)
    heatmap = plt.pcolor(sil,vmin=sMin,vmax=sMax)
    plt.colorbar()
#    plt.title("Silhouette Score")
    plt.title("Max Cluster Size")
    plt.show()
    if penalty:
        heatPlots(outPenalty(outs),clusterPenalty(clust),maxPenalty(sil),0,1,0,1,0,1,False)
#        heatmap = plt.pcolor(outPenalty(outs)*clusterPenalty(clust)*silPenalty(sil),vmin=0,vmax=.5)
        plt.figure(2,figsize=(4.5,4))
        heatmap = plt.pcolor(outPenalty(outs)*clusterPenalty(clust)*maxPenalty(sil),vmin=0,vmax=None)        
        plt.colorbar()
        plt.title("Total Value")
        plt.show()


# In[11]:

def clusterPlot3d(predLabels,dat,xdir=0,ydir=1,zdir=2,givenLabels=typeLabels):

    #mpld3.disable_notebook()

    fig = plt.figure(1,figsize=(18,8))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    #fig,(ax1,ax2) = plt.subplots(1,2)
    fig.figsize=(15,6)
    labels = predLabels
    #labels = typeLabels
    unique_labels = set(labels)
    #ax1(projection='3d')
    ax1.projection='3d'
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
    
        class_member_mask = (labels == k)
    
#        xy = dat[class_member_mask & core_samples_mask]
        xy = dat[class_member_mask]
        if k ==-1:
            ax1.plot(xy[:, xdir], xy[:, ydir], xy[:, zdir], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=3)
        else:
            ax1.plot(xy[:, xdir], xy[:, ydir], xy[:, zdir], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=6)

#        xy = dat[class_member_mask & ~core_samples_mask]
#        ax1.plot(xy[:, xdir], xy[:, ydir], xy[:, zdir], 'o', markerfacecolor=col,
#                 markeredgecolor='k', markersize=4)

    plt.title('Estimated number of clusters: %d' % len(unique_labels))
    #plt.subplot(1,2,2)
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    labels = givenLabels
    unique_labels = set(labels)
    #ax = plt.gca(projection='3d')
    ax2.projection='3d'
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)
    
#        xy = dat[class_member_mask & core_samples_mask]
        xy = dat[class_member_mask]
        ax2.plot(xy[:, xdir], xy[:, ydir], xy[:, zdir], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

#        xy = dat[class_member_mask & ~core_samples_mask]
#        ax2.plot(xy[:, xdir], xy[:, ydir], xy[:, zdir], 'o', markerfacecolor=col,
#                 markeredgecolor='k', markersize=4)

    plt.title('Chemist assigned labels')
    plt.show()
    #mpld3.enable_notebook()


# In[12]:

def dbscanGrid(epsGrid,sampleGrid,dat):
    clusters = np.zeros([len(epsGrid),len(sampleGrid)])
    #ami  = np.zeros([len(epsGrid),len(sampleGrid)])
    outliers = np.zeros([len(epsGrid),len(sampleGrid)])
    maxClusters  = np.zeros([len(epsGrid),len(sampleGrid)])
    dbPCAList = []
    for i in range(len(epsGrid)):
        dbPCAInnerList = []
        for j in range(len(sampleGrid)):
    #        dbPCA = DBSCAN(eps=epsGrid[i], min_samples=sampleGrid[j],algorithm='brute',metric='cosine').fit(dat)
            dbPCAInnerList.append(DBSCAN(eps=epsGrid[i], min_samples=sampleGrid[j]).fit(dat))
            core_samples_mask = np.zeros_like(dbPCAInnerList[-1].labels_, dtype=bool)
            core_samples_mask[dbPCAInnerList[-1].core_sample_indices_] = True
            clusters[i,j] = len(set(dbPCAInnerList[-1].labels_)) - (1 if -1 in dbPCAInnerList[-1].labels_ else 0)
            if clusters[i,j] > 1:
    #            silhouette[i,j] = metrics.silhouette_score(dat, dbPCAInnerList[-1].labels_)
    #            ami[i,j] = metrics.adjusted_mutual_info_score(typeLabels, dbPCAInnerList[-1].labels_)
                maxClusters[i,j] = np.bincount(dbPCAInnerList[-1].labels_+1)[1:].max()
            else:
    #            ami[i,j] = -2
                maxClusters[i,j]=0
            outliers[i,j] = -sum(dbPCAInnerList[-1].labels_[dbPCAInnerList[-1].labels_==-1])
        dbPCAList.append(dbPCAInnerList)
    return dbPCAList,outliers,clusters,maxClusters


# In[13]:

def getValByCluster(labels,valByItem,fixOutliers=False):
    valByCluster = []
    clusterLabels = labels.copy()
    if fixOutliers:
        clusterLabels[labels==-1] = clusterLabels.max()+1
    for i in range(len(np.unique(clusterLabels))):
        valByCluster.append(valByItem[clusterLabels==i])
    return valByCluster


# In[14]:

def printClusterByCount(clusterVals,clusterNames=None,thresh=0):
    if clusterNames is None:
        clusterNames = range(len(clusterVals))

    for cv,cname in zip(clusterVals,clusterNames):
        counts = Counter(cv)
        #(ckeys,cvals) = ((x,y) for (x,y) in (counts.keys(),counts.values()) if y > thresh)
        ckeys = (x for x in counts.keys() if counts[x]>thresh)
        cvals = (x for x in counts.values() if x > thresh)

        print cname, zip(ckeys,cvals)


# In[15]:

def averageMembership(clusterList,thresh=0):
    tMean = 0.0
    for clust in clusterList:
        if thresh == 0:
            tMean = tMean + len(np.unique(clust))
        else:
            for val in np.unique(clust):
                if len(clust[clust==val]) > thresh:
                    tMean = tMean + 1.0
    return(tMean/(len(clusterList)))


# In[16]:

#Identify clusters by the most characteristic item (using silhouette)
def bestRep(dat,labels,outName):
    bestExample = []
    silSamp = metrics.silhouette_samples(dat, labels)
    for num in np.unique(labels):
        clusterMask = labels==num
        bestExample.append(outName[clusterMask][np.argmax(silSamp[clusterMask])])
    return bestExample


# In[17]:

#Identify clusters by the most central item (using cluster mean)
#Labels must be contiguous integers!
def centerRep(dat,labels,outName,centers=None):
    if centers is None:
        centers = np.zeros((dat.shape[1],len(np.unique(labels))))
        for lab in np.unique(labels):
            centers[:,lab] = dat[labels==lab].mean(axis=0)
    bestExample = []
    for lab in np.unique(labels):
        centerDiffs = np.linalg.norm(dat[labels==lab]-np.outer(np.ones(dat[labels==lab].shape[0]),centers[:,lab]),axis=1)
        bestExample.append(outName[labels==lab][np.argmin(centerDiffs)])
    return bestExample


# In[18]:

#Penalty functions:
def gaussian(x,mu=0,sig=1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
def outPenalty(nOut,mu=500,sig=700):
    return gaussian(nOut,mu,sig)
def clusterPenalty(nCluster,mu=10,sig=3):
    return gaussian(nCluster,mu,sig)
def silPenalty(sScore,mu=1,sig=.7):
    return gaussian(sScore,mu,sig)
def maxPenalty(sScore,mu=1400,sig=500):
    return gaussian(sScore,mu,sig)


# In[19]:

#clusterPlot3d(typeLabels,pcaDat[:,1:])


# In[19]:




### DBSCAN

# In[20]:

#DBSCAN parameters
epsVals = np.linspace(2.5,4.5,10)
sampleVals = range(8,17,2)
print epsVals
print sampleVals


# In[21]:

dbList, outliersfull, clustersfull, maxClustersfull = dbscanGrid(epsVals,sampleVals,numDat)


# In[22]:

heatPlots(outliersfull,clustersfull,maxClustersfull,sMin=-0.5,penalty=True)


# In[23]:

#Optimal parameters
penMat = outPenalty(outliersfull)*clusterPenalty(clustersfull)*maxPenalty(maxClustersfull)
imax,jmax = np.unravel_index(penMat.argmax(), penMat.shape)
print (imax,jmax)
print epsVals[imax]
print sampleVals[jmax]

db = dbList[imax][jmax]


# In[24]:

print ("Num Clusters: %02d " % (len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)))
#print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(numDat, labels))
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(typeLabels, labels))
print("Mutual Info: %0.3f" % metrics.adjusted_mutual_info_score(typeLabels, db.labels_))
print("Mutual Info (Test Sets): %0.3f" % metrics.adjusted_mutual_info_score(setLabels, db.labels_))
print np.bincount(db.labels_+1)


# In[25]:

clusterPlot3d(db.labels_,pcaDat[:,0:])


#### Collaborative Filtering Outliers

# In[26]:

newLabels = db.labels_.copy()
nonOutLabels = np.unique(newLabels)
nonOutLabels = nonOutLabels[nonOutLabels>=0]

n_neighbors = 15
ngraph = kneighbors_graph(numDat,n_neighbors,mode='connectivity')


# In[27]:

converged = False
while not converged:
    converged = True
    outlierPos = np.arange(numDat.shape[0])[newLabels==-1]
    for i in range(len(outlierPos)):
        nnLabelCount = Counter(newLabels[ngraph[outlierPos[i]].nonzero()[1]]).most_common(1)
        if nnLabelCount[0][0]>=0 and nnLabelCount[0][1]>=n_neighbors/2:
            newLabels[outlierPos[i]] = nnLabelCount[0][0]
            converged = False
converged = False
while not converged:
    converged = True
    outlierPos = np.arange(numDat.shape[0])[newLabels==-1]
    for i in range(len(outlierPos)):
        nnLabelCount = Counter(newLabels[ngraph[outlierPos[i]].nonzero()[1]]).most_common(2)
        if nnLabelCount[0][0]>=0:
            newLabels[outlierPos[i]] = nnLabelCount[0][0]
            converged = False
        elif len(nnLabelCount)>1:
            newLabels[outlierPos[i]] = nnLabelCount[1][0]
            converged = False
#        print "Setting ",i," to ", nnLabelCount[0][0]
if len(newLabels[newLabels==-1]):
    newLabels[newLabels==-1] = len(nonOutLabels)+1
    print "Remaining outliers assigned to new cluster!" 


# In[28]:

clusterPlot3d(newLabels,pcaDat[:,1:])


# In[29]:

print("Mutual Info: %0.3f" % metrics.adjusted_mutual_info_score(typeLabels, newLabels))
print("Mutual Info (Test Sets): %0.3f" % metrics.adjusted_mutual_info_score(setLabels, newLabels))
print np.bincount(newLabels)


# In[29]:




# In[29]:




#### Nearest Cluster Outliers

# In[30]:

#newLabels = db.labels_.copy()
#outlierDat = numDat[newLabels==-1]
#nonOutLabels = np.unique(newLabels)
#nonOutLabels = nonOutLabels[nonOutLabels>=0]
#
#centers = np.zeros((numDat.shape[1],len(nonOutLabels)))
#for lab in nonOutLabels:
#    centers[:,lab] = numDat[db.labels_==lab].mean(axis=0)
#
#centerDiffs = np.zeros((outlierDat.shape[0],len(nonOutLabels)))
#for lab in nonOutLabels:
#    centerDiffs[:,lab] = np.linalg.norm(outlierDat-np.outer(np.ones(outlierDat.shape[0]),centers[:,lab]),axis=1)
#newLabels[newLabels==-1] = np.argmin(centerDiffs,axis=1)


# In[31]:

#clusterPlot3d(newLabels,pcaDat[:,:])


# In[31]:




# In[31]:




#### K-means on outliers

# In[32]:

#newLabels = db.labels_.copy()
#outlierDat = numDat[newLabels==-1]
#pca = PCA(n_components=40, copy=True, whiten=True)
#pcaOutlierDat = pca.fit_transform(outlierDat)
##print pca.explained_variance_ratio_
#kmeansOut = KMeans(n_clusters=2, n_init=1000, random_state=0).fit(pcaOutlierDat)
#newLabels[newLabels==-1]=kmeansOut.labels_+newLabels.max()+1
#print kmeansOut.inertia_#, " should be smaller than 27394.0295305"


# In[33]:

#clusterPlot3d(newLabels,pcaDat[:,0:])


# In[34]:

#print("Mutual Info: %0.3f" % metrics.adjusted_mutual_info_score(typeLabels, newLabels))
#print("Mutual Info (Test Sets): %0.3f" % metrics.adjusted_mutual_info_score(setLabels, newLabels))
#print np.bincount(newLabels)


#### Reaction Clusters

# In[35]:

clusterType = getValByCluster(newLabels,df.Type.values,False)
clusterTestSet = getValByCluster(newLabels,df.Dataset.values,False)


# In[36]:

printClusterByCount(clusterType,thresh=2)


# In[37]:

printClusterByCount(clusterTestSet,thresh=2)


# In[38]:

revClusterType = getValByCluster(typeLabels,newLabels,False)
revClusterSet = getValByCluster(setLabels,newLabels,False)
printClusterByCount(revClusterType,clusterNames=labelOptions, thresh=2)


# In[39]:

print averageMembership(revClusterSet,thresh=3)


# In[40]:

printClusterByCount(revClusterSet,clusterNames=testLabelOptions, thresh=2)


# In[41]:

#rxnReps = labelOptions[bestRep(numDat,newLabels,typeLabels)]
#rxnSetReps = testLabelOptions[bestRep(numDat,newLabels,setLabels)]
#print rxnReps
#print rxnSetReps


# In[42]:

rxnReps = labelOptions[centerRep(numDat,newLabels,typeLabels)]
rxnSetReps = testLabelOptions[centerRep(numDat,newLabels,setLabels)]
print rxnReps
print rxnSetReps


# In[42]:




#### Create Reaction Cluster Masks

# In[43]:

rxnClusters = np.zeros((len(newLabels),len(np.unique(newLabels))),dtype=bool)


# In[44]:

for i in range(len(np.unique(newLabels))):
    rxnClusters[:,i] = newLabels==i


# In[44]:




## Cluster Functionals

# In[45]:

clusterDat = scaling.inverse_transform(numDat).T


# In[136]:

functionals = df.columns[3:].values
#print functionals


# In[137]:

functionalType = np.genfromtxt('functionalInfo.csv', delimiter=',')
functionalType = functionalType[1:,1:]

isMeta = functionalType[:,0]>0
isHybrid = functionalType[:,1]>0
isDispersion = functionalType[:,2]>0

fLabels = np.zeros(len(isDispersion))
fLabels.dtype = int
for i in range(len(fLabels)):
    if isMeta[i]:# and not isLongRangeHybrid[i]:
        fLabels[i]= fLabels[i]+1;
    if isHybrid[i]:
        fLabels[i]= fLabels[i]+2;
    if isDispersion[i]:
        fLabels[i]= fLabels[i]+4;
    
fNames = np.asarray(['Local Pure','Local Pure Meta','Local Hybrid','Local Hybrid Meta'
                     ,'Non-local Pure','Non-local Pure Meta','Non-local Hybrid','Non-local Hybrid Meta',])


# In[48]:

pca = PCA(n_components=45, copy=True, whiten=True)
clusterTruncDat = pca.fit_transform(clusterDat)


# In[49]:

#clusterPlot3d(fLabels,clusterTruncDat[:,0:],givenLabels=fLabels)


# In[50]:

kmeansCluster = KMeans(n_clusters=10, n_init=500, random_state=0).fit(clusterDat)
funcLabels = kmeansCluster.labels_


# In[51]:

clusterPlot3d(funcLabels,clusterTruncDat[:,0:],givenLabels=fLabels)


# In[52]:

print metrics.adjusted_mutual_info_score(fLabels, funcLabels) #should be 0.529655901311 


# In[53]:

kmeansCluster.cluster_centers_.shape


# In[138]:

clusterNum = getValByCluster(funcLabels,fLabels)
clusterFType = getValByCluster(funcLabels,fNames[fLabels])
clusterFuncs = getValByCluster(funcLabels,functionals)
#funcReps = bestRep(numDat,funcLabels,functionals)
#Closest to center is better for k-means
#funcReps = centerRep(clusterDat,funcLabels,functionals)
funcReps = centerRep(clusterDat,funcLabels,functionals,centers=(kmeansCluster.cluster_centers_).T)


# In[139]:

printClusterByCount(clusterFType,thresh=0,clusterNames=funcReps)


# In[140]:

for clust in clusterFuncs: print clust


# In[141]:

for rep,clust in zip(funcReps,clusterNum): print rep,clust


# In[58]:

funcClusters = np.zeros((len(funcLabels),len(np.unique(funcLabels))),dtype=bool)
for i in range(len(np.unique(funcLabels))):
    funcClusters[:,i] = funcLabels==i


# In[58]:




### Cluster-Cluster Analysis

# In[191]:

errorDat = scaling2.inverse_transform(clusterDat).T
#errorDat = np.square(errorDat)


# In[192]:

nRxn = errorDat.shape[0]
mFunc = errorDat.shape[1]
nRxnCluster = rxnClusters.shape[1]
mFuncCluster = funcClusters.shape[1]
rmsdByCluster = np.zeros((nRxnCluster,mFuncCluster))
#print nRxnCluster
#print mFuncCluster


# In[193]:

for i in range(nRxnCluster):
    for j in range(mFuncCluster):
        rmsdByCluster[i,j] = errorDat[np.outer(rxnClusters[:,i],funcClusters[:,j])].mean()
#rmsdByCluster = np.sqrt(rmsdByCluster)


# In[194]:

#rmsdByCluster = rmsdByCluster/(np.outer(rmsdByCluster.mean(axis=1),np.ones(rmsdByCluster.shape[1])))
#scalingErr = StandardScaler(with_mean=False,with_std=True)
#rmsdByCluster = scalingErr.fit_transform(rmsdByCluster)
#rmsdByCluster = scaling.fit_transform(rmsdByCluster.T).T


# In[195]:

printClusterByCount(clusterType,thresh=2)
printClusterByCount(clusterFType,thresh=0)


# In[196]:

#Select clusters to plot
#plotOrderRxn = np.arange(nRxnCluster)
plotOrderRxn = np.array([0,1,2,3,5,6,7,8,9])
#plotOrderFunc = np.arange(mFuncCluster)
plotOrderFunc = np.array([0,1,2,5,6,9])

nRxnPlot = len(plotOrderRxn)
mFuncPlot = len(plotOrderFunc)

plotMaskRxn = np.zeros(nRxnCluster,dtype=bool)
plotMaskFunc = np.zeros(mFuncCluster,dtype=bool)
for i in plotOrderRxn: plotMaskRxn[i]=True
for i in plotOrderFunc: plotMaskFunc[i]=True

#nRxnPlot = len(plotOrderRxn)
#mFuncPlot = len(plotOrderFunc)

rmsdByCluster = rmsdByCluster[plotMaskRxn,:][:,plotOrderFunc]


# In[197]:

#mpld3.disable_notebook()
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(15)
spacing = np.arange(mFuncPlot)
colorList = ['b','g','r','c','m','y','k']
width = 0.8/nRxnPlot
rects = []
for i in range(nRxnPlot):
    rects.append(ax.bar(spacing+i*width,rmsdByCluster[i,:],width,color=colorList[i%len(colorList)]))
#    rects.append(ax.bar(spacing+i*width,-np.log(rmsdByPlot[:,i].T),width,color=colorList[i%len(colorList)]))

ax.set_ylabel('RMSE')
#ax.set_ylim(-.03,.02)
ax.set_xticks(spacing+width*nRxnPlot/2.0)
ax.set_xticklabels(np.array(funcReps)[plotOrderFunc])
ax.legend([rects[i][0] for i in range(nRxnPlot)], (rxnSetReps[plotOrderRxn]).tolist(),loc='best')
#ax.legend([rects[i][0] for i in range(nRxnPlot)], (rxnReps[plotOrderRxn]).tolist(),loc='best')
ax.set_title('Clustered Errors')
plt.show()
#mpld3.enable_notebook()


# In[66]:

for rep,clust in zip(funcReps,clusterNum): print rep,clust


# In[66]:




# In[67]:

rxnChem = np.zeros((len(typeLabels),len(np.unique(typeLabels))),dtype=bool)
for i in range(len(np.unique(typeLabels))):
    rxnChem[:,i] = typeLabels==i


# In[68]:

funcChem = np.zeros((len(fLabels),len(np.unique(fLabels))),dtype=bool)
for i in range(len(np.unique(fLabels))):
    funcChem[:,i] = fLabels==i


# In[69]:

nRxnChem = len(np.unique(typeLabels))
mFuncChem = len(np.unique(fLabels))
rmsdByChem = np.zeros((nRxnChem,mFuncChem))


# In[70]:

for i in range(nRxnChem):
    for j in range(mFuncChem):
        rmsdByChem[i,j] = errorDat[np.outer(rxnChem[:,i],funcChem[:,j])].mean()
#rmsdByChem = np.sqrt(rmsdByChem)


# In[71]:

#rmsdByChem = rmsdByChem/(np.outer(rmsdByChem.mean(axis=1),np.ones(rmsdByChem.shape[1])))
#scalingErr2 = StandardScaler(with_mean=False,with_std=True)
#rmsdByChem = scalingErr2.fit_transform(rmsdByChem)


# In[72]:

#Select clusters to plot
#plotOrderRxnChem = np.arange(nRxnChem)
plotOrderRxnChem = np.array([0,1,2,3,4,7])
plotOrderFuncChem = np.arange(mFuncChem)
#plotOrderFuncChem = np.array([0,1,2,4,6,8,9])

plotMaskRxnChem = np.zeros(nRxnChem,dtype=bool)
plotMaskFuncChem = np.zeros(mFuncChem,dtype=bool)
for i in plotOrderRxnChem: plotMaskRxnChem[i]=True
for i in plotOrderFuncChem: plotMaskFuncChem[i]=True

nRxnPlotChem = len(plotOrderRxnChem)
mFuncPlotChem = len(plotOrderFuncChem)

rmsdByChem = rmsdByChem[plotMaskRxnChem,:][:,plotOrderFuncChem]


# In[73]:

#mpld3.disable_notebook()
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(15)
spacing = np.arange(mFuncPlotChem)
colorList = ['b','g','r','c','m','y','k']
width = 0.8/nRxnPlotChem
rects = []
for i in range(nRxnPlotChem):
    rects.append(ax.bar(spacing+i*width,rmsdByChem[i,:],width,color=colorList[i%len(colorList)]))
#    rects.append(ax.bar(spacing+i*width,-np.log(rmsdByChem[:,i].T),width,color=colorList[i%len(colorList)]))

ax.set_ylabel('RMSE')
#ax.set_ylim(0,3)
ax.set_xticks(spacing+width*nRxnPlotChem/2.0)

ax.set_xticklabels(np.array(fNames)[plotOrderFuncChem])
ax.legend([rects[i][0] for i in range(nRxnPlotChem)], (labelOptions[plotOrderRxnChem]).tolist(),loc='best')
ax.set_title('Clustered Errors')
plt.show()
#mpld3.enable_notebook()


# In[74]:

labelOptions


# In[74]:




### Locally Linear Embedding of DF Clusters

# In[107]:

plotColors = plt.cm.Spectral(np.linspace(0, 1, len(np.unique(funcLabels))))


# In[108]:

embedding= manifold.MDS(n_components=2, n_init=1, max_iter=100, random_state=2).fit_transform(clusterDat).T


# In[173]:

#mpld3.enable_notebook()
fig = plt.figure(1, facecolor='w', figsize=(5, 5))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

scatter = plt.scatter(embedding[0], embedding[1], s=60, c=plotColors[funcLabels])

css = """
p  {
    font-family:Arial, Helvetica, sans-serif;
    border: 1px solid black;
    background-color: #ffffff;
}
"""
htmlLabels = []
for f in functionals:
    htmlLabels.append("<p><b>"+f)

#tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=functionals.tolist())
tooltip = mpld3.plugins.PointHTMLTooltip(scatter, hoffset=10, voffset=15, labels=htmlLabels,css=css)
mpld3.plugins.connect(fig, tooltip)

#plt.show()
#mpld3.show()


# In[77]:




# In[78]:

embeddingTSNE = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(numDat).T


# In[176]:

plotFColors = plt.cm.Spectral(np.linspace(0, 1, len(np.unique(newLabels))))

#mpld3.enable_notebook()

fig = plt.figure(1, facecolor='w', figsize=(10, 6))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

scatter = plt.scatter(embeddingTSNE[0], embeddingTSNE[1], s=60, c=plotFColors[newLabels])

htmlLabels = []
for rxn in rxnNames:
    htmlLabels.append("<p><b>"+rxn)

#tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=rxnNames.tolist())
tooltip = mpld3.plugins.PointHTMLTooltip(scatter, hoffset=10, voffset=15, labels=htmlLabels)
mpld3.plugins.connect(fig, tooltip)

#plt.show()
mpld3.show()


# In[80]:

#mpld3.disable_notebook()


# In[81]:

#plotFColors = plt.cm.Spectral(np.linspace(0, 1, len(np.unique(newLabels))))


# In[82]:

#embeddingISO = manifold.Isomap(30, n_components=2).fit_transform(numDat).T


# In[83]:

#plt.scatter(embeddingISO[0], embeddingISO[1], s=40, c=plotColors[typeLabels%len(plotFColors)])
#plt.show()


# In[84]:

#embeddingLLE = manifold.LocallyLinearEmbedding(100, n_components=2,method='standard').fit_transform(numDat).T


# In[85]:

#plt.scatter(embeddingLLE[0], embeddingLLE[1], s=40, c=plotFColors[newLabels])
#plt.show()


# In[86]:

#junk
#embeddingMLLE= manifold.LocallyLinearEmbedding(40, n_components=2,method='modified').fit_transform(numDat).T


# In[87]:

#plt.scatter(embeddingMLLE[0], embeddingMLLE[1], s=40, c=plotFColors[newLabels])
#plt.show()


# In[88]:

#embeddingMDS = manifold.MDS(n_components=2, n_init=1, max_iter=100).fit_transform(numDat).T


# In[89]:

#plt.scatter(embeddingMDS[0], embeddingMDS[1], s=40, c=plotFColors[newLabels])
#plt.show()


# In[90]:

#tsneDBSCAN = DBSCAN(eps=1.2, min_samples=8).fit(embeddingTSNE.T)


# In[110]:

#mpld3.disable_notebook()
#plt.scatter(embeddingTSNE[0], embeddingTSNE[1], s=40, c=plotColors[tsneDBSCAN.labels_%len(plotColors)])
#plt.show()


# In[111]:

#print ("Num Clusters: %02d " % (len(set(tsneDBSCAN.labels_)) - (1 if -1 in tsneDBSCAN.labels_ else 0)))
#print("Mutual Info: %0.3f" % metrics.adjusted_mutual_info_score(typeLabels, tsneDBSCAN.labels_))
#print("Mutual Info (Test Sets): %0.3f" % metrics.adjusted_mutual_info_score(setLabels, tsneDBSCAN.labels_))
#print np.bincount(tsneDBSCAN.labels_+1)


# In[92]:




# In[93]:

#embeddingTSNEpca = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(pcaDat).T


# In[94]:

#plt.scatter(embeddingTSNEpca[0], embeddingTSNEpca[1], s=40, c=plotColors[newLabels%len(plotColors)])
#plt.show()


# In[95]:

#tsneDat = manifold.TSNE(n_components=3, init='pca', random_state=0).fit_transform(numDat)


# In[96]:

#clusterPlot3d(tsneDBSCAN.labels_,tsneDat[:,:])


# In[96]:




# In[96]:




# In[112]:

#embedding = manifold.Isomap(30, n_components=2).fit_transform(clusterDat).T
#plt.scatter(embedding[0], embedding[1], s=40, c=plotColors[funcLabels])
#plt.show()


# In[113]:

#embedding= manifold.LocallyLinearEmbedding(40, n_components=2,method='standard').fit_transform(clusterDat).T
#plt.scatter(embedding[0], embedding[1], s=40, c=plotColors[funcLabels])
#plt.show()


# In[114]:

#embedding= manifold.LocallyLinearEmbedding(40, n_components=2,method='modified').fit_transform(clusterDat).T
#plt.scatter(embedding[0], embedding[1], s=40, c=plotColors[funcLabels])
#plt.show()


# In[100]:

#junk
#embedding= manifold.LocallyLinearEmbedding(40, n_components=2,method='hessian').fit_transform(clusterDat).T


# In[101]:

#junk
#embedding= manifold.LocallyLinearEmbedding(40, n_components=2,method='ltsa').fit_transform(clusterDat).T


# In[115]:

#embedding= manifold.MDS(n_components=2, n_init=1, max_iter=100,random_state=2).fit_transform(clusterDat).T
#plt.scatter(embedding[0], embedding[1], s=40, c=plotColors[funcLabels])
#plt.show()


# In[103]:

#junk
#embedding= manifold.SpectralEmbedding(n_components=2, random_state=0,eigen_solver="arpack").fit_transform(clusterDat).T


# In[116]:

#embedding= manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=20,early_exaggeration=1).fit_transform(clusterDat).T
#plt.scatter(embedding[0], embedding[1], s=40, c=plotColors[funcLabels])
#plt.show()


# In[ ]:



