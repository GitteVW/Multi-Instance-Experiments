'''
Creating scatterplots Section 4.3.2 and 5
'''


import data

import matplotlib.pyplot as plt
import numpy as np
import os.path

def getResults(path,datasetType): 
    """
    @param path: The directory with the processed data
    @return: A nested dictionary with as outer keys the classifier names, as inner keys the dataset names, and as values the performance results of the algorithm for the given dataset
    """
    from collections import defaultdict
    dic=defaultdict(defaultdict)
    for dataset in data.datasets[datasetType]:
        with open(os.path.join(path,dataset+'.txt'), 'r') as f:
            content = f.readlines()
        for line in content:
            results = [x.strip('\n  ') for x in line.split(',')]
            dic[results[0]][dataset]=results[1:]
    
    return dic

def drawScatterplots(datasetTypes,metrics,superDir,subDir1,subDir2,outDir,isBaglevel):
    """
    @param datasetTypes: List of dataset categories that can contain as items: 'uci', 'text', 'sival'.
    @param metrics: List of metrics that can contain as items: 'acc', 'bacc', 'auc'.
    @param superDir: Top directory that contains the results.
    @param subDir i: BagData_Regular, BagData_OnePerBag. The types of results that are plotted in the x and y direction.
    @param outDir: Directory where to save the scatter plots.
    @param isBaglevel: Boolean indicating whether 'BagData_Regular' versus 'BagData_OnePerBag' is plotted, or 'BagData_OnePerBag' versus 'Noisy'. This determines the axis labels.
    """
    
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    
    edgecolors={'uci':'Blue','uci_nominal':'Blue','text':'DarkGreen','sival':'Red'} 
    markers={'uci':'^','uci_nominal':'^','text':'o','sival':'s'}
    
    allData1={}; allData2={}
    for datasetType in datasetTypes: 
        allData1[datasetType]=getResults(os.path.join(superDir,'processed_'+datasetType,subDir1),datasetType)
        allData2[datasetType]=getResults(os.path.join(superDir,'processed_'+datasetType,subDir2),datasetType)
    
    if isBaglevel:
        learnerTuples=[(x,x) for x in data.milearners]
        xlabel='instance-level'; ylabel='bag-level'
    else:
        learnerTuples=data.learnerTuples
        xlabel='multi-instance'; ylabel='single-instance'
    for m,metric in enumerate(metrics):
        for learner1,learner2 in learnerTuples: # One scatter plot per performance metric and classifier tuple, with each point on the plot representing one dataset
            
            print metric,learner1
            plt.figure()
            plt.axis([0,1,0,1],fontsize=18)
            plt.xlabel(xlabel,fontsize=28)
            plt.ylabel(ylabel,fontsize=28)
            plt.tick_params(axis='both', which='major', labelsize=28)
            plt.plot([0, 1], [0, 1], 'k--',linewidth=0.5)
            plt.tight_layout()
            
            for datasetType in datasetTypes:
                data1 = allData1[datasetType][learner1].values()
                data1 = np.array(data1)[:,m]
                
                data2 = allData2[datasetType][learner2].values()
                data2 = np.array(data2)[:,m]      
                
                plt.scatter(data1,data2,facecolors='none', edgecolors=edgecolors[datasetType],s=120,marker=markers[datasetType]) 
                plt.draw()     
            
            plt.savefig(os.path.join(outDir,'si_vs_mi_'+learner1+'_'+metric+'.pdf'))
            plt.close()

        