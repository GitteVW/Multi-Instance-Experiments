'''
Creating a visualization of the rank correlation between bag-level and instance-level performance of the multi-instance algorithms.
Section 4.2.1 in the paper.
'''

import data
import operator
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os.path
import scipy.stats

def getResultsForDataset(path, dataset, metrics): 
    """
    @param path: Directory with the results of running the mi algorithms (string)
    @param dataset: Dataset for which to process the results (string)
    @param metrics: List of performance metrics for which to create figures (list)
    @return: A list of dictionaries with as keys the classifier names and as values the performance of that classifier for the given dataset. One dictionary for each metric (list)
    """
    with open(os.path.join(path,dataset+'.txt'), 'r') as f:
        content = f.readlines()
    
    d={}
    for line in content:
        results = [x.strip('\n  ') for x in line.split(',')]
    
        if results[0]=='CitationKNN':
            results[0]='Citation-kNN5'
        if results[0]=='MIWrapper':
            results[0]='MIWrapper-J48'
        if results[0]=='SimpleMI':
            results[0]='SimpleMI-J48'
        if results[0]=='TLDSimple':
            continue
        d[results[0]]=results[1:]
    
    dicts=[]   
    values=np.array(d.values())
    for i in range(len(metrics)):
        tmpValues=values[:,i]
        dicts.append({k:float(v) for (k,v) in zip(d.keys(),tmpValues)})
    return dicts

    
def createAccList(a1,a2):
    """           
     @param a1: Dictionary with key=classifier name and value performance
     @param a2: Dictionary with same structure as a1 
     @return: List of lists which has the following structure: [classname, a1, i, a2, j] with i and j the ranks of a1 and a2
     """

    index={};
    for k in a1.iterkeys(): # create index dictionary with all indices set to 0
        index[k]=0;
      
    dics=[a1,index,a2,index] # merge all dictionaries
    d={}
    for k in a1.iterkeys():
        d[k]=list([x[k] for x in dics]) # Put all performances and ranks per classifier key. 
    dlist=[]
    for k in d.iterkeys():
        subl = d[k]
        newsub=[k]
        for el in subl:
            newsub.append(el)
        dlist.append(newsub)
        
    dlist = sorted(dlist, key=operator.itemgetter(3), reverse=True) # sort descending on a1
    for i in range(len(d)):
        dlist[i][4]=i
        dlist[i][3]='{0:.3f}'.format(dlist[i][3]) 
        
    dlist = sorted(dlist, key=operator.itemgetter(1), reverse=True) # sort descending on a2
    for i in range(len(d)):
        dlist[i][2]=i
        dlist[i][1]='{0:.3f}'.format(dlist[i][1]) 
      
    return dlist


def drawColumns(li,pathToSaveFig,metric): 
    """
    @param li: List of two lists with each nested list consists of 4 elements: [classifier name, a1, rank1, a2, rank2]
    @return:  Visualization of the two lists in "li" where each list is sorted according to algorithmic performance and there are links between the two lists for performance results of the same classifier
    """
    
    columnName={'acc':'a','auc':'AUC','bacc':'bal. a'}
    x1=0.21; x2=x1+0.27 # column 1 and 2 
    ystart=0.95 # where to start the top of a column on the y axis
    yh=0.06 # how high is one number?
    tw=0.083 # text width for first column, to start line
    tw2=-0.005 # slightly left for second column to start line
    th=0.01 # middle of the number to start line from
    fig=plt.figure(facecolor='white')
    ax=plt.axes(frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.text(x1+0.05,ystart+yh,'$\mathbf{'+columnName[metric]+'_I}$', weight='bold',horizontalalignment='center',fontsize=15) 
    plt.text(x2+0.04,ystart+yh,'$\mathbf{'+columnName[metric]+'_B}$', weight='bold',horizontalalignment='center',fontsize=15)
    for el in li:

        plt.text(0.01,ystart-el[2]*yh,el[0], family='Lato', weight='semibold',size=12) # classifier name
        
        plt.text(x1+0.115,ystart-el[2]*yh, el[2]+1, family='Liberation Sans', weight='bold' ,horizontalalignment='right', size=13) # rank 1 
        plt.text(x1,ystart-el[2]*yh,el[1], family='Liberation Sans', weight='normal',size=13) # performance 1
        plt.text(x2-0.045,ystart-el[2]*yh,el[2]+1, family='Liberation Sans', weight='bold', horizontalalignment='left', size=13) # rank 2
        plt.text(x2,ystart-el[4]*yh,el[3], family='Liberation Sans',weight='normal', size=13) # performance 2
       
        plt.plot([x1+tw+0.043,x2+tw2-0.047],[ystart-el[2]*yh+th,ystart-el[4]*yh+th],'k') # connecting line
    plt.xlim([0,1]);plt.ylim([0,1])
    plt.savefig(pathToSaveFig)
    plt.close()

    
def computeCorrelationCoef(li):
    
    accs1=[]; accs2=[]
    d = defaultdict(list) 
    for i in range(len(li)):
        d[li[i][0]].append(li[i][1:])
        accs1.append(li[i][1])
        accs2.append(li[i][3])
    return scipy.stats.spearmanr(accs1, accs2)[0]


def mergeDics(dtype,dataset,metric,inDir):
    
    allDicBL=[];allDicIL=[]

    dics=[]
    for datatype in ['BagLevel','InstanceLevel']:
        dic=getResultsForDataset(os.path.join(inDir,datatype),dataset)
        dics.append(dic)
    allDicBL.append(dics[0]);allDicIL.append(dics[1])
    
    results=[]
    for i in range(2):
        d={}
        for dic in allDicBL:
            for k in dic.iterkeys():
                d[k]=list([x[k] for x in allDicBL])
        for k in d.iterkeys():
            d[k]=np.mean(d[k])
        results.append(d)

    return results

###############################################################################"

def createFigures(dir1, dir2, dataset, resultDir, metrics):
    """
    @param dir1: Directory with the performance results of the first type (Bag-level -> BagData_Regular, or instance-level -> BagData_OnePerBag) (string)
    @param dir2: Directory with the performance results of the first type (Bag-level or instance-level) (string)
    @param dataset: The dataset for which to create a figure (string)
    @param resultDir: The directory where to save the figures (string)
    @param metrics: List of performance metrics for which to create figures

    """
    
    dic1=getResultsForDataset(dir1 ,dataset, metrics)
    dic2=getResultsForDataset(dir2 ,dataset, metrics)
    for i,metric in zip(range(len(dic1)),metrics):
        resultDirMetric=os.path.join(resultDir,metric)
        if not os.path.exists(resultDirMetric):
            os.makedirs(resultDirMetric)
        
        dics=[dic1[i],dic2[i]]
        li = createAccList(dics[0],dics[1])
        
        drawColumns(li,os.path.join(resultDirMetric,dataset+'_uncropped.pdf'),metric)
        os.system('python pdf_crop.py -m "135 0 225 70" -i '+os.path.join(resultDirMetric,dataset+'_uncropped.pdf')+' -o '+os.path.join(resultDirMetric,dataset+'_fully_cropped.pdf')) # how much cutting left top right bottom
        os.system('python pdf_crop.py -m "45 0 225 70" -i '+os.path.join(resultDirMetric,dataset+'_uncropped.pdf')+' -o '+os.path.join(resultDirMetric,'_'.join(['linked',metric,dataset+'.pdf'])))


##########################################################################################################

def run(datasetTypes,metrics,inDir1,inDir2,outDir='./Figures/LinkedColumns/'):
    """
    @param datasetTypes: A list of dataset types for which to generated figures. Can contain "sival", "uci" and "text".
    @param metrics: A list of performance metrics. Can contain "acc", "auc" and "bacc" (Balanced ACCuracy).
    @param inDir1: Directory to the results as created by "processExperimentResults.py" that will be plotted in the first column of the figure. Example: "./Results/processed_'uci/BagData_OnePerBag" 
    @param inDir2: Directory to the results as created by "processExperimentResults.py" that will be plotted in the first column of the figure. Example: "./Results/processed_'uci/BagData_Regular" 
    @param outDir: Directory where the Figures will be saved.
    """

    for labelType in datasetTypes:
        print labelType
        alls=[]
        for label in data.datasets[labelType]:
            res=createFigures(inDir1, inDir2, label, outDir+labelType,metrics)

