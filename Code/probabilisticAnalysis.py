'''
Probabilistic analysis Section 4.2.2
'''
import data
import os,re,sys
import numpy as np
import itertools

def getResultsForDataset(path, dataset, metrics): 
    """
    @param path: the directory with the performance results (string)
    @param dataset: Dataset for which to compute the probabilities (string)
    @return: A list of dictionaries with as key the classifier name and as value the metric value for the given dataset
    """
    with open(os.path.join(path,dataset+'.txt'), 'r') as f:
        content = f.readlines()
    
    d={}
    for line in content:
        results = [x.strip('\n  ') for x in line.split(',')]

        d[results[0]]=results[1:]
    
    dicts=[]   
    values=np.array(d.values())
    for i in range(len(metrics)):
        tmpValues=values[:,i]
        dicts.append({k:float(v) for (k,v) in zip(d.keys(),tmpValues)})
    return dicts


def isLarger(vals):
    isLarger=[]
    values=np.array(vals.values())
    for pair in itertools.permutations(values, 2):
        isLarger.append(pair[0]>pair[1])
    return np.array(isLarger)

def compute_conditional_probs(vals1,vals2):
    """
    Point estimates of the conditional probabilities
    """
    larger1=isLarger(vals1); larger2=isLarger(vals2)
    nominator=larger1 * larger2 # and
    
    cond1=float(sum(nominator))/sum(larger1)
    cond2=float(sum(nominator))/sum(larger2)

    return cond1, cond2

def compute_binomial_ci(vals1,vals2):
    """
    95% Wilson confidence intervals for the conditional probabilities. Prob(vals1|vals2) 
    @param vals1: List with performance results of all algorithms for one dataset. 
    @param vals2: List with performance results of all algorithms for one dataset. 
    """
    larger1=isLarger(vals1); larger2=isLarger(vals2)
    nominator=larger1 * larger2 # AND
    
    n=sum(larger1) # always 91 except when two algorithms have identical performance (choosing 2 out of 14)
    q=float(sum(nominator))/sum(larger1)
    
    phat=q
    z=1.96 # z-score for 95% confidence interval
    return [((phat + z*z/(2*n) - z * np.sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n)),((phat + z*z/(2*n) + z * np.sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n))] # wilson 

def printLatexTable(resultDic, datasetType):
    """
    Latex table with point estimates of the probabilities
    """
    if datasetType=='uci':
        
        uciDatasets=['Adult','Diabetes','Spam','Tic-Tac-Toe','Transfusion']
        ratios = ['1/2','1/3','2/3','1/4','2/4','1/5','2/5','1/10','2/10']
        
        for i in range(2): # Prob(vals1|vals2)  and Prob(vals2|vals1) 
            line=[]
            print '\hline'
            print ' & '+' & '.join(ratios)+r' \\'
            print '\hline'
            for uciDataset in uciDatasets:
                line.append(uciDataset)
                for ratio in ratios:
                    ds='_'.join([uciDataset.lower().replace('-',''),ratio.replace('/','')])
                    if ds in resultDic.keys():
                        line.append('{0:.2f}'.format(resultDic[ds][i]))
                    else:
                        line.append('-')
                sys.stdout.write(' & '.join(line)+' \\\ \n')
        print '\hline'
        
    else:
        items=resultDic.items()
        half = int(np.ceil(len(items)/float(2))); odd=half%2
        for i in range(half-odd): # dataset, if odd, leave out last item and put in table later
            sys.stdout.write('{0} & {1:.2f} & {2:.2f} & '.format(re.sub('_',' ',items[i][0]),items[i][1][0],items[i][1][1]))
            sys.stdout.write('{0} & {1:.2f} & {2:.2f} \\\ \n'.format(re.sub('_',' ',items[half+i][0]),items[half+i][1][0],items[half+i][1][1]))
        if odd:
            sys.stdout.write('{0} & {1:.2f} & {2:.2f} \\\ '.format(re.sub('_',' ',items[half][0]),items[half][1][0],items[half][1][1]))
        print "\n----------------"

def printDatasetName(dataset,datasetType):

    if datasetType=='uci':
        ds, ratio = dataset.split('_')
        source=ds[0].upper()+ds[1:]
        rat=ratio[0]+'/'+ratio[1:]
        return ' '.join([source,rat])
    else:
        return dataset
            
def printLatexTable2(resultDic, datasetType):
    """ 
    Latex table with confidence intervals. Only Prob(vals1|vals2), as the confidence interval for Prob(vals2|vals1) is almost identical 
    """
    
    datasets = data.datasets[datasetType]

    half = len(datasets)/2; odd=half%2
    for i in range(half): # dataset, if odd, leave out last item
        datasetLeft=datasets[i]; datasetRight=datasets[half+i+odd]
        resultLeft=resultDic[datasetLeft]; resultRight=resultDic[datasetRight]
        sys.stdout.write('{0} & $[{1:.2f}, {2:.2f}]$ &'.format(re.sub('_',' ',printDatasetName(datasetLeft,datasetType)),resultLeft[0],resultLeft[1]))
        sys.stdout.write(' {0} & $[{1:.2f}, {2:.2f}]$ \\\ \n'.format(re.sub('_',' ',printDatasetName(datasetRight,datasetType)),resultRight[0],resultRight[1]))
    if odd:
        dataset=datasets[half]
        result=resultDic[dataset]
        sys.stdout.write('{0} & $[{1:.2f}, {2:.2f}]$ &'.format(re.sub('_',' ',printDatasetName(dataset,datasetType)),result[0],result[1]))
        sys.stdout.write(' & \\\ ')


def run(metrics, inDir1, inDir2, datasetType):
    """
    @param metrics: A list of performance metrics. Can contain "acc", "auc" and "bacc" (Balanced ACCuracy).
    @param inDir1: Directory to the results as created by "processExperimentResults.py" that will be plotted in the first column of the figure. Example: "./Results/processed_'uci/BagData_OnePerBag" 
    @param inDir2: Directory to the results as created by "processExperimentResults.py" that will be plotted in the first column of the figure. Example: "./Results/processed_'uci/BagData_Regular" 
    @param datasetType: 'uci', 'text' or 'sival'
    
    Example: run(['acc','auc','bacc'], './Results/processed_sival/BagData_Regular', './Results/processed_sival/BagData_OnePerBag','sival')
    """
    
    dics1={}; dics2={}
    for dataset in data.datasets[datasetType]:
        dics1[dataset] = getResultsForDataset(inDir1, dataset, metrics)
        dics2[dataset] = getResultsForDataset(inDir2, dataset, metrics)
        
    for m,metric in enumerate(metrics):
        print '\n---------------------------------\n'+metric+'\n-------------------------'; 
        results={}
        for dataset in data.datasets[datasetType]:
            data1=dics1[dataset][m]
            data2=dics2[dataset][m]
            confidenceInterval = compute_binomial_ci(data1,data2) 
            results[dataset]=confidenceInterval
        printLatexTable2(results, datasetType)


