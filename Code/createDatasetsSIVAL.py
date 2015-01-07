'''
Creation of the datasets for the multi-instance experiments. Four different types of datasets are created:
BagData_Regular: Regular multi-instance data
BagData_OnePerBag: Multi-instance data with only a single instance in each bag
SingleInstance_CorrectLabels: Single-instance data with correct labeling
SingleInstance_Noisy: Single instance data where each instance in a positive bag gets a positive label

@author: gitte
'''
import os,re
import numpy as np

    
def createHeaders(label):
    def createFeature(i): return '\t@attribute f'+str(i)+': numeric\n'
    featureList = list(map(createFeature, range(30)))
    
    def createBagID(i): return 'bag'+str(i)
    newBagIDs=list(map(createBagID, range(3850)))
    newBagIDs=', '.join(newBagIDs)
    
    bagHeader="@relation %s\n@attribute bagID {%s}\n@attribute bag relational\n%s@end bag\n@attribute class {0,1}\n\n@data\n" %(label,newBagIDs,''.join(featureList))
    instanceHeader="@relation %s\n\n%s@attribute class {0,1}\n\n@data\n\n" %(label,''.join(featureList))
    return bagHeader, instanceHeader

def createData(folds,label,i, outDir, bagid):
    
    bagHeader, instanceHeader = createHeaders(label) 
    directories={'BagData_Regular':bagHeader,'BagData_OnePerBag':bagHeader,'SingleInstance_CorrectLabels':instanceHeader,'SingleInstance_Noisy':instanceHeader}
    for directory in directories.keys():
        if not os.path.exists(os.path.join(outDir,directory)): os.makedirs(os.path.join(outDir,directory))
        with open(os.path.join(outDir,directory,label+'_'+str(i)+'.arff'),'w') as f:
            f.write(directories[directory])

    
    for bag in folds:
        p=re.search('(?P<bagID>.+),"(?P<instances>.+)",(?P<bagLabel>\d)',bag)
        instances = (p.group('instances')).split(r'\n')
        
        with open(os.path.join(outDir,'BagData_Regular',label+'_'+str(i)+'.arff'),'a') as f:
            f.write(','.join(['bag'+str(bagid),'"'+p.group('instances')+'"',p.group('bagLabel')+'\n']))
        
        for instance in instances:
            splitInstance=instance.split(','); 
            attributes=splitInstance[:-1]; instanceLabel=splitInstance[-1]
            
            with open(os.path.join(outDir,'BagData_OnePerBag',label+'_'+str(i)+'.arff'),'a') as f:
                f.write(','.join(['bag'+str(bagid),'"'+','.join(attributes)+'"',instanceLabel+'\n']))
                
            with open(os.path.join(outDir,'SingleInstance_CorrectLabels',label+'_'+str(i)+'.arff'),'a') as f:
                f.write(','.join(attributes+[instanceLabel+'\n']))
                
            with open(os.path.join(outDir,'SingleInstance_Noisy',label+'_'+str(i)+'.arff'),'a') as f:
                f.write(','.join(attributes+[p.group('bagLabel')+'\n']))
            
            bagid+=1
    
    f.close()
    
    return bagid

def extractDataset(label, outDir, content):
    """
    Extracting a subset of all bags from a given label (60), and 60 random bags from the other labels.
    ...
    """
    # 
    positives=np.array([])

    for i,line in enumerate(content):
        bag=re.search(label,line,re.IGNORECASE)
        if bag: 
            positives=np.append(positives,line)
    
    negatives=set(content).difference(positives) # speedup in comparison to append
    np.random.seed(10); indices=np.random.randint(low=0,high=len(negatives),size=60)
    negatives=np.array(list(negatives))[indices]
    
    # Generating 10 folds wit each 6 positive and 6 negative instances
    nbInst = int(np.round(len(positives)/10.0))
    bagid=0
    for foldInd,i in enumerate(xrange(0,len(positives),nbInst)): # 10 fold cross-validation
        foldP = positives[i:i+nbInst]; foldN = negatives[i:i+nbInst]
        folds=np.append(foldP,foldN)
        bagid = createData(folds,label,foldInd, outDir,bagid)
        
def run(labels, arffDir, outDir):
    for label in labels:
        print label
        with open(os.path.join(arffDir,label+'.arff'),'r') as f:
            content=f.readlines()
        extractDataset(label, outDir, content)

#########################################################################################################################################
# Run from terminal
#########################################################################################################################################

import argparse
from os.path import expanduser
home = expanduser("~")

def main(argv=None):

    parser = argparse.ArgumentParser(description='Create sival datasets in arff format from the original datasets, split into ten folds to be used with 10f-cross-validation')
    parser.add_argument('--labels', metavar='N', type=str, nargs='+',help='list of labels for which arff datasets need to be created. If not defined, datasets are created for all the labels.')
    parser.add_argument('--arffDir', default=home,help='Directory with the ARFF files as created by "toARFFSIVAL"')
    parser.add_argument('--outDir', default=home, type=str, help='Directory where the datasets are written to')
    
    args = parser.parse_args(argv)
    if not os.path.exists(args.outDir):
        try:
            os.makedirs(args.outDir)
        except OSError:
            print 'Error: outDir not created'
            pass
        
    if not args.labels:
        labels=['feltflowerrug','ajaxorange','apple','banana','bluescrunge','dirtyworkgloves','juliespot','checkeredscarf','wd40can','candlewithholder','glazedwoodpot','cokecan','smileyfacedoll','dataminingbook',\
        'rapbook','translucentbowl','greenteabox','cardboardbox','dirtyrunningshoe','largespoon','goldmedal','spritecan','stripednotebook','woodrollingpin','fabricsoftenerbox']
    if args.labels:
        labels=args.labels
        
    run(labels,args.arffDir,args.outDir)

import sys
if __name__ == "__main__":
    sys.exit(main())  
