'''
Creation of the datasets for the multi-instance experiments. Four different types of datasets are created:
BagData_Regular: Regular multi-instance data
BagData_OnePerBag: Multi-instance data with only a single instance in each bag
SingleInstance_CorrectLabels: Single-instance data with correct labeling
SingleInstance_Noisy: Single instance data where each instance in a positive bag gets a positive label
'''

from collections import defaultdict
import numpy as np
import os, data

def createHeaders(label,datasetType):
    """
    create arff header
    """
    featureList = data.createFeatureList(label,datasetType)
    
    def createBagID(i): return 'bag'+str(i)
    newBagIDs=list(map(createBagID, range(12000)))
    newBagIDs=', '.join(newBagIDs)
    
    bagHeader="@relation %s\n@attribute bagID {%s}\n@attribute bag relational\n%s@end bag\n@attribute class {0,1}\n\n@data\n" %(label,newBagIDs,''.join(featureList))
    instanceHeader="@relation %s\n\n%s@attribute class {0,1}\n\n@data\n\n" %(label,''.join(featureList))
    return bagHeader, instanceHeader

def createData(folds,label,i,outDir, instanceID, bagID, datasetType):
    
    bagHeader, instanceHeader = createHeaders(label,datasetType) 
    directories={'BagData_Regular':bagHeader,'BagData_OnePerBag':bagHeader,'SingleInstance_CorrectLabels':instanceHeader,'SingleInstance_Noisy':instanceHeader}
    for directory in directories.keys():
        if not os.path.exists(os.path.join(outDir,directory)): os.makedirs(os.path.join(outDir,directory))
        with open(os.path.join(outDir,directory,label+'_'+str(i)+'.arff'),'w') as f:
            f.write(directories[directory])

    for bagIDKey in folds.keys():
        """ Save in the appropriate formats """
        # regular bags
        instances=[]
        for instance in folds[bagIDKey]:
            bagLabel=instance[0]; instanceLabel=instance[2]; attributes=','.join(instance[3:]).strip() # instanceID None for uci data
            instances.append(attributes)
                        
            # bags with a single instance
            with open(os.path.join(outDir,'BagData_OnePerBag',label+'_'+str(i)+'.arff'),'a') as f:
                f.write("bag"+str(instanceID)+',"'+attributes+'",'+instanceLabel+'\n')
            
            # single instance correct label   
            with open(os.path.join(outDir,'SingleInstance_CorrectLabels',label+'_'+str(i)+'.arff'),'a') as f:
                f.write(attributes+','+instanceLabel+'\n')
            
            # single instance one sided noise 
            with open(os.path.join(outDir,'SingleInstance_Noisy',label+'_'+str(i)+'.arff'),'a') as f:
                f.write(attributes+','+bagLabel+'\n')
                
            instanceID+=1
        
        with open(os.path.join(outDir,'BagData_Regular',label+'_'+str(i)+'.arff'),'a') as f:
            f.write("bag"+str(bagID)+',"'+r"\n".join(instances)+',"'+bagLabel+'\n')
        
        bagID+=1
        
    return instanceID, bagID
       

####################################################################################################################################################"

def extractFold(content,outDir,label,datasetType):
    positiveBags = defaultdict(list)
    negativeBags = defaultdict(list)

    for line in content:
        l=line.split(' ')
        if l[1]=='1': positiveBags[l[0]].append(l[1:])
        else : negativeBags[l[0]].append(l[1:])
        
    if label.split('_')[0]=='adult': # Otherwise datasets too big.
        posItems=positiveBags.items(); posItems=posItems[1:600]; positiveBags=dict(posItems)
        negItems=negativeBags.items(); negItems=negItems[1:600]; negativeBags=dict(negItems)
    
    # Generating 10 folds wit each 5 positive and 5 negative instances
    instanceID=0; bagID=0
    nbInst = int(np.ceil(len(positiveBags)/10.0))
    for foldInd,i in enumerate(xrange(0,len(positiveBags),nbInst)): # 10 fold cross-validation
    
        foldP = {k: positiveBags[k] for k in positiveBags.keys()[i:i+nbInst]}
        foldN = {k: negativeBags[k] for k in negativeBags.keys()[i:i+nbInst]}
        folds=dict(foldP.items()+foldN.items())
        instanceID, bagID = createData(folds,label,foldInd, outDir, instanceID, bagID, datasetType)
        

def run(allLabels, inDir, outDir, datasetType):
    for label in allLabels:
        print label
        with open(os.path.join(inDir,label+'.txt'),'r') as f:
            content = f.readlines()[6:] # skip header
        extractFold(content,outDir,label,datasetType)

#########################################################################################################################################
# Run from terminal
#########################################################################################################################################
"""
Example:
python ./createDatasets_UCI_Text.py --inDir  '/home/me/input' --outDir '/home/me/output' --datasetType 'uci'   
"""
import argparse
from os.path import expanduser
home = expanduser("~")

def main(argv=None):

    parser = argparse.ArgumentParser(description='Create uci or text datasets in arff format from the original matlab datasets, split into ten folds to be used in 10f-cross-validation')
    parser.add_argument('--inDir', default=home, help='Directory with the original datasets in Matlab format')
    parser.add_argument('--outDir', default=home, type=str, help='Directory to save the arff datasets')
    parser.add_argument('--datasetType', type=str, help='uci or text')
    
    args = parser.parse_args(argv)
    if not os.path.exists(args.outDir):
        try:
            os.makedirs(args.outDir)
        except OSError:
            print 'Error: outDir not created'
            pass
        
    if args.datasetType == 'uci':
        datasets=['tictactoe_12', 'tictactoe_13', 'tictactoe_23','adult_12', 'adult_13', 'adult_23', 'adult_14', 'adult_24', 'adult_15', 'adult_25','transfusion_12', 'transfusion_13', 'transfusion_23', 'tictactoe_12', 'tictactoe_13', 'tictactoe_23', 'spam_12', 'spam_13', 'spam_23', 'spam_14', 'spam_24', 'spam_15', 'spam_25', 'spam_110', 'spam_210', 'adult_12', 'adult_13', 'adult_23', 'adult_14', 'adult_24', 'adult_15', 'adult_25', 'adult_110', 'adult_210', 'diabetes_12', 'diabetes_13', 'diabetes_23']
        
    elif args.datasetType == 'text':
        datasets=['alt_atheism','comp_graphics','comp_os_ms-windows_misc','comp_sys_ibm_pc_hardware','comp_sys_mac_hardware','comp_windows_x','misc_forsale','rec_autos','rec_motorcycles',\
            'rec_sport_baseball','rec_sport_hockey','sci_crypt','sci_electronics','sci_med','sci_space','soc_religion_christian','talk_politics_guns','talk_politics_mideast','talk_politics_misc','talk_religion_misc']
    else: print 'Error, incorrect dataset type'    
        
    run(datasets, args.inDir, args.outDir,args.datasetType)
    

import sys
if __name__ == "__main__":
    sys.exit(main())  


########################################################################################################
 
     
def changeFileNames(ftype,fdir):
    """
    Change filenames to remove dots (does not work with Weka) 
    """
    for f in os.listdir(fdir):
        fileName, ext = f.rsplit('.',1)
        if ftype == 'text': newName = fileName.replace('.','_')
        if ftype == 'uci': newName = fileName.replace('.','_',1).replace('.','')
        os.rename(os.path.join(fdir,f),os.path.join(fdir,newName+'.'+ext))  
