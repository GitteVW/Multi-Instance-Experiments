'''
In a first step the experiments where run and the individual predictions saved in files.
This script processes these results, computes different performance metrics (accuracy, AUC, weighted accuracy 0.5*(TP+TN)), and saves these processed results to new files.
One file is created for each dataset that contains the results for all algorithms
'''
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import os


def extractLabelsFromFile(filename):
    import csv
    prob0=[]; prob1=[]; truelabels=[]
    
    with open(filename,'r') as f:
        reader=csv.reader(f, delimiter=',')
        for row in reader:
            prob0.append(float(row[0]))
            prob1.append(float(row[1]))
            truelabels.append(int(float(row[2])))
    return prob0,prob1,truelabels

         
def run(inDir, outDir, algorithms, datasets):
    """
    @param inDir: The directory with the files that contain the individual predictions for each train/test case
    @param outDir: The directory where to save the processed results
    @param algorithms: List of algorithms (string)
    @param datasets: List of dataset labels (string)
    """
    
    for dataset in datasets: open(os.path.join(outDir,dataset+'.txt'),'w').close()
    
    for algo in algorithms:
        for dataset in datasets:
            alltruelabels=[]; allprobs1=[]; foldAccuracies=[]; balancedAcc=[]; nbPredictions=0; nbCorrect=0
            for fold in range(10):
                
                print algo, dataset, str(fold)
                try:
                    prob0,prob1,truelabels = extractLabelsFromFile(os.path.join(inDir,algo+'_'+dataset+'_'+str(fold)+'.txt'))
                except:
                    print "Skipping dataset/learner combination"
                    continue
                allprobs1=allprobs1+prob1
                alltruelabels = alltruelabels+truelabels
                
                predictions=[int(pred1 > pred0) for (pred0,pred1) in zip(prob0,prob1)]
                foldAccuracies.append(accuracy_score(truelabels,predictions))
                TPR=sum((np.array(truelabels)==1) * (np.array(predictions)==1))/float(sum(np.array(truelabels)==1))
                TNR=sum((np.array(truelabels)==0) * (np.array(predictions)==0))/float(sum(np.array(truelabels)==0))
                balancedAcc.append(0.5*(TPR+TNR))
                
                nbPredictions+=len(predictions) # for binomial proportion test
                nbCorrect+=sum(np.array(truelabels)==np.array(predictions))
             
            pooledAUC = roc_auc_score(alltruelabels,allprobs1);
            acc = np.mean(foldAccuracies)
            balancedAccuracy=np.mean(balancedAcc)
            
            with open(os.path.join(outDir,dataset+'.txt'),'a') as f:
                f.write(', '.join([algo, str(acc), str(pooledAUC),str(balancedAccuracy), str([nbPredictions,nbCorrect])+'\n']))

#########################################################################################################################################
# Run from terminal
#########################################################################################################################################

import argparse

def main(argv=None):

    parser = argparse.ArgumentParser(description='Compute from the individual predictions accuracy, pooled AUC and weighted accuracy')
    parser.add_argument('--inDir', metavar='./', type=str, help="The directory with the files that contain the individual predictions for each train/test case")
    parser.add_argument('--outDir', default='./', type=str, help="The directory where to save the processed results")
    parser.add_argument('--algorithmType', type=str, help='BagLevel, Noisy')
    parser.add_argument('--datasetType', type=str, help='uci, text, sival')
    
    args = parser.parse_args(argv)
    if not os.path.exists(args.outDir):
        try:
            os.makedirs(args.outDir)
        except OSError:
            print 'Error: outDir not created'
            pass
        
    if args.algorithmType=='BagLevel' and (args.datasetType=='uci' or args.datasetType=='text' or args.datasetType=='sival'):
        algorithms=['MILR','AdaBoostM1','MILRC','MIRI','MIDD','MDD','MIOptimalBall','TLD','MIWrapper','CitationKNN','MISMO','SimpleMI','MIEMDD','MISVM']
    elif args.algorithmType=="Noisy":
        algorithms=['AdaBoostM1','RBFNetwork', 'J48','SMO','IBk','Logistic']
    else: algorithms= args.algorithms
    
    if args.datasetType == 'uci':
        datasets=['transfusion_12', 'transfusion_13', 'transfusion_23', 'tictactoe_12', 'tictactoe_13', 'tictactoe_23', 'spam_12', 'spam_13', 'spam_23', 'spam_14', 'spam_24', 'spam_15', 'spam_25', 'spam_110', 'spam_210', 'adult_12', 'adult_13', 'adult_23', 'adult_14', 'adult_24', 'adult_15', 'adult_25', 'adult_110', 'adult_210', 'diabetes_12', 'diabetes_13', 'diabetes_23']
    elif args.datasetType == 'text':
        datasets=['alt_atheism','comp_graphics','comp_os_ms-windows_misc','comp_sys_ibm_pc_hardware','comp_sys_mac_hardware','comp_windows_x','misc_forsale','rec_autos','rec_motorcycles',\
            'rec_sport_baseball','rec_sport_hockey','sci_crypt','sci_electronics','sci_med','sci_space','soc_religion_christian','talk_politics_guns','talk_politics_mideast','talk_politics_misc','talk_religion_misc']
    elif args.datasetType == 'sival':
        datasets=['feltflowerrug','ajaxorange','apple','banana','bluescrunge','dirtyworkgloves','juliespot','checkeredscarf','wd40can','candlewithholder','glazedwoodpot','cokecan','smileyfacedoll','dataminingbook','rapbook','translucentbowl','greenteabox','cardboardbox','dirtyrunningshoe','largespoon','goldmedal','spritecan','stripednotebook','woodrollingpin','fabricsoftenerbox']
    else: print 'Error, incorrect dataset type'    
    run(args.inDir, args.outDir, algorithms, datasets)
    

import sys
if __name__ == "__main__":
    sys.exit(main())  
