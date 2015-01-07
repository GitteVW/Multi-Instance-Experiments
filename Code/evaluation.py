'''
Running multi-instance experiments
'''
import os.path

def evaluateClassifiers(train, test, test2, outputDirectory, classifierName):
        print classifierName+' '+train+' '+test
        options=''; p=''
        # MI algorithms (except AdaBoost)
        if classifierName in ['weka.classifiers.mi.MIOptimalBall','weka.classifiers.mi.MIEMDD','weka.classifiers.mi.MISVM', 'weka.classifiers.mi.TLD','weka.classifiers.mi.MDD',\
                              'weka.classifiers.mi.MIDD']:
            options=''
            
        if classifierName=='weka.classifiers.mi.MILR':
            options=options='-A 0' 
        if classifierName=='weka.classifiers.mi.MIWrapper':
            options = '-P 1 -A 1 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2'
        if classifierName=='weka.classifiers.mi.SimpleMI':
            options = '-M 1 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2'
        if classifierName=='weka.classifiers.mi.CitationKNN':
            options = '-R 5 -C 5'  
        if classifierName =='weka.classifiers.mi.MISMO':
            options='-M -K weka.classifiers.mi.supportVector.MIRBFKernel'
        
        if classifierName == 'weka.classifiers.mi.MIRI':
            options='-L -K 0 -M 1'
        if classifierName == 'weka.classifiers.mi.MILR_C':
            classifierName = 'weka.classifiers.mi.MILR'
            options='-A 1' 
        
        # AdaBoost, multi-instance and single-instance settings
        if classifierName =='weka.classifiers.meta.AdaBoostM1':
            options='-W weka.classifiers.trees.J48'
        if classifierName =='weka.classifiers.mi.AdaBoostM1':
            classifierName = 'weka.classifiers.meta.AdaBoostM1'
            options='-W weka.classifiers.mi.MITI -- -K 0 -M 1'
            
        # SI algorithms
        if classifierName=='weka.classifiers.trees.J48':
            options = '-C 0.25 -M 2'
        if classifierName=='weka.classifiers.lazy.IBk':
            options = '-K 5'
        if classifierName=='weka.classifiers.functions.RBFNetwork':
            options = '-B 1'
        if classifierName=='weka.classifiers.functions.SMO': # number of folds internal cv not necessary bc use gridsearch or cvparameterselection and there defined
            options='-M -K weka.classifiers.functions.supportVector.RBFKernel'
        if classifierName=='weka.classifiers.functions.Logistic':
            options =''

        
        if test2:
            strTest2=' -test2 '+test2
        else:
            strTest2=''
        
        cmd = 'java -jar cv.jar -train '+train+' -test '+test+strTest2+' -W "'+classifierName+' '+str(options)+'"'+' -out '+outputDirectory+" "+p
        print cmd
        os.system(cmd)  


def createTrainingset(datasetDir, label, fold):
    
    folds=set(range(10)).difference(set([fold]))
    print folds
    train=[]
    for ftmp in folds:
        f = open(os.path.join(datasetDir,label+'_'+str(ftmp)+'.arff'),'r')
        content=f.read()
        header,data = content.split('@data')
        train.append(data)
    train=''.join(train)
    newContent=header+'@data\n'+train
    with open(os.path.join(datasetDir,label+'_'+str(fold)+'_train.arff'),'w') as f:
        f.write(newContent)
    return os.path.join(datasetDir,label+'_'+str(fold)+'_train.arff')

def doExperimentFor(labels, datasetDir, outputDirectory, classifierList, secondDatadir, folds):
    for label in labels:
        print label
        for fold in folds:
            print '------------------------------------------------------\nfold\n---------------------------------------------------------------'
            print fold
            train = createTrainingset(datasetDir, label, fold)
            test=os.path.join(datasetDir,label+'_'+str(fold)+'.arff')            
            if secondDatadir: test2=os.path.join(secondDatadir,label+'_'+str(fold)+'.arff')
            else: test2='' 
            
            for classifierName in classifierList:
                evaluateClassifiers(train,test,test2,outputDirectory,classifierName)

#################################################################################################################################################################################### 
import argparse
from os.path import expanduser
home = expanduser("~")

def main(argv=None):
    
    parser = argparse.ArgumentParser(description='Running multi instance experiments')
    parser.add_argument('--labels', metavar='N', type=str, nargs='+',help='List of labels, optional. If not specified, all datasets present in the dataPath folder are evaluated.')
    parser.add_argument('--datatype', type=str, help='Datatype: BagLevel or Noisy')
    parser.add_argument('--outPath', default=os.path.join(home,'results'), type=str, help='Directory where the prediction results are written')
    parser.add_argument('--dataPath', default=os.path.join(home,'data'), type=str, help='Directory where the intermediately generated training and test files should be written to.')
    parser.add_argument('--dataPath2', default='', type=str, help='Directory for the second type of test files (optional).')
    parser.add_argument('--classifiers', default=[], type=str,  nargs='+', help='Optional. If not specified, all classifiers are evaluated.')
    parser.add_argument('--folds', default=range(10), type=int,  nargs='+', help='Optional. If not specified, all 10 folds are evaluated.')
    
    args = parser.parse_args(argv)
    print args
    if not os.path.exists(args.outPath):
        try: os.makedirs(args.outPath)
        except OSError: print 'Error: outPath directory not created'; pass

        
    if args.datatype == 'Noisy' and not args.classifiers:
        listClassifiers=['weka.classifiers.meta.AdaBoostM1','weka.classifiers.functions.RBFNetwork', 'weka.classifiers.trees.J48','weka.classifiers.functions.SMO','weka.classifiers.lazy.IBk','weka.classifiers.functions.Logistic']
    if (args.datatype == 'BagLevel' or args.datatype=='InstanceLevel') and not args.classifiers:
        listClassifiers=['weka.classifiers.mi.MIRI','weka.classifiers.mi.MIDD','weka.classifiers.mi.AdaBoostM1','weka.classifiers.mi.MIOptimalBall','weka.classifiers.mi.MILR','weka.classifiers.mi.MIEMDD','weka.classifiers.mi.MISVM','weka.classifiers.mi.TLD','weka.classifiers.mi.MISMO','weka.classifiers.mi.MDD','weka.classifiers.mi.MIWrapper','weka.classifiers.mi.SimpleMI','weka.classifiers.mi.CitationKNN']
    if args.classifiers and (args.datatype=='InstanceLevel' or args.datatype=='BagLevel'):
        listClassifiers=['weka.classifiers.mi.'+x for x in args.classifiers]
    if args.classifiers and args.datatype=='Noisy':
        listClassifiers=['weka.classifiers.'+x for x in args.classifiers]

    if not args.datatype and not args.classifiers:
        print 'Warning, datatype not set, no experiments'
        listClassifiers=[]
    if not args.labels:
        labels=['feltflowerrug','ajaxorange','apple','banana','bluescrunge','dirtyworkgloves','juliespot','checkeredscarf','wd40can','candlewithholder','glazedwoodpot','cokecan','smileyfacedoll','dataminingbook','rapbook','translucentbowl','greenteabox','cardboardbox','dirtyrunningshoe','largespoon','goldmedal','spritecan','stripednotebook','woodrollingpin','fabricsoftenerbox']
    if args.labels:
        labels=args.labels
    
    doExperimentFor(labels, args.dataPath, args.outPath, listClassifiers, args.dataPath2, args.folds)

import sys
if __name__ == "__main__":
    sys.exit(main())   
