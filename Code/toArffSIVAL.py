'''
This script converts the c4.5 format datasets to arff format. No partitioning into folds happens yet. 

Data format (Settles 2008) 1499 images (1500 images where manually labeled and one was discarded). 1 image (=bag) contains approximately 30 segments (usually 31 or 32). One segment or instance contains 30 features.

'''
import os.path

def splitLine(line):
    segment=line.split(',')
    bagID=segment[0]
    features=segment[2:-1]+[str(int(float(segment[-1])))] # We also keep the instance label as a feature. This is feature 31
    instanceLabel=str(int(float(segment[-1])))
    return bagID,features,instanceLabel

def createData(label, dataDir):
    data=[]
    
    fr_data=open(os.path.join(dataDir,label+'.data'),'r')
    
    firstline=fr_data.readline()
    bagID, features, instanceLabel = splitLine(firstline)
    segmentFeatures=[", ".join(features)]
    bagLabel=instanceLabel
    
    poscount=int(instanceLabel)
    instancecount=1
    allcounts=[]

    for line in fr_data.readlines():
        nextBagID, nextFeatures, nextInstanceLabel = splitLine(line)
        nextFeatures=", ".join(nextFeatures)

        if nextInstanceLabel=='1':
            bagLabel='1'
            poscount+=1; 
        if nextBagID==bagID:
            segmentFeatures.append(nextFeatures)
            instancecount+=1
        if nextBagID!=bagID:
            segments="\\n".join(segmentFeatures)
            data.append(",".join([bagID,'"'+segments+'"',bagLabel]))

            segmentFeatures=[nextFeatures]
            bagLabel='0'
            if poscount != 0:
                allcounts.append(poscount/float(instancecount))
            poscount=0; instancecount=0
        
        bagID=nextBagID
    return '\n'.join(data)    
    

import data

def run(dataDir,arffOutDir):
    """
    Converting SIVAL datasets to ARFF format
    @param dataDir: Directory with the original SIVAL data.
    @param arffOutDir: Directory where to write the arff files to.
    """
    if not os.path.exists(arffOutDir):
        os.makedirs(arffOutDir)
    for label in data.datasets['sival']:
        print label
        dataString = createData(label)
        fw_data=open(os.path.join(arffOutDir,label+'.arff'),'w')
        fw_data.write(dataString)
        fw_data.close()
        
