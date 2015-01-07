'''
Datasets and algorithms
'''

labels_sival=['ajaxorange','apple','banana','bluescrunge','dirtyworkgloves','juliespot','checkeredscarf','wd40can','candlewithholder','glazedwoodpot','cokecan','smileyfacedoll','dataminingbook','rapbook','feltflowerrug','translucentbowl','greenteabox','cardboardbox','dirtyrunningshoe','largespoon','goldmedal','spritecan','stripednotebook','fabricsoftenerbox','woodrollingpin']
labels_text=['alt_atheism','comp_graphics','comp_os_ms-windows_misc','comp_sys_ibm_pc_hardware','comp_sys_mac_hardware','comp_windows_x','misc_forsale','rec_autos','rec_motorcycles',\
            'rec_sport_baseball','rec_sport_hockey','sci_crypt','sci_electronics','sci_med','sci_space','soc_religion_christian','talk_politics_guns','talk_politics_mideast','talk_politics_misc','talk_religion_misc']
labels_uci=['adult_12', 'adult_13', 'adult_23', 'adult_14', 'adult_24', 'adult_15', 'adult_25', 'adult_110', 'adult_210','diabetes_12', 'diabetes_13', 'diabetes_23','spam_12', 'spam_13', 'spam_23', 'spam_14', 'spam_24', 'spam_15', 'spam_25', 'spam_110', 'spam_210','tictactoe_12', 'tictactoe_13', 'tictactoe_23', 'transfusion_12', 'transfusion_13', 'transfusion_23']


datasets={'uci':labels_uci,'text':labels_text,'sival':labels_sival}

milearners=['MILR','AdaBoostM1','MILRC','MIRI','MIDD','MDD','MIOptimalBall','TLD','MIWrapper','CitationKNN','MISMO','SimpleMI','MIEMDD','MISVM']
silearners=['AdaBoostM1','RBFNetwork', 'J48','SMO','IBk','Logistic']
learnerTuples=[('SimpleMI','J48'),('AdaBoostM1','AdaBoostM1'),('MIWrapper','J48'),('CitationKNN','IBk'),('MILR','Logistic'),('MISMO','SMO')]

def createFeatureList(label,datasetType):
    """
    Create a list with each item being the description of one single-instance feature (arff format) for a given label and dataset type (SIVAL, text, UCI).
    """
    def createFeature(i): 
        return '\t@attribute f'+str(i)+': numeric\n'
    
    if datasetType=='sival':
        featureList = list(map(createFeature, range(30)))
    if datasetType =='text':
        featureList = list(map(createFeature, range(200)))
    if datasetType =='uci':
        if label.split('_')[0]=='transfusion':
            featureList = list(map(createFeature, range(4)))
        if label.split('_')[0]=='tictactoe':
            featureList = list(map(createFeature, range(9)))
        if label.split('_')[0]=='spam':
            featureList = list(map(createFeature, range(57)))
        if label.split('_')[0]=='adult':
            featureList = list(map(createFeature, range(14)))
        if label.split('_')[0]=='diabetes': # pima indians
            featureList = list(map(createFeature, range(8)))
            
    return featureList


    