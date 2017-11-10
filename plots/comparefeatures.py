import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

# LANGS: FR-IT, FR-DE, FR-EN
genBase = 0.5030    
genEmbs = np.mean([0.5804,0.5304,0.5085])
genLstm = np.mean([0.8045,0.6505,0.5949])

numBase = 0.6968 
numEmbs = np.mean([0.6804,0.6623,0.6563])
numLstm = np.mean([0.9413,0.9463,0.9278])

PerBase = 0.6141
PerEmbs = np.mean([0.5648,0.5789,0.6017])
PerLstm = np.mean([0.6777,0.6727,0.6888])

TenBase = 0.7629
TenEmbs = np.mean([0.7219,0.7090,0.7483])
TenLstm = np.mean([0.9019,0.8880,0.8897])

MooBase = 0.2450
MooEmbs = np.mean([0.4752,0.4515,0.4908])
MooLstm = np.mean([0.8182,0.8070,0.8041])

genBaseStd = 0.0043
numBaseStd = 0.0073
PerBaseStd = 0.0392
TenBaseStd = 0.0238
MooBaseStd = 0.0504

genEmbsstd = np.sqrt(0.0272**2 + 0.0321**2 + 0.0357**2)
numEmbsstd = np.sqrt(0.0131**2 + 0.0106**2 + 0.0184**2)
perEmbsstd = np.sqrt(0.0984**2 +  0.0493**2 + 0.0405**2)
tenEmbsstd = np.sqrt(0.0051**2 +  0.0466**2 +  0.0073**2)
mooEmbsstd = np.sqrt(0.0370**2 + 0.0640**2 + 0.0250**2)

genLstmstd =np.sqrt(0.0094**2 + 0.0228**2 + 0.0106**2)
numLstmstd = np.sqrt(0.0016**2 + 0.0036**2 + 0.0050**2)
perLstmstd = np.sqrt(0.0329**2 +  0.0297**2 +  0.0220**2)
tenLstmstd = np.sqrt(0.0080**2 +  0.0086**2 + 0.0169**2)
mooLstmstd = np.sqrt(0.0067 **2 + 0.0126 **2 + 0.0240**2)

baseAcc = [genBase, numBase, PerBase, TenBase, MooBase]
embdAcc = [genEmbs, numEmbs, PerEmbs, TenEmbs, MooEmbs]
lstmAcc = [genLstm, numLstm, PerLstm, TenLstm, MooLstm]
yerrs = [genEmbsstd, genLstmstd, numEmbsstd, numLstmstd, perEmbsstd, perLstmstd, tenEmbsstd, tenLstmstd, mooEmbsstd, mooLstmstd]

colors = ['darkred','red', 'navy', 'blue','lightskyblue']
markers = ['X','o','s', '^', 'D']
featNames = ['gender', 'number', 'person', 'tense','mood']

plot_set = 'all_langs'
#plot_set = 'by_lang'

if plot_set == 'all_langs':
    feat_set = 'nominal_and_tense'
    myrange = []
    if feat_set == 'all':
        myrange = range(5)
    elif feat_set == 'nominal':
        myrange = range(0,2)
    elif feat_set == 'verbal':
        myrange = range(2,5)
    elif feat_set == 'nominal_and_tense':
        myrange = (0,1,3)
        

    selectfeatNames = []
    for i in myrange:
        x = np.array([1,2,3])
        y = np.array([baseAcc[i],embdAcc[i],lstmAcc[i]])	
        #yerrsnow = np.array([yerrs[i*2],yerrs[i*2+1]])
        #plt.errorbar(x, y, color=extras[i], linestyle='--', marker='o', linewidth=3, markersize=15, yerr=yerrsnow)
        plt.errorbar(x, y, color=colors[i], linestyle='--', marker=markers[i], linewidth=2, markersize=12)
        selectfeatNames.append(featNames[i])


    #my_xticks = ['','Word embed.', '', '', 'LSTM state','']
    #xrangei = [0,1,2,3,4,5]
    my_xticks = ['Majority class', 'Word embedding', 'LSTM state']
    xrangei = [1,2,3]
    plt.xticks(xrangei, my_xticks,size=15)
    plt.yticks(size=12)
    plt.ylim((0.2,1.0))
    plt.legend(selectfeatNames, loc='lower right',prop={'size': 12})
    plt.ylabel('Mean accuracy',size=12)
    #plt.title('Morphological feature prediction',size=15)
    #plt.show()

    filename = 'lang_avg_' + feat_set + '.pdf'
    plt.savefig(filename, bbox_inches='tight')

### 
# elif plot_set == 'by_lang':
#     feature = 'gender'
#     if feature == 'gender':
#         baseAcc = genBase
#     
#     selectLangNames = []
#     for i in range(3):
#        x = np.array([1,2,3])
#        y = np.array([baseAcc[i],embdAcc[i],lstmAcc[i]])	
         
    