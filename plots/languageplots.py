"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib.pyplot as plt

base = {}
baseStd = {}

acc = {}
acc['embed'] = {}
acc['lstmo'] = {}

std = {}
std['embed'] = {}
std['lstmo'] = {}

# LANGS: FR-IT, FR-DE, FR-EN

base['gen'] = 0.5030 
base['num'] = 0.6968  
base['Per'] = 0.6141 
base['Ten'] = 0.7629 
base['Moo'] = 0.2450 
base['genITM'] = base['gen']  
base['avgAllFeats'] = np.mean([base['gen'],base['num'],base['Per'],base['Ten'],base['Moo']])
base['genNumTen'] = np.mean([base['gen'],base['num'],base['Ten']])

baseStd['gen'] = 0.0043
baseStd['num'] = 0.0073
baseStd['Per'] = 0.0392
baseStd['Ten'] = 0.0238
baseStd['Moo'] = 0.0504
baseStd['genITM'] = baseStd['gen']  
#baseStd['avgAllFeats'] = np.mean([baseStd['gen'],baseStd['num'],baseStd['Per'],baseStd['Ten'],baseStd['Moo']])
baseStd['avgAllFeats'] = 0  ## HACK! 
baseStd['genNumTen'] = 0  ## HACK! 

#gender
acc['embed']['gen'] = (0.5804, 0.5304, 0.5085)
std['embed']['gen'] = (0.0272, 0.0321, 0.0357)
#gender with itNoMasc (2nd lang)
acc['embed']['genITM'] = (0.5804, 0.5196, 0.5304, 0.5085)
std['embed']['genITM'] = (0.0272, 0.0226, 0.0321, 0.0357)
# number
acc['embed']['num'] = (0.6804, 0.6623, 0.6563)
std['embed']['num'] = (0.0131, 0.0106, 0.0184)
# person
acc['embed']['Per'] = (0.5648, 0.5789, 0.6017)
std['embed']['Per'] = (0.0984, 0.0493, 0.0405)
# tense
acc['embed']['Ten'] = (0.7219, 0.7090, 0.7483)
std['embed']['Ten'] = (0.0051, 0.0466, 0.0073)
# mood
acc['embed']['Moo'] = (0.4752,0.4515, 0.4908)
std['embed']['Moo'] = (0.0370, 0.0640, 0.0250)
#
# all features averaged
layer = 'embed'
acc_array = []
for L in range(3):
    acc_array.append(np.mean([acc[layer]['gen'][L],acc[layer]['num'][L],acc[layer]['Per'][L],acc[layer]['Ten'][L],acc[layer]['Moo'][L]]))
acc[layer]['avgAllFeats'] = acc_array
print(acc[layer]['avgAllFeats'])
acc_array = []
for L in range(3):
    acc_array.append(np.mean([acc[layer]['gen'][L],acc[layer]['num'][L],acc[layer]['Ten'][L]]))
acc[layer]['genNumTen'] = acc_array
print(acc[layer]['genNumTen'])
# std_array = []
# for L in range(3):
#     std_array.append(np.mean([std[layer]['gen'][L],std[layer]['num'][L],std[layer]['Per'][L],std[layer]['Ten'][L],std[layer]['Moo'][L]]))
# std[layer]['avgAllFeats'] = std_array
#print(std[layer]['avgAllFeats'])
std[layer]['avgAllFeats'] = (0,0,0)  # HACK!
std[layer]['genNumTen'] = (0,0,0)  # HACK!

#gender
acc['lstmo']['gen'] = (0.8045,0.6505,0.5949)
std['lstmo']['gen'] = (0.0094,0.0228,0.0106)
#gender with itNoMasc (2nd lang)
acc['lstmo']['genITM'] = (0.8045,0.6191,0.6505,0.5949)
std['lstmo']['genITM'] = (0.0094,0.0175,0.0228,0.0106)
#number
acc['lstmo']['num'] = (0.9413, 0.9463, 0.9278)
std['lstmo']['num'] = (0.0016,0.0036, 0.0050)
#person
acc['lstmo']['Per'] = (0.6777, 0.6727, 0.6888)
std['lstmo']['Per'] = (0.0329, 0.0297, 0.0220)
# tense
acc['lstmo']['Ten'] = (0.9019, 0.8880, 0.8897)
std['lstmo']['Ten'] = (0.0080, 0.0086, 0.0169)
#mood
acc['lstmo']['Moo'] = (0.8182, 0.8070, 0.8041)
std['lstmo']['Moo'] = (0.0067, 0.0126, 0.0240)
#
# all features averaged
layer = 'lstmo'
acc_array = []
for L in range(3):
    acc_array.append(np.mean([acc[layer]['gen'][L],acc[layer]['num'][L],acc[layer]['Per'][L],acc[layer]['Ten'][L],acc[layer]['Moo'][L]]))
acc[layer]['avgAllFeats'] = acc_array
print(acc[layer]['avgAllFeats'])
acc_array = []
for L in range(3):
    acc_array.append(np.mean([acc[layer]['gen'][L],acc[layer]['num'][L],acc[layer]['Ten'][L]]))
acc[layer]['genNumTen'] = acc_array
print(acc[layer]['genNumTen'])
# std_array = []
# for L in range(3):
#     std_array.append(np.mean([std[layer]['gen'][L],std[layer]['num'][L],std[layer]['Per'][L],std[layer]['Ten'][L],std[layer]['Moo'][L]]))
# std[layer]['avgAllFeats'] = std_array
#print(std[layer]['avgAllFeats'])
std[layer]['avgAllFeats'] = (0,0,0)   # HACK!
std[layer]['genNumTen'] = (0,0,0)  # HACK!

#############
#############

feats = ['gen','num','Per','Ten','Moo','avgAllFeats','genITM','genNumTen']
featNames = ['Gender','Number','Person','Tense','Mood','All 5 Features','Gender', 'All Features']
#for i in range(6):
for i in range(7,8):    # only genNumTen
    feat = feats[i]
    featName = featNames[i]
    
    
    N = 3   # for: baseline, embedding, lstm-state
    if feat == 'genITM':
        N = 4

        
    ind = np.arange(N)  # the x locations for the groups
    width = 0.3       # the width of the bars

    fig, ax = plt.subplots()

    colors1 = ('#9999FF')
    colors2 = ('#0000FF')
    #if feat == 'genITM':        # use diff color for Fr-ITnoMasc
    #    colors1 = ('#9999FF','#85e085','#9999FF','#9999FF')
    #    colors2 = ('#0000FF','#248F24','#0000FF','#0000FF')


    rects0 = ax.bar(0.5*width, base[feat], width, color='#FF9900', yerr=baseStd[feat])
    rects1 = ax.bar(2.5*width+ind+0.5*width,         acc['embed'][feat], width, color=colors1, yerr=std['embed'][feat])
    rects2 = ax.bar(2.5*width+ind+0.5*width + width, acc['lstmo'][feat], width, color=colors2, yerr=std['lstmo'][feat])


    # add some text for labels, title and axes ticks
    ax.set_ylabel('Prediction Accuracy',size=12)
    ax.set_title(featName + ' prediction',size=16)
    xticks = (np.arange(N+1) + 0.05)
    xticks[0] = width/2
    #ax.set_xticks(width/2, np.arange(N) + width / 2)
    ax.set_xticks(xticks)    
    ax.set_xticklabels(('majClass', 'FR-IT', 'FR-DE', 'FR-EN'))
    if feat == 'genITM':
        ax.set_xticklabels(('majClass', 'FR-IT', 'FR-IT*', 'FR-DE', 'FR-EN'))
    ax.set_ylim(0.2,1)

    ax.legend((rects1[0], rects2[0]), ('Word embedding', 'LSTM state'))


    filename = feat + '_byLang.pdf'
    plt.savefig(filename, bbox_inches='tight')
    #plt.show()
