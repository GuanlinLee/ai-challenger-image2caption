import json
import itertools
import collections
path = 'G:/' + 'ai_challenger_caption_train_20170902/'
path2='G:/'+'ai_challenger_caption_validation_20170910/'
fullpath = path + 'caption_train_annotations_20170902' + '.json'
fullpath2=path2+'caption_validation_annotations_20170910'+'.json'
fp = open(fullpath, 'r')
images = json.load(fp)
fp2=open(fullpath2,'r')
images2=json.load(fp2)
i=0
listoftext = []
for imginfo in images+images2:

    i=i+1
    for labelnum in range(5):
        for word in imginfo['caption'][labelnum]:
            listoftext.append(word)
        print(imginfo['caption'][labelnum])

print(listoftext)
word_dic=collections.Counter(itertools.chain(listoftext))
voc=[]
numofword=0
for wordofseq in word_dic:
    voc.append({'word':wordofseq[0],'id':numofword})
    numofword=numofword+1
with open("./voca.json",'w',encoding='utf-8') as json_file:
    json.dump(voc,json_file,ensure_ascii=False)
print(i)