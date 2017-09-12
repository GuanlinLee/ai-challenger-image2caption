import json
pathofvoc = 'G:/' + 'code/'

fullpathofvoc = pathofvoc + 'voca' + '.json'
fpvoc = open(fullpathofvoc, 'r',encoding='utf-8')
vocabulary = json.load(fpvoc)
path = 'G:/' + 'ai_challenger_caption_train_20170902/'

fullpath = path + 'caption_train_annotations_20170902' + '.json'
fp = open(fullpath, 'r')
images = json.load(fp)
listoftext=[]
for i in range(1):
    for labelnum in range(5):
        for word in images[i]['caption'][labelnum]:
            listoftext.append(word)
print(listoftext)
id=[]
for word in listoftext:
    for i in vocabulary:
        num=i['id']
        if i['word']==word:
            id.append(num)

print(id)