"""
Below code is for the Part of speech tagging using Hidden Markov Model. The code contains 3 classes:
1. FileRead class: it is to read the content of file from given file location.
2. HmmLearn class: It is to train the hmm model and create the transition and emission probability matrices.
3. HmmDecode class: It is to decode the unknown file and predict the part of speech for each word.

To improve the accuracy of the model, I have modified the transition and emission probabilities with "Add one smoothing" which improved the 
accuracy of the model from 82% to 88%.
The code is generalized to perform well on different languages. Its been tested on English, Chinese and Hindi and got high accuracy
"""

"""importing all the dependencies"""

import time
import math
import json
import sys

"""below class is to read the file content from a given file location"""
class FileRead(object):

    """A function to read the file and append start word ("1bi10") with START tag and end word ("it006") with END tag to every sentence"""
    def readFile(self, fileName, flag):
        #with open(sys.argv[1]) as fTrain:
        with open(fileName, encoding="utf8") as f:
            content = f.readlines()
        if flag ==0:
            content = ["1bi10/START "+x.strip()+ " it006/END" for x in content]
        else:
            content = [x.strip() for x in content]
        return content

"""below class is to train the hmm model and create the transition and emission probability matrices"""
class HmmLearn(object):

    """A function that takes the file content as input and returns a list with all the distinct tags and one dictionary with tags as the key 
    and their count as the value."""
    def getTags(self, content):
        tags = []
        tagsDict = {}
        for line1 in content:
            line = line1.split(" ")
            for word in line:
                wordTemp,tagsTemp = word.rsplit("/",1)
                tags.append(tagsTemp)
                if tagsTemp in tagsDict:
                    tagsDict[tagsTemp] +=1
                else:
                    tagsDict[tagsTemp] =1
        tags= list(set(tags))
        return tags,tagsDict

    """A function that takes tags and content as the input and returns a nested dictionary dictTransition (Transition matrix) 
    with key as the previous tag and value as another dictionary with key as current tag and value as the count of previous 
    to current transition in the input content"""
    def createTransitionMatrix(self, tags, content):

        dictTransition = {}
        for i in range(len(tags)):
            if tags[i] !='END':
                for j in range(len(tags)):
                    if tags[i] =="START" and tags[j] =="END":
                        continue
                    if tags[j]!= 'START':
                        if tags[i] in dictTransition:
                            dictTransition[tags[i]][tags[j]] = 1
                        else:
                            dictTransition[tags[i]]= {tags[j]:1}

        for lines in range(len(content)):
            line = content[lines].split(" ")
            for word in range(1,len(line)):
                wordSplit = line[word].rsplit("/",1)
                wordSplitPrev = line[word-1].rsplit("/",1)
                dictTransition[wordSplitPrev[1]][wordSplit[1]] += 1
        return dictTransition

    """A function that takes transition matrix and content as the input and returns two dictionaries with the total count of transition and emission.
    This will be useful in calculating the transition and emission probability"""
    def totalTransitionAndTotalEmission(self, dictTransition, content):
        dictTotalTransition = {}
        for i,val in dictTransition.items():
            countTransition = 0
            for j,count in val.items():
                if j!= 'END':
                    countTransition += count
            dictTotalTransition[i] = countTransition
            
        dictTotalEmission= {}
        for i,val in dictTransition.items():
            countTransition = 0
            for j,count in val.items():
                countTransition += count
            if i in dictTotalEmission:
                dictTotalEmission[i] += countTransition
            else:
                dictTotalEmission[i] = countTransition

        return dictTotalTransition, dictTotalEmission

    
    """A function that converts the count transition matrix into proabability transition martix. here we take log probability to overcome 
    any float overflow error and make the computation faster"""
    def calculateTransitionProbablity(self, dictTransition, dictTotalTransition):
        for i,val in dictTransition.items():
            for j,count in val.items():
                #if j!='END':
                dictTransition[i][j] = math.log(count/dictTotalTransition[i])
        return dictTransition
    
    """A function that takes content as the input and returns a nested dictionary dictEmission (Emission matrix) 
    with key as the tag and value as another dictionary with key as word and value as the count of tag word emission
     in the input content"""
    def creatEmissionMatrix(self, content):
        dictEmission = {}
        for lines in range(len(content)):
            line = content[lines].split(" ")
            for word in range(0,len(line)):
                wordSplit = line[word].rsplit("/",1)
                if wordSplit[0] in dictEmission:
                    if wordSplit[1] in dictEmission[wordSplit[0]]:
                        dictEmission[wordSplit[0]][wordSplit[1]] +=1
                    else:
                        dictEmission[wordSplit[0]][wordSplit[1]] =1
                else:
                    dictEmission[wordSplit[0]] ={wordSplit[1]:1}
        return dictEmission
   
    """A function that converts the count emission matrix into proabability emission martix. here we take log probability to overcome 
    any float overflow error and make the computation faster"""
    def calculateEmissionProbablity(self, dictEmission):
        for i,val in dictEmission.items():
            for j, count in val.items():
                if j!= "END":
                    dictEmission[i][j] = math.log(count/dictTotalEmission[j])
        return dictEmission

"""Below class is to decode the unknown file and predict the part of speech for each word"""
class HmmDecode(object):

    """A function to contruct the viterbi matrix that would be used to select the best path in the markov chain"""
    def createViterbiMatrix(self, statement,tags):
        
        viterbiMatrix = []
        
        """initilize the viterbi matrix with "-inf" value, its a 2-D matrix with shape as len(tags)*len(statement)"""
        for i in range(len(tags)):
            temp = []
            for j in range(len(statement)):
                temp.append((float("-inf"),0))
            viterbiMatrix.append(temp)
    
        """update the start tag value, i.e. the first column in the viterbi matrix"""
        for i in range(len(tags)):
            if tags[i]!='END' and tags[i]!= 'START':
                if statement[0] not in dictEmission:
                    for tagVal in range(len(tags)):
                        if tagVal ==0:
                            dictEmission[statement[i]] = {tags[tagVal]:0}
                        else:
                            dictEmission[statement[i]][tags[tagVal]] = 0
                if tags[i] in dictEmission[statement[0]]:
                    viterbiMatrix[i][0] =(dictEmission[statement[0]][tags[i]] + dictTransition["START"][tags[i]], "START")
        
        """update all the values in the viterbi matrix as per the viterbi formula"""
        for i in range(1,len(viterbiMatrix[0])):
            for j in range(len(tags)):
                if tags[j]!= "END" and tags[j] !="START":
                    maxVal = (float("-inf"),"START")
                    if statement[i] not in dictEmission:
                        for tagVal in range(len(tags)):
                            if tagVal ==0:
                                dictEmission[statement[i]] = {tags[tagVal]:0}
                            else:
                                dictEmission[statement[i]][tags[tagVal]] = 0
                    for k in range(len(tags)):
                        if tags[k] !="END"and tags[j] !="START":
                            temp = (float("-inf"),0)
                            if viterbiMatrix[k][i-1][0]>=float("-inf") and tags[j] in dictEmission[statement[i]]:
                                temp = (dictEmission[statement[i]][tags[j]] + dictTransition[tags[k]][tags[j]]+viterbiMatrix[k][i-1][0], tags[k])
                            if   maxVal[0] < temp[0]:
                                maxVal = temp
                    viterbiMatrix[j][i] = maxVal
        return viterbiMatrix

    """A function to back track the viterbi matrix and find the best tag sequence. It takes viterbi matrix as input and returns a list of tag sequence"""
    def backTracking(Self, viterbiMatrix,tags,ln):
        maxVal = (float("-inf"),0)
        TagSeq = []
        for i in range(len(tags)):
            if viterbiMatrix[i][ln] !=0:
                if maxVal[0] < viterbiMatrix[i][ln][0]:
                    maxVal = viterbiMatrix[i][ln]
                    index = i
        TagSeq.append(tags[index])
        TagSeq.append(maxVal[1])
        for i in range(len(viterbiMatrix[0])-2, -1,-1):
            tagIndex = tags.index(TagSeq[-1])
            TagSeq.append(viterbiMatrix[tagIndex][i][1])
        TagSeq = TagSeq[::-1]

        return TagSeq
    

    """write the decoded part of speech into the output file"""
    def writeFile(self, file, TagSeq, statement):
        for i in range(len(statement)):
            file.write(statement[i])
            file.write('/')
            file.write(TagSeq[i+1])
            file.write(" ")
        file.write('\n')

        
if __name__ == '__main__':
    file = FileRead()
    
    """First command line argument would be training file location and name"""
    fileName = sys.argv[1]
    
    """Second command line argument would be test file location and name"""
    fileNameTest = sys.argv[2]
    
    """Third command line argument would be output file location and name"""
    outFileName = sys.argv[3]
    
    """read the content of the training file"""
    contentTrain = file.readFile(fileName,0)
    hmmLearn = HmmLearn()
    
    """generate the tag and tagDict (they are defined in the definition of the getTags function)"""
    tags,tagsDict = hmmLearn.getTags(contentTrain)
    
    """generate the transition dictionary"""
    dictTransition = hmmLearn.createTransitionMatrix(tags,contentTrain)
    
    """generate the count for calculating transition and emission probabilities"""
    dictTotalTransition, dictTotalEmission = hmmLearn.totalTransitionAndTotalEmission(dictTransition, contentTrain)
    
    """update the dictTransition with the log probability"""
    dictTransition = hmmLearn.calculateTransitionProbablity(dictTransition, dictTotalTransition)
    
    """generate the emission dictionary"""
    dictEmission = hmmLearn.creatEmissionMatrix(contentTrain)
    
    """update the dictEmission with the log probability"""
    dictEmission = hmmLearn.calculateEmissionProbablity(dictEmission)
    print("**************** Training is completed ****************")

    """read the content of the testing file"""
    contentTest = file.readFile(fileNameTest,1)
    hmmDecode = HmmDecode()
    
    """creating/opening a hmmoutput file to write the output"""
    print("**************** Decoding start ****************")
    print("**************** Writing the output to", outFileName, "file ****************")
    with open(outFileName,'w', encoding="utf8") as fileWrite:
        for line in contentTest:
            statement = line.split(" ")
            
            """generate the viterbi matrix"""
            viterbiMatrix = hmmDecode.createViterbiMatrix(statement,tags)
            ln = len(statement)-1
            
            """generate the tag sequence"""
            TagSeq = hmmDecode.backTracking(viterbiMatrix,tags,ln)
            
            """write the desired output into the file"""
            hmmDecode.writeFile(fileWrite, TagSeq, statement)
    fileWrite.close()
    print("**************** Decoding complete ****************")