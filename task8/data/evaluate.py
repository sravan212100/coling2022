#!/usr/bin/env python
import sys
import os
import os.path
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# as per the metadata file, input and output directories are the arguments
[_, input_dir, output_dir] = sys.argv
#input_dir = "/content/"
#output_dir = "/content/"

# unzipped submission data is always in the 'res' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
submission_file_name = 'answer.txt'
submission_dir = os.path.join(input_dir, 'res')
submission_path = os.path.join(submission_dir, submission_file_name)
if not os.path.exists(submission_path):
    message = "Expected submission file '{0}', found files {1}"
    sys.exit(message.format(submission_file_name, os.listdir(submission_dir)))

# unzipped reference data is always in the 'ref' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
y_true = []
y_true_id = []
i = 0
with open(os.path.join(input_dir, 'ref', 'truth.txt')) as truth_file:
    truth = truth_file.readlines()
    for line in truth:
      i+=1
      line = line.strip("\n").split("\t")
      #print(line[1])
      if(i!=1):
        y_true.append(int(line[1]))
        y_true_id.append(line[0])
    #print(y_true_id)

y_pred = []
j = 0
with open(submission_path) as submission_file:
    #submission = submission_file.read()
    submission = submission_file.readlines()
    for id in y_true_id:
      #print("id: ", id)
      tag = 0
      submission_file.seek(0)
      for line in submission:
        j+=1
        line = line.strip("\n").split("\t")
        #print(line[1])
        if(line[0]==id):
          tag = 1
          y_pred.append(int(line[1]))
      if(tag==0):
        y_pred.append(2)
    #print(y_pred)
    #print(len(y_pred))


# the scores for the leaderboard must be in a file named "scores.txt"
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions

with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
    #score = 1 if truth == submission else 0
    #print(truth)
    #print(y_pred)
    f1_micro = f1_score(y_true,y_pred,average='micro')
    print("F1-score: ", f1_micro)

    output_file.write("F1-score:{0}\n".format(f1_micro))
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: ", accuracy)

    output_file.write("Accuracy:{0}\n".format(accuracy))
