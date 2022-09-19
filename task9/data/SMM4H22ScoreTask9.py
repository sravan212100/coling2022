#!/usr/bin/env python
import sys
import logging as log
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

LOG_FILE = '/tmp/eval.log'
log.basicConfig(level=log.DEBUG,
                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                handlers=[log.StreamHandler(sys.stdout), log.FileHandler(LOG_FILE)])
log.info("-------------------------------------------")

# as per the metadata file, input and output directories are the arguments
[_, input_dir, output_dir] = sys.argv
# If you want to run the script directly, comment above line and use the below field to specify input and output direcory
#output_dir = ''
#input_dir=''

log.warning("Start scoring")
print("Start scoring")

# unzipped reference data is always in the 'ref' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions

dtruth = [x for x in os.listdir(os.path.join(input_dir, 'ref')) if x in ["test.tsv", "valid.tsv"]][0]
dtruth = pd.read_csv(os.path.join(input_dir, 'ref', dtruth), sep='\t')
#dtruth = pd.read_csv('ref\valid.tsv', sep='\t')
assert 'id' in dtruth.columns, "I was expecting the column id to be in the tsv file"
assert 'label' in dtruth.columns, "I was expecting the column label to be in the tsv file"

dtruth.set_index('id', inplace=True)

# unzipped submission data is always in the 'res' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
submission_dir = os.path.join(input_dir, 'res')
dpred = [x for x in os.listdir(submission_dir) if x[-3:] in ["tsv"]][0]
submission_path = os.path.join(submission_dir, dpred)

# If you want to run the script directly, comment above linew and use the below field to specify input and output direcory
#output_dir = 'C:\\Users\\yourname\\prediction-file'

if not os.path.exists(submission_path):
    log.fatal('Expected submission named predictions.tsv, found files: ' + submission_path)
    raise Exception('Expected submission named predictions.tsv, found files: ' + submission_path)

dpred = pd.read_csv(submission_path, sep='\t')
assert 'id' in dpred.columns, "I was expecting the column id to be in the tsv file, it was not found."
assert 'label' in dpred.columns, "I was expecting the column label containing the predictions of the classifier to evaluate to be in the tsv file."

dpred.set_index('id', inplace=True)
dpred.rename(columns={"label": "pred"}, inplace=True)

assert len(dtruth) == len(dpred), "The number of posts predicted " + str(len(dpred)) + " is not equal to the number of posts annotated in the test set " + str(len(dtruth))
assert dtruth.sort_index().index.equals(dpred.sort_index().index), "The post IDs in the test set do not correspond to the post IDs in the set of posts predicted"

dEval = pd.concat([dtruth, dpred], axis=1, join='inner')
dEval.to_csv('/tmp/out.tsv', sep='\t')

#cf = confusion_matrix(list(dEval['label']), list(dEval['Class']))
#cr = classification_report(list(dEval['label']), list(dEval['Class']),digits=4)
prec = precision_score(list(dEval['label']), list(dEval['pred']), pos_label=1, average='binary')
rec = recall_score(list(dEval['label']), list(dEval['pred']), pos_label=1, average='binary')
f1 = f1_score(list(dEval['label']), list(dEval['pred']), pos_label=1, average='binary')

#print(f"cf matrix: {cf}\ncr :{cr}\nPrec:{prec}, Rec:{rec}, F1:{f1}")

log.warning("scores computed")
print("scores computed")

# the scores for the leaderboard must be in a file named "scores.txt"
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
    output_file.write("Task9F: " + str(f1)+"\n")
    output_file.write("Task9P: " + str(prec)+"\n")
    output_file.write("Task9R: " + str(rec)+"\n")
    output_file.flush()

log.warning("output file written")
print("output file written")