import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='parse outputfile from mltagger')
parser.add_argument('path', metavar='path', type=str, nargs='+',
                    help='path for outfile from experiment')
parser.add_argument('split', metavar='split', type=str, nargs='+',
                    help='dev or test or both')
parser.add_argument('task', metavar='task', type=str, nargs='+',
                    help='Primary task (e.g. fce, semeval_neg_tok, semeval_pos_sent')
parser.add_argument('seed', metavar='seed', type=str, nargs='+',
                    help='seed')

args = parser.parse_args()

resfile = args.path[0]
task = args.task[0]
split=args.split[0]
seed=args.seed[0]

metrics = ['sent_p', 'sent_r' ,'sent_f']

content = open(resfile).read()
endidx = 0
realendidx = 0
endofdoc = False

while not endofdoc:
    newstartidx = content[realendidx:].find('=== TASK: {} ==='.format(task))
    newstartidx += realendidx
    endidx = content[newstartidx+50:].find('==='.format(task)) #moving 50 characters ahead to not find the same header again
    if endidx == -1:
        endidx = len(content)
    realendidx = newstartidx
    realendidx += endidx
    realendidx += 50

    res = content[newstartidx:realendidx] #takes the last occurence of split
    if split in res:
        #parse results
        reslist = res.split('\n')[1:] #don't take header
        reslist = [result.split(':') for result  in reslist]
        reslist = [[li.lstrip(task).strip() for li in result] for result in reslist] #remove task - same for all
        reslist = [res for res in reslist if len(res) == 2]
        reslist = [[li.replace(split+"_","",1) for li in result] for result in reslist] #remove split
        resdict = dict(reslist)
        resdict = dict((k,float(v)) for k,v in resdict.items()) # convert to floats
        continue #important - otherwise it does not evaluate next if statement
    if endidx == len(content):
        endofdoc=True
        
for metric in metrics:
    print(resfile, '\t', seed, '\t',  metric, '\t', resdict[metric])

