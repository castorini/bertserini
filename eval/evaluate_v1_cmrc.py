# -*- coding: utf-8 -*-
'''
Evaluation script for CMRC 2018
version: v4
Note: fixed segmentation issues
'''
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import nltk
import pdb

#from utils import init_logger
#logger = init_logger("evaluation")

# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
	in_str = str(in_str).lower().strip()
	segs_out = []
	temp_str = ""
	sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
			   '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
			   '「','」','（','）','－','～','『','』']
	for char in in_str:
		if rm_punc and char in sp_char:
			continue
		if re.search(u'[\u4e00-\u9fa5]', char) or char in sp_char:
			if temp_str != "":
				ss = nltk.word_tokenize(temp_str)
				segs_out.extend(ss)
				temp_str = ""
			segs_out.append(char)
		else:
			temp_str += char

	#handling last part
	if temp_str != "":
		ss = nltk.word_tokenize(temp_str)
		segs_out.extend(ss)

	return segs_out


# remove punctuation
def remove_punctuation(in_str):
	in_str = str(in_str).lower().strip()
	sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
			   '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
			   '「','」','（','）','－','～','『','』']
	out_segs = []
	for char in in_str:
		if char in sp_char:
			continue
		else:
			out_segs.append(char)
	return ''.join(out_segs)


# find longest common string
def find_lcs(s1, s2):
	m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
	mmax = 0
	p = 0
	for i in range(len(s1)):
		for j in range(len(s2)):
			if s1[i] == s2[j]:
				m[i+1][j+1] = m[i][j]+1
				if m[i+1][j+1] > mmax:
					mmax=m[i+1][j+1]
					p=i+1
	return s1[p-mmax:p], mmax

#
def evaluate(ground_truth_file_name, prediction_file_name):
    ground_truth_file = json.load(open(ground_truth_file_name,'r'))
    prediction_file = json.load(open(prediction_file_name,'r'))
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    for instance in ground_truth_file:
        context_id   = instance['context_id'].strip()
        context_text = instance['context_text'].strip()
        for qas in instance['qas']:
            total_count += 1
            query_id    = qas['query_id'].strip()
            query_text  = qas['query_text'].strip()
            answers 	= qas['answers']

            if query_id not in prediction_file:
                sys.stderr.write('Unanswered question: {}\n'.format(query_id))
                skip_count += 1
                continue
            prediction 	= prediction_file[query_id]
            f1_now = calc_f1_score(answers, prediction)
            em_now = calc_em_score(answers, prediction)
            f1 += f1_now
            em += em_now
            # print("Q:{}\tG:{}\tP:{}".format(query_text, answers, prediction))
            #     print("Q:{}\tG:{}\tP:{}\tf1:{}".format(query_text, answers, prediction, f1_now))

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score, total_count, skip_count

def evaluate_general(ground_truth_file, prediction_file):
	f1 = 0
	em = 0
	total_count = 0
	skip_count = 0
	# for instance in ground_truth_file:
		# context_id   = instance['context_id'].strip()
		# context_text = instance['context_text'].strip()
	for qas in ground_truth_file:
		total_count += 1
		query_id    = str(qas['id'])
		query_text  = qas['questions']
		answers 	= qas['answers']

		if query_id not in prediction_file:
			sys.stderr.write('Unanswered question: {}\n'.format(query_id))
			skip_count += 1
			continue

		prediction 	= [prediction_file[query_id]]
		#logger.info("Q:{} G:{} P:{}".format(query_text, answers, prediction))
		f1 += calc_f1_score(answers, prediction)
		em += calc_em_score(answers, prediction)

	f1_score = 100.0 * f1 / total_count
	em_score = 100.0 * em / total_count
	return f1_score, em_score, total_count, skip_count


def calc_f1_score(answers, prediction):
	f1_scores = []
	for ans in answers:
		ans_segs = mixed_segmentation(ans, rm_punc=True)
		prediction_segs = mixed_segmentation(prediction, rm_punc=True)
		lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
		if lcs_len == 0:
			f1_scores.append(0)
			continue
		precision 	= 1.0*lcs_len/len(prediction_segs)
		recall 		= 1.0*lcs_len/len(ans_segs)
		f1 			= (2*precision*recall)/(precision+recall)
		f1_scores.append(f1)
	return max(f1_scores)


def calc_em_score(answers, prediction):
	em = 0
	for ans in answers:
		ans_ = remove_punctuation(ans)
		prediction_ = remove_punctuation(prediction)
		if ans_ == prediction_:
			em = 1
			break
		#else:
		#	print("{} {}".format(ans_, prediction_)) #logger.info("{} {}".format(ans_, prediction_))
	return em

if __name__ == '__main__':
	ground_truth_file   = json.load(open(sys.argv[1], 'rb'))
	prediction_file     = json.load(open(sys.argv[2], 'rb'))
	F1, EM, TOTAL, SKIP = evaluate(ground_truth_file, prediction_file)
	AVG = (EM+F1)*0.5
	#logger.info('AVG: {:.3f} F1: {:.3f} EM: {:.3f} TOTAL: {} SKIP: {} FILE: {}'.format(AVG, F1, EM, TOTAL, SKIP, sys.argv[2]))
