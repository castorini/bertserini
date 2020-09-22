# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import nltk
import pdb
from hanziconv import HanziConv

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

def calc_overlap_score(answers, prediction):
	overlap_scores = []
	for ans in answers:
		ans_segs = mixed_segmentation(ans, rm_punc=True)
		prediction_segs = mixed_segmentation(prediction, rm_punc=True)
		lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
		if lcs_len == 0:
			overlap_scores.append(0)
		else:
			overlap_scores.append(1)
	return max(overlap_scores)

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
	return em

def evaluate(dataset, predictions):
	cover = precision = recall = f1 = exact_match = total = overlap = 0
	for article in dataset:
		for paragraph in article['paragraphs']:
			for qa in paragraph['qas']:
				total += 1
				if qa['id'] not in predictions:
					message = 'Unanswered question ' + qa['id'] + \
							  ' will receive score 0.'
					# print(message, file=sys.stderr)
					continue
				ground_truths = list(map(lambda x: HanziConv.toSimplified(x['text']), qa['answers']))
				prediction = HanziConv.toSimplified(predictions[qa['id']])
				# cover += metric_max_recall(cover_score, prediction, ground_truths)
				exact_match += calc_em_score(ground_truths, prediction)
				f1_now = calc_f1_score(ground_truths, prediction)
				# if (f1_now == 1):
				#     print("Q: ", qa['question'], "\tGT: ", ground_truths, "\tP: ", prediction, "\tF1: ", f1_now)
				f1 += f1_now
				overlap += calc_overlap_score(ground_truths, prediction)
				# precision += metric_max_recall(
				# 	precision_score, prediction, ground_truths)
				# recall += metric_max_recall(
				# 	recall_score, prediction, ground_truths)

	# print("evaluation total: ", total)
	exact_match = 100.0 * exact_match / total
	f1 = 100.0 * f1 / total
	# recall = 100.0 * recall / total
	overlap = 100.0 * overlap / total
	# cover = 100.0 * cover / total
	# precision = 100.0 * precision / total

	return {'exact_match': exact_match, 'f1': f1, "overlap": overlap}


def evaluation(dataset_filename, prediction_filename):
	# expected_version = '1.1'
	with open(dataset_filename) as dataset_file:
		dataset_json = json.load(dataset_file)
		# if dataset_json['version'] != expected_version:
		#     print('Evaluation expects v-' + expected_version +
		#           ', but got dataset with v-' + dataset_json['version'],
		#           file=sys.stderr)
		dataset = dataset_json['data']
	with open(prediction_filename) as prediction_file:
		predictions = json.load(prediction_file)
	ans = evaluate(dataset, predictions)
	print(json.dumps(ans))
	return ans


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset_file', type=str,
						help='number of top k paragraphs into bert')
	parser.add_argument('prediction_file', type=str,
						help='Path to index file')
	args = parser.parse_args()
	evaluation(args.dataset_file, args.prediction_file)

