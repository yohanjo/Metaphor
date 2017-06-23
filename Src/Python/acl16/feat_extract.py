'''
Generate "TargetTopic", "TopicDiff", "TopicSim", "TopicTrans", and 
"TopicTransSim" features in the ACL16 paper.

@author: Yohan
@version: June 22, 2017
'''
from csv_utils import *
import re, sys
from collections import defaultdict
import numpy as np

post_path = sys.argv[1]  # Labeled data path
phi_path = sys.argv[2]  # SLDA result Phi path
slda_path = sys.argv[3]  # SLDA result InstAssign path
out_dir = sys.argv[4]  # Output feature files directory
feat_target_topic_path = out_dir + "/feat_target_topic.csv"
feat_topic_diff_path = out_dir + "/feat_topic_diff.csv"
feat_topic_sim_path = out_dir + "/feat_topic_sim.csv"
feat_topic_trans_path = out_dir + "/feat_topic_trans.csv"
feat_topic_trans_sim_path = out_dir + "/feat_topic_trans_sim.csv"

keywords = [set(["road","roads"]), set(["candle","candles"]),
            set(["light","lights"]), set(["spice","spices"]), 
            set(["ride","rides","rode"]), set(["train","trains"]), 
            set(["boat","boats"])]


# Topic similarity
phi_list = []
for row in iter_csv_header(phi_path, map_format=False):
    phi_list.append(map(float, row[1:]))
phi = np.array(phi_list)
phi_den = np.outer(np.power(np.power(phi,2).sum(axis=0),0.5), 
                   np.power(np.power(phi,2).sum(axis=0),0.5))
topic_similarity = np.dot(phi.T, phi) / phi_den
n_topics = phi.shape[1]

posts = dict()
instance_ids = []
header = []
for row in iter_csv_header(post_path, header=header):
    posts[row['instanceID']] = row
    instance_ids.append(row['instanceID'])

inst_topics = defaultdict(list)  # Sentential topics for each instance
inst_tgt = dict()  # Target sentence number for each instance
for row in iter_csv_header(slda_path):  # Sentence
    instance_id = row['DocId']
    metaphor_id = int(posts[instance_id]['metaphorID'])
    sentence_id = int(row['Sentence'])  # Sentence number
    text = row['Text']
    topic = int(row['Topic'])
    inst_topics[instance_id].append(topic)
    
    if instance_id not in inst_tgt and \
            len(set(re.split("[^a-zA-Z]+", text.lower())) & \
                keywords[metaphor_id] ) > 0:
        inst_tgt[instance_id] = sentence_id


topic_sim_before = dict()
topic_sim_after = dict()
topic_trans_sim_before = dict()
topic_trans_sim_after = dict()


# TargetTopic
with open(feat_target_topic_path, 'w') as f:
    out_csv = csv.writer(f)
    out_csv.writerow(header + \
                     ["T"+str(t) for t in xrange(n_topics)])
    for instance_id in instance_ids:
        if instance_id in inst_topics:
            tgt_sent = inst_tgt[instance_id]
            tgt_topic = inst_topics[instance_id][tgt_sent]
            feats = [1 if t == tgt_topic else 0 for t in xrange(n_topics)]
        else:
            feats = [0 for t in xrange(n_topics)]
        out_csv.writerow([posts[instance_id][h] for h in header] + feats)
        
        
# TopicSim
with open(feat_topic_sim_path, 'w') as f:
    out_csv = csv.writer(f)
    out_csv.writerow(header + \
                     ["topic_sim_before", "topic_sim_after"])
    for instance_id in instance_ids:
        if instance_id in inst_topics:
            topics = inst_topics[instance_id]
            tgt_sent = inst_tgt[instance_id]
            
            # Right before
            if tgt_sent == 0: before = 0
            else: before = \
                    topic_similarity[topics[tgt_sent], topics[tgt_sent-1]]
            
            # Right after
            if tgt_sent == len(topics)-1: after = 0
            else: after = \
                    topic_similarity[topics[tgt_sent], topics[tgt_sent+1]]

            feats = [before, after]
        else:
            feats = [0,0]
        out_csv.writerow([posts[instance_id][h] for h in header] + feats)
        
        
# TopicDiff
with open(feat_topic_diff_path, 'w') as f:
    out_csv = csv.writer(f)
    out_csv.writerow(header + \
                     ["topic_diff_before", "topic_diff_after"])
    for instance_id in instance_ids:
        if instance_id in inst_topics:
            tgt_sent = inst_tgt[instance_id]
            topics = inst_topics[instance_id]
            tgt_topic = topics[tgt_sent]
            
            # Right before
            if tgt_sent == 0: before = 0
            else: before = 1 if topics[tgt_sent-1] != tgt_topic else 0
            
            # Right after
            if tgt_sent == len(topics)-1: after = 0
            else: after = 1 if topics[tgt_sent+1] != tgt_topic else 0

            feats = [before, after]
        else:
            feats = [0,0]
        out_csv.writerow([posts[instance_id][h] for h in header] + feats)


# TopicTrans
with open(feat_topic_trans_path, 'w') as f:
    out_csv = csv.writer(f)
    out_csv.writerow(header + \
                     ["topic_trans_before"+str(t) for t in xrange(n_topics)] +\
                     ["topic_trans_after"+str(t) for t in xrange(n_topics)])
    for instance_id in instance_ids:
        if instance_id in inst_topics:
            tgt_sent = inst_tgt[instance_id]
            topics = inst_topics[instance_id]
            tgt_topic = topics[tgt_sent]
            
            # Before
            s = None
            for s in range(tgt_sent-1, -1, -1):
                if topics[s] != topics[tgt_sent]: break
            if s == None: before = [0 for t in xrange(n_topics)]
            else: before = \
                    [1 if t == topics[s] else 0 for t in xrange(n_topics)]
            
            # After
            s = None
            for s in range(tgt_sent+1, len(topics)):
                if topics[s] != topics[tgt_sent]: break
            if s == None: after = [0 for t in xrange(n_topics)]
            else: after = \
                    [1 if t == topics[s] else 0 for t in xrange(n_topics)]

            feats = before + after
        else:
            feats = [0 for t in xrange(2 * n_topics)]
        out_csv.writerow([posts[instance_id][h] for h in header] + feats)


# TopicTransSim
with open(feat_topic_trans_sim_path, 'w') as f:
    out_csv = csv.writer(f)
    out_csv.writerow(header + \
                     ["topic_trans_sim_before", "topic_trans_sim_after"])
    for instance_id in instance_ids:
        if instance_id in inst_topics:
            topics = inst_topics[instance_id]
            tgt_sent = inst_tgt[instance_id]
            
            # Before
            s = None
            for s in range(tgt_sent-1, -1, -1):
                if topics[s] != topics[tgt_sent]: break
            if s == None: before = 0
            else: before = topic_similarity[topics[tgt_sent], topics[s]]
            
            # After
            s = None
            for s in range(tgt_sent+1, len(topics)):
                if topics[s] != topics[tgt_sent]: break
            if s == None: after = 0
            else: after = topic_similarity[topics[tgt_sent], topics[s]]

            feats = [before, after]
        else:
            feats = [0,0]
        out_csv.writerow([posts[instance_id][h] for h in header] + feats)

