'''
Run SLDA.

@author: Yohan
@version: July 24, 2017
'''
import subprocess

jar_path = "/Users/Yohan/Dropbox/Research/workspace/YUtils/jar/SLDA.jar"
n_topics = 10  # Number of topics
n_iters = 1000  # Number of iterations
data_dir = "/Users/Yohan/Dropbox/Research/metaphor/data/SLDA_10000_new"
train_filename = "data.csv"  # Training data
test_filename = "data2.csv"  # Testing data
model_path = "/Users/Yohan/Dropbox/Research/metaphor/data/SLDA_10000_new/" +\
             "SLDA-data-T10-A0.1-B0.001-G1.0/SLDA-data-T10-A0.1-B0.001-G1.0-I1000"\
              # Prefix of the path of output files from training
              

# Train SLDA on large (unlabeled) data
subprocess.call(map(str, 
                    ["java", "-jar", jar_path, 
                     "-t", n_topics, "-a", 0.1, 
                     "-b", 0.001, "-g", 1, "-d", data_dir, 
                     "-data", train_filename, "-tok", "-i", n_iters, 
                     "-log", 100]))

# Infer topics for labeled data
subprocess.call(map(str,
                    ["java", "-jar", jar_path,
                     "-t", n_topics, "-a", 0.1, 
                     "-b", 0.001, "-g", 1, "-d", data_dir,
                     "-data", test_filename, "-tok", "-i", n_iters, 
                     "-log", 100, "-model", model_path]))
