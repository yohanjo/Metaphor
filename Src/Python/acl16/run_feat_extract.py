'''
Run feat_extract.py.

@author: Yohan
@version: June 22, 2017
'''
import subprocess

data_dir = "/Users/Yohan/Dropbox/Research/metaphor/data"  
post_path = data_dir + "/breastcancer/sigdial2015_cv_dataset.csv"  # Labeled data path
phi_path = data_dir + "/SLDA_10000_new/SLDA-data-T5-A0.1-B0.001-G0.5/SLDA-data-T5-A0.1-B0.001-G0.5-I100-PhiF.csv"  # SLDA result Phi path
slda_path = data_dir + "/SLDA_10000_new/SLDA-data-T5-A0.1-B0.001-G0.5/SLDA-data-T5-A0.1-B0.001-G0.5-I100-InstAssign.csv"  # SLDA result InstAssign path
out_dir = data_dir + "/SLDA_10000_new/SLDA-data-T5-A0.1-B0.001-G0.5"  # Output feature files directory


# Run feat_extract
subprocess.call(["python", "feat_extract.py", post_path, phi_path, slda_path, out_dir])