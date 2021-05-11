import os
import time
meta_steps = '400'
gerry_steps = '100'
weight_pairs = [('1', '0'), ('1', '.1'), ('1','.5'), ('1', '.75'), ('1','1')]
for pair in weight_pairs:
    os.system('python3 chain_on_subsets_of_faces.py -mcs ' +  meta_steps+  " -gcs " + gerry_steps + ' -ws ' + pair[0] + ' -wf ' + pair[1]+ ' &')
    time.sleep(60)