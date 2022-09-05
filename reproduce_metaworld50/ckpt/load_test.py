checkpoint = "/home/slowlab/Desktop/MetaRL/MetaRL_Implementations/reproduce_metaworld50/ckpt/checkpoint-301"

import numpy as np

a = np.load(checkpoint, allow_pickle=True)
print(a.shape)