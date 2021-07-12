from gen_neg_samples import selecting_neg_poly
import numpy as np 

a = np.random.randint(1,10,(1,5))
b = np.random.randint(1,10,(3,5))

c = selecting_neg_poly(a,b)
