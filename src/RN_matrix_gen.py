"""
RN_matrix_gen: requires integer values pertaining to nnuber of Fluid or solid particles (Fluid.pos = 16238 x 2 array, length denotes 16238)
creates a (n x n) array of random number.
 
Dependencies:
	NIL
	
"""
import numpy

def Xi(n_vectors):
    print('\n####################################')
    print('# Random numbers class constructed #')
    print('####################################\n')
    
    numpy.random.seed(2023)
    xi = numpy.random.normal(0, 1, (n_vectors, n_vectors))
    
    return  xi
