import numpy as np


def vector_to_skew(vector):
    v = vector
    skew = np.array([[ 0 , -v[2] , v[1] ],
                     [ v[2] , 0 , -v[0] ],
                     [ -v[1] , v[0] , 0 ]
                    ])
    return skew