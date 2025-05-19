from joblib import Parallel, delayed
import numpy as np

def f(x):
    return np.array([x]) + 1

if __name__ == "__main__":
    out = Parallel(n_jobs=2)(delayed(f)(i) for i in range(10))
    print(out)
