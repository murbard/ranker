import numpy as np
import random

class Oracle():

    def __init__(self, n, replace = True):
        self.n = n        
        self.count = 0
        self.replace = replace
        self.returned = [] 

    def query(self):
        # draw random pair of distinct elements
        while True:
            i, j = random.sample(range(self.n), 2)
            ordered = (i, j) if i < j else (j, i)
            if self.replace or ordered not in self.returned:
                break
        if not self.replace:
            self.returned.append(ordered)
        self.count += 1
        return ordered           
        
def test(n, replace = True):

    oracle = Oracle(n, replace)
    smaller = np.eye(n)
    while True:        
        i, j = oracle.query()        
        smaller[i, j] = 1
        # transitive closure
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if smaller[i, k] and smaller[k, j]:
                        smaller[i, j] = 1
        if np.all(smaller + smaller.T > 0):
            break
    return oracle.count

def monte_carlo(n, tol=0.01, replace = True):
    
    total_count = 0
    total_count2 = 0
    for trial in range(1, 1000000):    
        count = test(n, replace)
        total_count += count
        total_count2 += count**2
        mean = total_count / trial
        std = np.sqrt((total_count2 / trial - mean**2) / trial)
        if trial > 3 and std < tol * mean:
            break
    return mean, std
                                
if __name__ == "__main__":
    for i in range(16, 20):
        mean, std = monte_carlo(i)
        print('w/ replace:', i, mean, std)
        
    for i in range(2, 20):
        mean, std = monte_carlo(i, replace = False)
        print('wo replace:', i, mean, std)
        
        

            
    





    