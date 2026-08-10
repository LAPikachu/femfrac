import numpy as np

def weibull_redistribute(values, k=4.0, scale=1.0):
    n = len(values)
    return (np.random.weibull(k, size = n) * scale).tolist()

def get_yield(damage, ymin, ymax, penal = 3):
    return (ymin + (1 - damage)** penal*(ymax-ymin))

if __name__ == "__main__":
    array = np.ones(1000)
    weibull_array = weibull_redistribute(array, k=10.0)
    import matplotlib.pyplot as plt
    counts, bins = np.histogram(weibull_array) 
    plt.hist(bins[:-1], bins, weights=counts)
    plt.show()