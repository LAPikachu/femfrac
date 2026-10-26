import numpy as np

def weibull_redistribute(values, k=4.0, scale=1.0):
    if np.isscalar(values):
        value = (np.random.weibull(k, size = 1) * scale).item()
        return value
    else:
        n = len(values)
        return np.array((np.random.weibull(k, size = n) * scale).tolist())

def get_yield(thresholds, ymin, ymax, penal = 3):
    thresholds = np.asarray(thresholds)
    return ymin + thresholds**penal * (ymax - ymin)

if __name__ == "__main__":
    weibull_sigma_dict = {0.5 : [], 1.0 : [], 1.5 : [] }
    for key in weibull_sigma_dict.keys():
        weibull_sigma_dict[key] = np.array(weibull_redistribute(np.ones(1000), k=4.0, scale=key))
    import matplotlib.pyplot as plt
    for sigma, array in zip(weibull_sigma_dict.keys(),weibull_sigma_dict.values()):
        counts, bins = np.histogram(array) 
        plt.hist(bins[:-1], bins,  weights=counts, alpha=0.5, label=sigma)
    plt.legend(loc='upper right')
    plt.show()