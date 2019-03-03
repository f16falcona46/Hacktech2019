import matplotlib.pyplot as plt
import numpy as np

errs = [
        1.4142135623730951,
        1.4142135623730951,
        0.05150524761204821,
        0.05889850398833626,
        0.07142492739258055,
        0.05150524761204821,
        0.05150524761204842,
        0.07142492739258055,
        0.014284985478516204,
        0.04285495643554845,
        0.07692700106933249,
        0.04285495643554845,
        0.03194219858755938
        ]

n = [i for i in range(5, 70, 5)]

errs = np.array(errs)
n = np.array(n)

plt.plot(n, errs)
plt.xlabel('N')
plt.ylabel('Error')
plt.savefig('errs.png')
plt.show()
