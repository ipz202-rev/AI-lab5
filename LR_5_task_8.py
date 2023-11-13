import numpy as np
import neurolab as nl
import numpy.random as rand
import pylab as pl

# Генерація даних
skv = 0.07
centr = np.array([[0.1, 0.1], [0.2, 0.4], [0.5, 0.3], [0.2, 0.7], [0.6, 0.5]])
rand_norm = skv * rand.randn(100, 5, 2)
inp = np.array([centr + r for r in rand_norm])
inp.shape = (100 * 5, 2)
rand.shuffle(inp)

# Створення мережі з 2 входами і 5 нейронами
net = nl.net.newc([[0.0, 1.0], [0.0, 1.0]], 5)

# Тренування за алгоритмом «Переможець отримує все» (CWTA) на 200 ітерацій
# та виводення помилки кожних 20 епох
error = net.train(inp, epochs=200, show=20)

# Створення графіків
pl.title('Classification Problem')
pl.subplot(211)
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('error (default MAE)')
w = net.layers[0].np['w']

pl.subplot(212)
pl.plot(inp[:, 0], inp[:, 1], '.', centr[:, 0], centr[:, 1], 'yv', w[:, 0], w[:, 1], 'p')
pl.legend(['train samples', 'real centers', 'train centers'])
pl.show()
