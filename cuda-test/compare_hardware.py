import numpy
import time
import matmul

size = int(2e2)

rng = numpy.random.default_rng()
states = rng.random(size=[size])
weights = rng.random(size=[size, size])

#test numpy's matrix multiply speed:
tstart = time.perf_counter()
y1 = numpy.matmul(weights, states)
telapsed = time.perf_counter() - tstart
print("Numpy time: %.3f"%telapsed)
# print(y1.shape)

tstart = time.perf_counter()
y2 = matmul.matmul_cpu(weights, states)
telapsed = time.perf_counter() - tstart
print("Custom CPU time: %.3f"%telapsed)

tstart = time.perf_counter()
y3 = matmul.matmul_gpu(weights, states)
telapsed = time.perf_counter() - tstart
print("Custom GPU time: %.3f"%telapsed)

tstart = time.perf_counter()
y3 = matmul.matmul_gpu(weights, states)
telapsed = time.perf_counter() - tstart
print("Custom GPU time: %.3f"%telapsed)

print(numpy.mean(y1-y2))
print(numpy.mean(y1-y3))

# print(weights[1, 1])
# print(weights[2, 1])