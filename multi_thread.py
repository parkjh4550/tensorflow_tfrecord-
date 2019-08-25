##################################
# stack queue using multi threads
##################################


import tensorflow as tf
import threading
import time

def add():
    for i in range(10):
        sess.run(enque)

sess = tf.InteractiveSession()

#  generate queue that can have 100 items.
print("==== generate queue")
gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
enque = queue.enqueue(gen_random_normal)


# generate 10 threads
print("==== generate 10 threads")
threads =  [ threading.Thread(target=add, args=()) for i in range(10)]

print(threads)

# check queue size several times
# start threads
for t in threads:
    t.start()

print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))


# deque 10 items from queue
x = queue.dequeue_many(10)
print("\n====== deque")
print("after deque 10 : ", sess.run(queue.size()))
print("dequed items : ", x.eval())
