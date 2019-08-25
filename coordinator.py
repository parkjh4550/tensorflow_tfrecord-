##################################
# multi thread with Coordinator
##################################

import tensorflow as tf
import threading
import time

def add(coord, i):
    while not coord.should_stop():  #check whether stop this loop
        sess.run(enque)
        """
        # if thread index == 1 come, it stops all threads -> queue size will not be 100, because threads are early stopped.
        if i == 1:
            coord.request_stop()    # if any thread calls this command, it makes all thread stop.
        """
        # never stop condition
        if i == 11:
            coord.request_stop()    # if any thread calls this command, it makes all thread stop.


sess = tf.InteractiveSession()


"""
#1. iteratively run to generate multiple threads.

# generate queue
gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
enque = queue.enqueue(gen_random_normal)


# generate coordinate
coord = tf.train.Coordinator()
threads = [threading.Thread(target=add, args=(coord, i)) for i in range(10)]
coord.join(threads)

# start thread
for t in threads:
    t.start()

print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))


"""

#2. use QueueRunner instead of iterative running.

gen_random_normal = tf.random_normal(shape=())
queue = tf.RandomShuffleQueue(capacity=100, dtypes=[tf.float32], min_after_dequeue= 1)
enque = queue.enqueue(gen_random_normal)

# run 4 threads parallelly
qr = tf.train.QueueRunner(queue, [enque]*4)
coord = tf.train.Coordinator()  # create Coordinate that will manage queues
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

coord.request_stop()    # stop all threads
coord.join(enqueue_threads) # wait until all threads stop

print("###################")
print(sess.run(queue.size()))