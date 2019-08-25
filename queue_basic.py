##############################
# queue example code
# stack and deque process
##############################

import tensorflow as tf

sess = tf.InteractiveSession()
queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string])

print("========enque process")
enque_op = queue1.enqueue(["F"])

print("before enque op : ", sess.run(queue1.size()))

enque_op.run()
print("after enque op : ", sess.run(queue1.size()))


enque_op = queue1.enqueue(["I"])
enque_op.run()

enque_op = queue1.enqueue(["F"])
enque_op.run()

enque_op = queue1.enqueue(["O"])
enque_op.run()

print("after more op : ", sess.run(queue1.size()))

print("\n=========deque process")
print("before deque op : ", sess.run(queue1.size()))

x = queue1.dequeue()
print("after deque op : ", sess.run(queue1.size()))
print("dequed element : ", x.eval())