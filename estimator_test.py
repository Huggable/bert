import tensorflow as tf
import numpy as np
def model_fn_wrapper():
    def model_fn(features,labels,mode):
        x = tf.cast(features['x'],dtype=tf.float64)

        print("x",x.shape.as_list())
        w = tf.get_variable(name='W',shape = [1],dtype=tf.float64)
        b = tf.get_variable(name='b',shape = [1],dtype=tf.float64)
        pre = w*x+b
        train_op = None
        loss = None
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            y = tf.cast(features['y'], dtype=tf.float64)
            loss = tf.reduce_sum(tf.square(y-pre))
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_steps  = tf.train.get_global_step()
            train_op = tf.train.AdamOptimizer(0.01).minimize(loss,global_step=global_steps)

        return tf.estimator.EstimatorSpec(mode=mode,
        predictions=pre,
        loss=loss,
        train_op=train_op,)
    return model_fn


estimator = tf.estimator.Estimator(model_fn=model_fn_wrapper(),model_dir='./d2')
# define our data sets
x_train = np.arange(0,40)
y_train = x_train*2+1.01

x_eval = np.arange(40,80)
y_eval = x_eval*2+1.01

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train,"y":y_train}, None, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train,"y":y_train}, None, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval,"y":y_eval}, None, batch_size=4, num_epochs=1000, shuffle=False)

# train
estimator.train(input_fn=input_fn, steps=1000)
# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r" % train_metrics)
print("eval metrics: %r" % eval_metrics)
print(estimator.get_variable_names())
print(estimator.get_variable_value('W'))
print(estimator.get_variable_value('b'))

estimator2 = tf.estimator.Estimator(model_fn=model_fn_wrapper(),model_dir='./d2')
pre_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train[0:1]},shuffle=False)
r = estimator2.predict(input_fn=pre_input_fn)
for s in r:
    print(s)
