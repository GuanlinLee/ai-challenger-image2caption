import collections
import tensorflow as tf
from random import seed
from random import randint
import time
import numpy as np
import os
import json


slim=tf.contrib.slim

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
def resnet_arg_scope(is_training=True,weight_decay=0.01,batch_norm_decay=0.997,batch_norm_epsilon=1e-5,batch_norm_scale=True):

        batch_norm_params={'is_training':is_training,'decay':batch_norm_decay,'epsilon':batch_norm_epsilon,'scale':batch_norm_scale,'updates_collections':tf.GraphKeys.UPDATE_OPS,}
        with slim.arg_scope([slim.conv2d],weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm],**batch_norm_params):
                with slim.arg_scope([slim.max_pool2d],padding='SAME') as arg_sc:
                    return arg_sc

class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):
    'A named tuple describing a resnet block'

    def subsample(inputs, factor, scope=None):
        if factor == 1:
            return inputs
        else:
            return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

    def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
        if stride == 1:

            return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', scope=scope)

        else:
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inpu = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            return slim.conv2d(inpu, num_outputs, kernel_size, stride=stride, padding='VALID', scope=scope)





    @slim.add_arg_scope
    def stack_blocks_dense(net,blocks,outputs_collections=None):
        for block in blocks:
            with tf.variable_scope(block.scope,'block',[net]) as sc:
                for i,unit in enumerate(block.args):
                    with tf.variable_scope('unit_%d'%(i+1),values=[net]):
                        unit_depth,unit_depth_bottleneck,unit_stride=unit
                        net=block.unit_fn(net,depth=unit_depth,depth_bottleneck=unit_depth_bottleneck,stride=unit_stride)
                        net=slim.utils.collect_named_outputs(outputs_collections,sc.name,net)

        return net




    @slim.add_arg_scope
    def bottleneck(inputs,depth,depth_bottleneck,stride,outputs_collections=None,scope=None):
        with tf.variable_scope(scope,'bottleneck_v2',[inputs]) as sc:
            depth_in=slim.utils.last_dimension(inputs.get_shape(),min_rank=4)
            preact=slim.batch_norm(inputs,activation_fn=None,scope='preact')
            if depth == depth_in:
                shortcut=Block.subsample(inputs,stride,'shortcut') #shortcut=slim.conv2d(preact,depth,[1,2],stride=stride,normalizer_fn=None,activation_fn=tf.nn.relu,scope='shortcut')
            else:
                shortcut=slim.conv2d(preact,depth,[1,1],stride=stride,normalizer_fn=None,activation_fn=tf.nn.relu,scope='shortcut')#

            residual=slim.conv2d(preact,depth_bottleneck,[1,1],stride=1,scope='conv1')
            residual=Block.conv2d_same(residual,depth_bottleneck,3,stride,scope='conv2')
            residual=slim.conv2d(residual,depth,[1,1],stride=1,scope='conv3')#normalizer_fn=None,activation_fn=None,
            output=shortcut+residual

            return slim.utils.collect_named_outputs(outputs_collections,sc.name,output)

    def resnet_v2(inputs,blocks,num_classes=None,global_pool=True,include_root_block=True,reuse=None,scope=None):
        with tf.variable_scope(scope,'resnet_v2',[inputs],reuse=reuse) as sc:
            end_points_collection=sc.original_name_scope+'_end_points'
            with slim.arg_scope([slim.conv2d,Block.bottleneck,Block.stack_blocks_dense],outputs_collections=end_points_collection):
                net=inputs
                if include_root_block:
                    with slim.arg_scope([slim.conv2d],activation_fn=tf.nn.relu,normalizer_fn=None):
                        net=Block.conv2d_same(net,64,7,stride=2,scope='conv1')

                    net=slim.max_pool2d(net,[3,3],stride=1,scope='pool1')
                net=Block.stack_blocks_dense(net,blocks)
                net=slim.batch_norm(net,activation_fn=None,scope='postnorm')
                if global_pool:
                    net=tf.reduce_mean(net,[1,2],name='pool5',keep_dims=True)
                if num_classes is not None:
                    net=slim.conv2d(net,num_classes,[1,1],normalizer_fn=None,scope='logits')#activation_fn=None,normalizer_fn=None,
                end_points=slim.utils.convert_collection_to_dict(end_points_collection)
                if num_classes is not None:
                    end_points['predictions']=slim.softmax(net,scope='predictions')
                return net,end_points

def resnet_v2_200(inputs,num_classes=None,global_pool=True,reuse=None,scope='resnet_v2_200'):
        blocks=[Block('block1',Block.bottleneck,[(256,64,1)]*2+[(256,64,2)]),
                Block('block2',Block.bottleneck,[(512,128,1)]*3+[(512,128,2)]),
                Block('block3',Block.bottleneck,[(1024,256,1)]*5+[(1024,256,2)]),
                Block('block4',Block.bottleneck,[(2048,512,1)]*3)]
        return Block.resnet_v2(inputs,blocks,num_classes,global_pool,include_root_block=True,reuse=reuse,scope=scope)


training = 20
training_epochs = 500
testing = 30
testing_epochs = 2000

dimofheight=720
dimofweight=640
batch_size = 64
path = 'G:/' + 'ai_challenger_caption_train_20170902/'

fullpath = path + 'caption_train_annotations_20170902' + '.json'
fp = open(fullpath, 'r')
images = json.load(fp)


imagespath = path + '/' + 'caption_train_images_20170902' + '/'
reader = tf.WholeFileReader()
key, value = reader.read(tf.train.string_input_producer([imagespath + images[0]['image_id']]))
image0 = tf.image.decode_jpeg(value)
resized_image_AREA = tf.image.resize_images(image0, [720, 640], method=tf.image.ResizeMethod.AREA)
print(key)
print(key.shape)
for i in range(1,batch_size-1):
    imagespath = path + '/' + 'caption_train_images_20170902' + '/'
    reader = tf.WholeFileReader()
    key, value = reader.read(tf.train.string_input_producer([imagespath + images[i]['image_id']]))

    image0 = tf.image.decode_jpeg(value)
    resized_image_AREA =tf.concat([tf.image.resize_images(image0, [720, 640], method=tf.image.ResizeMethod.AREA),resized_image_AREA],0)
imgs = tf.placeholder(tf.float32, shape=[batch_size,dimofheight,dimofweight,3], name='images')
resized_image_AREA=tf.reshape(resized_image_AREA,[batch_size,dimofheight,dimofweight,3])
learning_rate=tf.placeholder(tf.float32,shape=[],name='lr')

with slim.arg_scope(resnet_arg_scope()):
    net,end_points=resnet_v2_200(resized_image_AREA,num_classes=None)

lstminput=tf.reshape(net,[batch_size,-1])


lstm = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_dim,forget_bias=1,input_size=config.embed_dim)
lstm_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=_dropout_placeholder, output_keep_prob=_dropout_placeholder)
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_dropout] * .config.layers)
initial_state = stacked_lstm.zero_state(config.batch_size, tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(stacked_lstm, all_inputs, initial_state=initial_state, scope='LSTM')
output = tf.reshape(outputs, [-1, config.hidden_dim]) # for matrix multiplication
_final_state = final_state
print ('Outputs (raw):', outputs.get_shape())
print ('Final state:', final_state.get_shape())
print ('Output (reshaped):', output.get_shape())


        # Softmax layer
with tf.variable_scope('softmax'):
        softmax_w = tf.get_variable('softmax_w', shape=[config.hidden_dim, vocab_size])
        softmax_b = tf.get_variable('softmax_b', shape=[vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
print ('Logits:', logits.get_shape())


        # Predictions
logits = logits
_predictions = predictions = tf.argmax(logits,1)
print ('Predictions:', predictions.get_shape())


        # Minimize Loss
targets_reshaped = tf.reshape(_targets_placeholder,[-1])
print ('Targets (raw):', _targets_placeholder.get_shape())
print ('Targets (reshaped):', targets_reshaped.get_shape())
with tf.variable_scope('loss'):
            # _targets is [-1, ..., -1] so that the first and last logits are not used
            # these correspond to the img step and the <eos> step
            # see: https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#sparse_softmax_cross_entropy_with_logits
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets_reshaped, name='ce_loss'))
print ('Loss:', loss.get_shape())
with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss)



sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=1)

init = tf.global_variables_initializer()

sess.run(init)

if not os.path.exists('out/'):
    os.makedirs('out/')
if not os.path.exists('model/'):
    os.makedirs('model/')
if not os.path.exists('test/'):
    os.makedirs('test/')
if not os.path.exists('tmp/'):
    os.makedirs('tmp/')
lr=5e-5
decay=0.8

print(net.shape)
print(lstminput.shape)