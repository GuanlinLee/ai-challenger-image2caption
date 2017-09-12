import tensorflow as tf
import json
path = 'G:/' + 'ai_challenger_caption_train_20170902/'

fullpath = path + 'caption_train_annotations_20170902' + '.json'
fp = open(fullpath, 'r')
images = json.load(fp)
for i in range(100):
        print(images[i]['image_id'])
        for j in range(5):
            print(images[i]['caption'][j])
        imagespath = path + '/' + 'caption_train_images_20170902' + '/'
        reader = tf.WholeFileReader()
        key, value = reader.read(tf.train.string_input_producer([imagespath + images[i]['image_id']]))
        image0 = tf.image.decode_jpeg(value)
        resized_image_AREA = tf.image.resize_images(image0, [720, 640], method=tf.image.ResizeMethod.AREA)
        #resized_image_BICUBIC = tf.image.resize_images(image0, [256, 256], method=tf.image.ResizeMethod.BICUBIC)
        #resized_image_BILINEAR = tf.image.resize_images(image0, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
        #resized_image_NEAREST_NEIGHBOR = tf.image.resize_images(image0, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        #img_resize_summary_raw=tf.summary.image('raw'+str(i),tf.expand_dims(image0,0))
        img_resize_summary_AREA = tf.summary.image('AREA' + str(i), tf.expand_dims(resized_image_AREA, 0))
        #img_resize_summary_BICUBIC = tf.summary.image('BICUBIC' + str(i), tf.expand_dims(resized_image_BICUBIC, 0))
        #img_resize_summary_BILINEAR = tf.summary.image('BILINEAR' + str(i), tf.expand_dims(resized_image_BILINEAR, 0))
        #img_resize_summary_NEAREST_NEIGHBOR = tf.summary.image('NEAREST_NEIGHBOR' + str(i), tf.expand_dims(resized_image_NEAREST_NEIGHBOR, 0))


merged = tf.summary.merge_all()
init_op=tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init_op)
    cord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=cord)
    img=image0
    print(img.eval())
    #print(resized_image_AREA.eval())
    #print(resized_image_BICUBIC.eval())
    #print(resized_image_BILINEAR.eval())
    #print(resized_image_NEAREST_NEIGHBOR.eval())
    cord.request_stop()
    cord.join(threads)

    summary_writer = tf.summary.FileWriter('/tmp/', sess.graph)

    summary_all = sess.run(merged)

    summary_writer.add_summary(summary_all, 0)

    summary_writer.close()