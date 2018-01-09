import os
import time

import network as nw
import tensorflow as tf

import data_loader as dl

PATH_2_SEG_BIN = "D:/Vision_Images/Berkeley_segmented/BSDS300/segments_squares/"
MODEL_NAME = "squares_first_run"
IMAGE_SIZE=(32,32,3)

def main(_arg_):
    tf.logging.set_verbosity(tf.logging.INFO)
    siamese = nw.siamese_fc(image_size=IMAGE_SIZE, margin=3)

    supsim = dl.SUPSIM(PATH_2_SEG_BIN, 128, 10001)
    print("Starting loading data")
    start = time.time() * 1000
    supsim.load_data()
    print((time.time() * 1000) - start, "ms")

    # print("Going to cycle")
    # for (batch1,batch2,lbls),st in supsim.train:
    #     print(st)
    #     print("Next batch")
    #     print(len(batch1))
    #     print(len(batch2))

    sess = tf.InteractiveSession()
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(
        loss=siamese.loss,
        global_step=tf.train.get_global_step()
    )
    tf.global_variables_initializer().run()

    merged_summary = tf.summary.merge_all()
    model_ckpt = 'model/' + MODEL_NAME + '/model.ckpt'
    saver = tf.train.Saver()
    if os.path.exists("model/" + MODEL_NAME + "/checkpoint"):
        saver.restore(sess, model_ckpt)

    if True:
        file_writer = tf.summary.FileWriter('board/logs/' + MODEL_NAME, sess.graph)
        # dl.SUPSIM.visualize=True
        for epoch in range(5):
            for (batch_1, batch_2, labels), step in supsim.train:
                l_rate = 0.002 / float(epoch + 1)
                summary, _, loss_v = sess.run(
                    [merged_summary, train_step, siamese.loss], feed_dict={
                        siamese.x1: batch_1,
                        siamese.x2: batch_2,
                        siamese.y: labels,
                        learning_rate: l_rate
                    })
                if step % 10 == 0:
                    file_writer.add_summary(summary, step)
                    print("Step: {:04d} loss: {:3.8f}".format(step, loss_v))

                if step % 5000 == 0 and step > 0:
                    model_name_path = 'model/' + MODEL_NAME
                    if not os.path.exists(model_name_path):
                        os.mkdir(model_name_path)
                    save_path = saver.save(sess, model_name_path + '/model.ckpt')
                    print("Model saved to file %s" % save_path)
                    x_s_1, x_s_2, x_l = supsim.next_batch(supsim.test.data,batch_size=6)
                    siamese.training = False
                    vec1 = siamese.network1.eval({siamese.x1: x_s_1})
                    vec2 = siamese.network2.eval({siamese.x2: x_s_2})
                    similarity = sess.run(nw.similarity(vec1, vec2))
                    result = list(zip(similarity, x_l))
                    print(result)
                    siamese.training = True
    else:
        tp = fp = tn = fn = 0
        siamese.training = False

        for i in range(200):
            x_s_1, x_s_2, x_l = supsim.next_batch(supsim.test.data, batch_size=128)
            vec1 = siamese.network1.eval({siamese.x1: x_s_1})
            vec2 = siamese.network2.eval({siamese.x2: x_s_2})
            similarity = sess.run(nw.similarity(vec1, vec2))
            result = list(zip(similarity, x_l))
            print(result)
            tp += sum(1 for i in result if i[0] < 1 and i[1] == 1)
            fp += sum(1 for i in result if i[0] < 1 and i[1] == 0)
            tn += sum(1 for i in result if i[0] >= 1 and i[1] == 0)
            fn += sum(1 for i in result if i[0] >= 1 and i[1] == 1)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        f1_score = 2*(precision*recall)/(precision+recall)

        print('Precision: {:0.6f} Recall: {:0.6f} Accuracy: {:0.6f} F1: {:0.6f}'.format(precision,recall,accuracy,f1_score))


if __name__ == "__main__":
    tf.app.run()
