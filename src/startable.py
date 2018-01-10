import os
import time

import tensorflow as tf
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

import network as nw
import data_loader as dl
import evaluation_metrics as em

PATH_2_SEG_BIN = "D:/HudecL/Berkeley300/segments_squares"
MODEL_NAME = "squares_dropout_var"
IMAGE_SIZE=(32,32,3)
MAX_ITERS = 20001

def main(_arg_):
    tf.logging.set_verbosity(tf.logging.INFO)
    siamese = nw.siamese_fc(image_size=IMAGE_SIZE, margin=3)

    supsim = dl.SUPSIM(PATH_2_SEG_BIN, 128, MAX_ITERS, IMAGE_SIZE)
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
            print("Epoch {:01d}".format(epoch))
            for (batch_1, batch_2, labels), step in supsim.train:
                dropout_prob = 0.5 - epoch / 10
                step = MAX_ITERS * epoch + step
                l_rate = 0.002 / (2*float(epoch + 1))
                summary, _, loss_v = sess.run(
                    [merged_summary, train_step, siamese.loss], feed_dict={
                        siamese.x1: batch_1,
                        siamese.x2: batch_2,
                        siamese.y: labels,
                        learning_rate: l_rate,
                        siamese.dropout_prob: dropout_prob
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
                    x_s_1, x_s_2, x_l = supsim.next_batch(supsim.test.data, batch_size=30,image_size=IMAGE_SIZE)
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
        all_results_siam = None
        all_results_cosine = None
        all_results_s_col = None
        siamese.training = False
        for i in range(5):
            x_s_1, x_s_2, x_l = supsim.next_batch(supsim.test.data, batch_size=128, image_size=IMAGE_SIZE)
            vec1 = siamese.network1.eval({siamese.x1: x_s_1})
            vec2 = siamese.network2.eval({siamese.x2: x_s_2})
            similarity = sess.run(nw.similarity(vec1, vec2))
            s_cosine = [em.cosine_similarity(x1, x2) for x1,x2 in zip(x_s_1,x_s_2)]
            s_color = [em.s_colour(x1, x2) for x1,x2 in zip(x_s_1,x_s_2)]
            result = list(zip(similarity, x_l))
            print(s_cosine)
            print(s_color)

            if all_results_siam is None:
                all_results_siam = result
                all_results_cosine = s_cosine
                all_results_s_col = s_color
            else:
                all_results_siam = np.concatenate((all_results_siam,result))
                all_results_s_col = np.concatenate((all_results_s_col,s_color))
                all_results_cosine = np.concatenate((all_results_cosine, s_cosine))
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

        thr = np.arange(0, 600, 5) / 100
        tpr,fpr = np.array([em.tpr_fpr(all_results_siam,th) for th in thr]).transpose()

        auc = metrics.auc(fpr,tpr)
        plt.figure()
        plt.plot(fpr,tpr ,color='darkorange',label='ROC squares aoc: {:0.3f}'.format(auc))
        plt.plot([1,0],[0,1], color='navy', linestyle='--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()
        print(thr)


if __name__ == "__main__":
    tf.app.run()
