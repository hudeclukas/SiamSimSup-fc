import os
import time

import tensorflow as tf
import numpy as np
from sklearn import metrics
import sklearn.preprocessing as prep
import skimage.filters as filters
import matplotlib.pyplot as plt

import network as nw
import data_loader as dl
import evaluation_metrics as em

# PATH_2_SEG_BIN = "D:/Vision_Images/Berkeley_segmented/BSDS300/segments_c3+labels"
PATH_2_SEG_BIN_Berk = "D:/HudecL/Pexels/TexDat/berk+texd/TexDat/berkeley"
PATH_2_SEG_BIN_TexD = "D:/HudecL/Pexels/TexDat/official_textures"
PATH_2_SEG_Test = "D:/HudecL/test-textures"
PATH_2_SIMILARITIES = "D:/HudecL/Pexels/TexDat/similarities"
SIMILARITIES_EXTENSION = ".sim"
MODEL_NAME = "texdat_official_model"
IMAGE_SIZE = (150, 150, 1)
MAX_ITERS = 15001


def main(_arg_):
    tf.logging.set_verbosity(tf.logging.INFO)
    siamese = nw.siamese_fc(image_size=IMAGE_SIZE, margin=2.5)

    # supsim_berk = dl.SUPSIM(PATH_2_SEG_BIN_Berk, 100, IMAGE_SIZE, True)
    supsim_texd = dl.SUPSIM(PATH_2_SEG_BIN_TexD, 2, IMAGE_SIZE, True)
    print("Starting loading data")
    start = time.time() * 1000
    # supsim_berk.load_data()
    supsim_texd.load_data()
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
    model_ckpt = 'model/' + MODEL_NAME + '/model'
    saver = tf.train.Saver(var_list=tf.global_variables('siamese-fc'),max_to_keep=10)
    if os.path.exists("model/" + MODEL_NAME + "/checkpoint"):
        saver.restore(sess, model_ckpt)

    if False:
        supsim = supsim_texd
        model_name_path = 'model/' + MODEL_NAME
        if not os.path.exists(model_name_path):
            os.mkdir(model_name_path)
        saver.save(sess, model_name_path + '/model')
        file_writer = tf.summary.FileWriter('board/logs/' + MODEL_NAME, sess.graph)
        # dl.SUPSIM.visualize=True
        for epoch in range(4, 7):
            print("Epoch {:01d}".format(epoch))
            # if epoch == 3:
            #     supsim = supsim_texd
            # if epoch < 3:
            #     print("Training on Berkeley...")
            # else:
            print("Training on TexDat...")
            dropout_prob = 0.5 - epoch / 50
            siamese.dropout_prob = dropout_prob
            for step in range(0, MAX_ITERS):
                step = MAX_ITERS * epoch + step
                (batch_1, batch_2, labels) = supsim.next_batch(supsim.train.data, batch_size=100, image_size=IMAGE_SIZE)
                l_rate = 0.002 / (1.75 * float(epoch + 1))
                summary, _, loss_v = sess.run(
                    [merged_summary, train_step, siamese.loss], feed_dict={
                        siamese.x1: batch_1,
                        siamese.x2: batch_2,
                        siamese.y: labels,
                        learning_rate: l_rate
                    })
                if step % 10 == 0:
                    print("Step: {:04d} loss: {:3.8f}".format(step, loss_v))

                if step % 100 == 0 and step > 0:
                    file_writer.add_summary(summary, step)

                    # if step % 2500 == 0:
                    #     x_s_1, x_s_2, x_l = supsim.next_batch(supsim.test.data, batch_size=30, image_size=IMAGE_SIZE)
                    #     siamese.training = False
                    #     vec1 = siamese.network1.eval({siamese.x1: x_s_1})
                    #     vec2 = siamese.network2.eval({siamese.x2: x_s_2})
                    #     tf_sim = nw.similarity(vec1, vec2)
                    #     similarity = sess.run(tf_sim)
                    # error_idx = [i for i in range(len(similarity)) if
                    #              (x_l[i] == 1 and similarity[i] >= 1) or (
                    #              x_l[i] == 0 and similarity[i] < 1)]
                    # error_patch = np.array([[x_s_1[i] * 255, x_s_2[i] * 255, x_l[i], similarity[i]] for i in error_idx])
                    # if step > 5000 and step % 5000 == 0:
                    #     with tf.variable_scope('test', 'err_patch'):
                    #         ix = IMAGE_SIZE[0]
                    #         iy = IMAGE_SIZE[1]
                    #         for i in range(len(error_patch)):
                    #             img1 = tf.reshape(error_patch[i][0].astype(np.uint8), [1, ix, iy, 1])
                    #             img2 = tf.reshape(error_patch[i][1].astype(np.uint8), [1, ix, iy, 1])
                    #             tfim1 = tf.summary.image('e_p-'+str(i)+'-s1-'+str(int(error_patch[i][2]))+'sim-'+str(error_patch[i][3]), img1, max_outputs=1)
                    #             tfim2 = tf.summary.image('e_p-'+str(i)+'-s2-'+str(int(error_patch[i][2]))+'sim-'+str(error_patch[i][3]), img2, max_outputs=1)
                    #             file_writer.add_summary(tfim1.eval(), step)
                    #             file_writer.add_summary(tfim2.eval(), step)
                    #             tf.delete_session_tensor(img1)
                    #             tf.delete_session_tensor(img2)
                    #             tf.delete_session_tensor(tfim1)
                    #             tf.delete_session_tensor(tfim2)
                    # result = list(zip(similarity, x_l))
                    # print(result)
                    # del similarity
                    # siamese.training = True

                if step % 5000 == 0 and step > 0:
                    save_path = saver.save(sess, model_name_path + '/model', step, write_meta_graph=False)
                    print("Model saved to file %s" % save_path)
                    x_s_1, x_s_2, x_l = supsim.next_batch(supsim.test.data, batch_size=30, image_size=IMAGE_SIZE)
                    siamese.training = False
                    vec1 = siamese.network1.eval({siamese.x1: x_s_1})
                    vec2 = siamese.network2.eval({siamese.x2: x_s_2})
                    similarity = sess.run(nw.similarity(vec1, vec2))
                    result = list(zip(similarity, x_l))
                    del similarity
                    print(result)
                    siamese.training = True
    else:
        if True:
            supsim = supsim_texd
            tp = fp = tn = fn = 0
            siamese.training = False
            all_results_siam = None
            siamese.training = False
            path = "D:/HudecL/TestBatches/texdat_official_105k_train/"
            for i in range(90):
                x_s_1, x_s_2, x_l = supsim.next_batch(supsim.train.data, batch_size=100, image_size=IMAGE_SIZE)
                # x_s_1 = [supsim.train.data[8].superpixels[15]]
                # x_s_2 = [supsim.train.data[8].superpixels[15]]
                # x_l = [1.0]
                # x_s_2[0] = (x_s_2[0] - x_s_2[0].min())
                # x_s_2[0] = x_s_2[0] / x_s_2[0].max()
                # kernel = np.ones((5, 5)) / 9
                # x_s_2[0] = filters.median(x_s_2[0].reshape((150, 150)), kernel)
                # x_s_2[0] = prep.scale(x_s_2[0].reshape(150 * 150)).reshape((150, 150, 1))
                os.mkdir(path+"batch-"+str(i))
                os.mkdir(path+"batch-"+str(i)+"/tp")
                os.mkdir(path+"batch-"+str(i)+"/tn")
                os.mkdir(path+"batch-"+str(i)+"/fp")
                os.mkdir(path+"batch-"+str(i)+"/fn")
                vec1 = siamese.network1.eval({siamese.x1: x_s_1})
                vec2 = siamese.network2.eval({siamese.x2: x_s_2})
                tf_sim = nw.similarity(vec1, vec2)
                similarity = sess.run(tf_sim)
                for j in range(len(x_l)):
                    if x_l[j] == 1 and similarity[j] < 1:
                        name1 = path+"batch-"+str(i)+"/tp"+"/b"+str(i)+"-pair"+str(j)+"-s1-"+str(x_l[j])+"-"+str(similarity[j])+".png"
                        name2 = path+"batch-"+str(i)+"/tp"+"/b"+str(i)+"-pair"+str(j)+"-s2-"+str(x_l[j])+"-"+str(similarity[j])+".png"
                    if x_l[j] == 1 and similarity[j] > 1:
                        name1 = path + "batch-" + str(i) + "/fn" + "/b" + str(i) + "-pair" + str(j) + "-s1-" + str(x_l[j]) + "-" + str(similarity[j]) + ".png"
                        name2 = path + "batch-" + str(i) + "/fn" + "/b" + str(i) + "-pair" + str(j) + "-s2-" + str(x_l[j]) + "-" + str(similarity[j]) + ".png"
                    if x_l[j] == 0 and similarity[j] > 1:
                        name1 = path + "batch-" + str(i) + "/tn" + "/b" + str(i) + "-pair" + str(j) + "-s1-" + str(x_l[j]) + "-" + str(similarity[j]) + ".png"
                        name2 = path + "batch-" + str(i) + "/tn" + "/b" + str(i) + "-pair" + str(j) + "-s2-" + str(x_l[j]) + "-" + str(similarity[j]) + ".png"
                    if x_l[j] == 0 and similarity[j] < 1:
                        name1 = path + "batch-" + str(i) + "/fp" + "/b" + str(i) + "-pair" + str(j) + "-s1-" + str(x_l[j]) + "-" + str(similarity[j]) + ".png"
                        name2 = path + "batch-" + str(i) + "/fp" + "/b" + str(i) + "-pair" + str(j) + "-s2-" + str(x_l[j]) + "-" + str(similarity[j]) + ".png"

                    plt.imsave(name1, x_s_1[j].reshape((150,150)), cmap="gray")
                    plt.imsave(name2, x_s_2[j].reshape((150,150)), cmap="gray")
                result = list(zip(similarity, x_l))
                del tf_sim
                del similarity
                del vec1
                del vec2
                if all_results_siam is None:
                    all_results_siam = result
                else:
                    all_results_siam = np.concatenate((all_results_siam, result))
                print("{:d}. ".format(i))
                print(result)
                tp += sum(1 for i in result if i[0] <= 1 and i[1] == 1)
                fp += sum(1 for i in result if i[0] <= 1 and i[1] == 0)
                tn += sum(1 for i in result if i[0] > 1 and i[1] == 0)
                fn += sum(1 for i in result if i[0] > 1 and i[1] == 1)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            f1_score = 2 * (precision * recall) / (precision + recall)

            print('Precision: {:0.6f} Recall: {:0.6f} Accuracy: {:0.6f} F1: {:0.6f}'.format(precision, recall, accuracy,
                                                                                            f1_score))

            thr = np.arange(0, 600, 1) / 100
            tpr, fpr = np.array([em.tpr_fpr(all_results_siam, th) for th in thr]).transpose()
            auc = metrics.auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', label='ROC Siam squares aoc: {:0.3f}'.format(auc))
            plt.plot([1, 0], [1, 0], color='navy', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.show()
            print(thr)


        image_number = '86016'
        image_data = [i for i in supsim.test.images if i.name == 'segments_'+image_number+'_0.supl'][0]
        if False:

            print('File {:s} objects similarity'.format(image_data.name))

            batch_of_all_patches = dl.resize_batch_images([j for i in image_data.objects for j in i.superpixels], IMAGE_SIZE)
            batch_of_all_labels = [j for i in image_data.objects for j in i.labels]

            vec1 = siamese.network1.eval({siamese.x1: batch_of_all_patches})
            end = len(image_data.objects)
            for i in range(18, 20):
                print('Object {:d}/{:d}'.format(i+1,len(image_data.objects)))
                for j in range(len(image_data.objects[i].superpixels)):
                    sups_count = len(image_data.objects[i].labels)
                    print('\t{:d}/{:d} Superpixel: {:d} '.format(j+1, sups_count, image_data.objects[i].labels[j]))
                    batch_of_one_patch = dl.resize_batch_images([image_data.objects[i].superpixels[j]] * len(batch_of_all_patches),IMAGE_SIZE)
                    vec2 = siamese.network2.eval({siamese.x2: batch_of_one_patch})
                    tf_sim = nw.similarity(vec1, vec2)
                    similarity = sess.run(tf_sim)
                    image_data.objects[i].similarity.append(similarity)
                    del tf_sim
                    del vec2
                    del similarity
                image_data.write_similarities_to_file(PATH_2_SIMILARITIES+image_number+SIMILARITIES_EXTENSION, batch_of_all_labels, ith=i)


        if False:
            tp = tn = fp = fn = 0

            for i in range(len(image_data.objects) - 1):
                x2 = image_data.objects[i].superpixels
                x2 = dl.resize_batch_images(x2, IMAGE_SIZE)
                for si in image_data.objects[i].superpixels:
                    x1 = [si] * len(image_data.objects[i].superpixels)
                    x1 = dl.resize_batch_images(x1, IMAGE_SIZE)
                    vec1 = siamese.network1.eval({siamese.x1: x1})
                    vec2 = siamese.network2.eval({siamese.x2: x2})
                    tf_sim = nw.similarity(vec1, vec2)
                    similarity = sess.run(tf_sim)
                    print('Inner similarity of object {:d}: '.format(i))
                    print(similarity)
                    tp += sum(1 for i in similarity if i < 1)
                    fn += sum(1 for i in similarity if i >= 1)
                    del tf_sim
                    del x1
                    del vec1
                    del vec2
                    del similarity
                for j in range(i + 1, len(image_data.objects)):
                    for sj in image_data.objects[j].superpixels:
                        x1 = [sj] * len(x2)
                        x1 = dl.resize_batch_images(x1, IMAGE_SIZE)
                        vec1 = siamese.network1.eval({siamese.x1: x1})
                        vec2 = siamese.network2.eval({siamese.x2: x2})
                        tf_sim = nw.similarity(vec1, vec2)
                        similarity = sess.run(tf_sim)
                        print('Outer similarity of objects i {:d} j {:d}: '.format(i, j))
                        print(similarity)
                        fp += sum(1 for i in similarity if i < 1)
                        tn += sum(1 for i in similarity if i >= 1)
                        del tf_sim
                        del vec1
                        del vec2
                        del x1
                        del similarity
                    recall = tp / (tp + fn)
                    precision = tp / (tp + fp)
                    accuracy = (tp + tn) / (tp + fp + tn + fn)
                    f1_score = 2 * (precision * recall) / (precision + recall)

                    print('Precision: {:0.6f} Recall: {:0.6f} Accuracy: {:0.6f} F1: {:0.6f}'.format(precision, recall, accuracy,
                                                                                                    f1_score))


if __name__ == "__main__":
    tf.app.run()
