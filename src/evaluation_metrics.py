import numpy as np
import math


def tpr_fpr(result, threshold):
    tp = sum(1 for i in result if i[0] <= threshold and i[1] == 1)
    fp = sum(1 for i in result if i[0] <= threshold and i[1] == 0)
    tn = sum(1 for i in result if i[0] > threshold and i[1] == 0)
    fn = sum(1 for i in result if i[0] > threshold and i[1] == 1)
    tcond = tp + fn
    if tcond > 0:
        tpr = tp / tcond
    else:
        tpr = 0
    fcond = tn + fp
    if fcond > 0:
        fpr = fp / fcond
    else:
        fpr = 0
    return tpr, fpr


def cosine_similarity(r1, r2):
    im1, im2 = image_to_channels(r1, r2)
    w, h, c = r1.shape
    hist1 = color_histogram(im1, 25, [0, 1], True)
    hist2 = color_histogram(im2, 25, [0, 1], True)

    hist1 /= w * h
    hist2 /= w * h
    hist1 = hist1.flatten()
    hist2 = hist2.flatten()

    dot = np.dot(hist1, hist2)
    denom_r1 = sum([i * i for i in hist1])
    denom_r2 = sum([i * i for i in hist2])
    if denom_r1 > 0 and denom_r2 > 0:
        return dot / (math.sqrt(denom_r1) * math.sqrt(denom_r2))
    else:
        return 0


def s_colour(r1, r2):
    im1, im2 = image_to_channels(r1, r2)

    hist1, _ = color_histogram(im1, 25, [0, 1])
    hist2, _ = color_histogram(im2, 25, [0, 1])

    hist1 = np.transpose(hist1)
    hist2 = np.transpose(hist2)

    similarity = 0
    for i in range(len(hist1)):
        similarity += min((hist1[i][0], hist2[i][0])) + min((hist1[i][1], hist2[i][1])) + min(
            (hist1[i][2], hist2[i][2]))

    return similarity


def image_to_channels(r1, r2):
    w, h, c = r1.shape
    im1 = np.reshape(r1, (w * h, c))
    im2 = np.reshape(r2, (w * h, c))
    im1 = np.transpose(im1)
    im2 = np.transpose(im2)
    return im1, im2


def color_histogram(im1, bins, range, multiply_with_color=False):
    hist_b, range_out = np.histogram(im1[0], bins=bins, range=range)
    hist_b = hist_b / np.linalg.norm(hist_b)
    hist_g = np.histogram(im1[1], bins=bins, range=range)[0]
    hist_g = hist_g / np.linalg.norm(hist_g)
    hist_r = np.histogram(im1[2], bins=bins, range=range)[0]
    hist_r = hist_r / np.linalg.norm(hist_r)
    range_out = range_out[:-1]
    if multiply_with_color:
        hist_r = hist_r * range_out
        hist_g = hist_g * range_out
        hist_b = hist_b * range_out
        hist1 = np.stack((hist_r, hist_g, hist_b))
        return hist1
    else:
        hist1 = np.stack((hist_r, hist_g, hist_b))
        return hist1, range_out

