import cv2
import os
import numpy as np
import random


class Gaussians:
    mu: float
    sigma: float
    weights: float
    omega: float


K = 4
T = 0.6
alpha = 0.2
capture = cv2.VideoCapture("umcp.mpg")
width = int(capture.get(3))
height = int(capture.get(4))

size = (width, height)
print("heigh is " + str(height))
print("width is " + str(width))


# mean = np.zeros(K, np.float64)
# mean = np.zeros(K, width * height)
# variance = np.zeros(K, width * height)
# sigma = np.ones(K, np.float64)
# weights = np.zeros([K, width * height], np.float64)
# weights = np.empty(shape=(K, width * height))
# weights.fill(1.5)
# weights = np.ones(K)/K


# weights = np.zeros(K)


def intialize():
    weights = []
    for i in range(0, T):
        rand = random.randint(1, 20)
        weights.append(rand)
    print(weights)
    nf = sum(weights)
    weights = [val / nf for val in weights]
    print(weights)


# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid|preset;slow"


def new():
    mean = np.zeros([3, 3, 4], np.float64)
    mean[:, :, :] = 5
    print(np.sqrt(mean[0]))


def gaussian_function(x: float, mu: float, sigma: float):
    value = (1 / (np.sqrt(2 * 3.14) * sigma)) * (np.exp(-0.5 * (((x - mu) / sigma) ** 2)))
    return value


def change_to_gray():
    i = 0
    frames = np.zeros((height, width, 999), np.uint8)
    print(frames.shape)
    while capture.isOpened():
        status, frame = capture.read()
        if status:
            # print(frame.shape)
            # print(i)
            frameGRAY = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # print(frameGRAY)
            frames[:, :, i] = frameGRAY
            i = i + 1
        else:
            print("end processing the video")
            break
    # print(frames[:,:,0])
    return frames, size


def gaussian_mixture_model(p, mu, sigma, weights):
    # print(weights)
    # print(mu)
    # print(sigma)
    foreground = False
    distance = abs(mu - p)
    # matched = np.zeros(K)
    min_distance_index = -1
    min_distance = -1
    matched_indexes = []
    min_weight_index = -1
    # print("p value is " + str(p))
    # print(mu)
    for i in range(0, K):
        if weights[i] != 0 and ((mu[i] + 2.5 * sigma[i]) > p > (mu[i] - 2.5 * sigma[i])):
            # matched[i] = 1
            matched_indexes.append(i)
            if min_distance == -1:
                min_distance = distance[i]
                min_distance_index = i
            else:
                if min_distance >= distance[i]:
                    min_distance = distance[i]
                    min_distance_index = i
    if min_distance_index == -1:
        # for i in range(0,K):
        #     if min_weight_index == -1:
        #         min_weight_index = i
        #     else:
        #         if weights[min_weight_index] >= weights[i]:
        #             min_weight_index = i
        min_weight_index = np.argmin(weights)
        min_distance_index = min_weight_index
        mu[min_weight_index] = p
        sigma[min_weight_index] = 20

    else:
        for i in range(0, K):
            if i in matched_indexes:
                weights[i] = (1 - alpha) * (weights[i]) + alpha
                ro = alpha * gaussian_function(p, mu[i], sigma[i])
                # print(ro)
                mu[i] = (1 - ro) * mu[i] + ro * p
                # print(mu[i])
                sigma[i] = (1 - ro) * sigma[i] + ro * np.power(p - mu[i], 2)
            else:
                weights[i] = (1 - alpha) * (weights[i])

    weights = weights / np.sum(weights)
    # print(weights)
    sortedIndexes = np.argsort(weights / sigma)
    # print(sortedIndexex)
    total = 0
    z = 0
    while (total < T):
        total = total + weights[sortedIndexes[z]]
        z += 1
        if z == K - 1:
            break
    # print(sortedIndexex)
    # print(sortedIndexex[z:])

    # if i in sortedIndexex[z:]:

    # if min_distance_index in sortedIndexex[z:]:
    #     foreground = True
    if (len(np.intersect1d(sortedIndexes[z:], matched_indexes)) > 0) or (min_distance_index in sortedIndexes[z:]):
        foreground = True
    return foreground, mu, sigma, weights


def substract_background(frame, mean, sigma, weights):
    flatFrame = np.reshape(frame, -1)
    # print(len(flatFrame))
    foreground = np.zeros(width * height)
    for i in range(0, len(flatFrame)):
        foregrroundFlag, mean, sigma, weights = gaussian_mixture_model(flatFrame[i], mean, sigma, weights)
        # print(foregrroundFlag)

        weights = weights / np.sum(weights)
        # print(weights)
        if foregrroundFlag:
            foreground[i] = flatFrame[i]

    frame = frame.reshape(-1, width)
    foregroundFrame = foreground.reshape(-1, width)
    backgroundFrame = cv2.subtract(frame, foregroundFrame)
    return backgroundFrame, mean, sigma, weights


def process_video():
    grayFrames, size = change_to_gray()
    # mean = np.zeros(K,width*height)
    # mean[0] = grayFrames[:, 0]
    flatFrame = grayFrames.reshape(height * width, 999)
    (no_of_pixels, no_of_frames) = flatFrame.shape
    no_of_frames = 100
    backgrounFrames = grayFrames
    foregroundFrames = np.zeros((height, width, 999), np.uint8)
    mean = np.zeros((K, height * width), np.float64)
    # print(mean.shape)
    # print(grayFrames[:,:, 0])
    # print(flatFrame[:,0])
    for k in range(0, K):
        mean[k, :] = flatFrame[:, k]
    print(mean.shape)
    sigma = np.ones((K, height * width), np.float64)
    sigma = np.multiply(sigma, 20)
    # print(sigma.shape)
    weights = np.ones((K, height * width), np.float64)
    # print(weights.shape)
    # weights = np.ones(K) / K
    # sigma = np.multiply(sigma, 2)
    # print(mean)
    # print(sigma)
    # print(weights)
    # weights[0] = np.array([1 for i in range(variance[0].shape[0])])
    back = cv2.VideoWriter('back1005.mp4',
                           cv2.VideoWriter_fourcc(*'XVID'),
                           30, size, False)
    fore = cv2.VideoWriter('fore1005.mp4',
                           cv2.VideoWriter_fourcc(*'XVID'),
                           30, size, False)
    for i in range(0, height - 1):
        for j in range(0, width - 1):
            # print("processing frame no : " + str(i))
            for z in range(0, no_of_frames):
                # print("processing frame no : " + str(z))
                # print("mean",mean)
                foreground, mean[:, (i)*width + j], sigma[:, (i)*width + j], weights[:, (i)*width + j] = gaussian_mixture_model(
                    grayFrames[i, j, z], mean[:, (i)*width + j], sigma[:, (i)*width + j], weights[:, (i)*width + j])
                print(foreground)
                if foreground:
                    backgrounFrames[i, j, z] = np.mean(grayFrames[i, j, :])
                    foregroundFrames[i,j,z] = 255
                    # for a in range(0, 5):
                    #     for s in range(0, 5):
                    #         for d in range(0, 5):
                    #             if((i+a) < height-1 and(j+s) < width -1 and (z+d) < no_of_frames-1):
                    #                 backgrounFrames[i + a, j + s, z + d] = np.mean(grayFrames[i, j, :])


    for i in range(no_of_frames):
        background = backgrounFrames[:, :, i]
        foreground = foregroundFrames[:,:,i]
        # background = cv2.resize(background, (352, 240))
        back.write(background)
        fore.write(foreground)
    back.release()
    fore.release()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    process_video()

