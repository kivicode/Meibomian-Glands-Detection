import numpy as np
import cv2

name = 'test4.png'


def find_sizes(pts):  # return width and height of contour given by points
    min_x, max_x = pts[0][0], pts[0][0]
    min_y, max_y = pts[0][1], pts[0][1]
    for p in pts:
        if p[0] > max_x:
            max_x = p[0]
        if p[0] < min_x:
            min_x = p[1]
        if p[1] > max_y:
            max_y = p[1]
        if p[1] < min_y:
            max_y = p[1]
    w, h = max_x - min_x, max_y - min_y
    return w, h


def gamma(img, gama=1.0):  # highlight the whole image
    invGama = 1.0 / gama
    table = np.array([((i / 255.0) ** invGama) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def find_peaks(a):  # return 'peaks' of glands on the image row
    ind = [0]
    for i in range(1, len(a) - 2):
        cur, nxt = a[i], a[i + 1]
        if cur != 0 and nxt - cur >= 100:
            ind.append(i)
    return ind


def get_row(img, orig, r, center):
    img = gamma(img, gama=10)  # highlight the image for nice working
    mask = np.zeros_like(img)  # create mask
    h, w, _ = img.shape
    f = center[0] - r[0]
    t = center[0] + r[0]
    for y in range(center[1] - r[1], center[1] + r[1]):  # go through the lines of the masked ellipse
        a = []
        for x in range(f, t):
            col = img[y][x][0]
            a.append(col)
        mask = draw_picks(mask, a, y, f)  # draw peaks of brightness of line on the mask

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # convert mask to gray scale
    mask = cv2.medianBlur(gamma(cv2.GaussianBlur(mask, (1, 11), 0), 1.5), 5)  # bluring
    mask = cv2.medianBlur(gamma(cv2.GaussianBlur(mask, (1, 11), 0), 1.5), 5)  # bluring
    mask = cv2.threshold(mask, 120, 255, 0)[1]  # delete noise [stage 1]
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours on the mask
    mask = np.zeros_like(orig) # clear previous mask
    for c in contours:  # delete noise [stage 2]
        if cv2.arcLength(c, True) > 30:
            cv2.drawContours(orig, [c], -1, (255, 0, 255), -1)  # draw final glands on the original image
            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)  # draw final glands on the mask
    cv2.imshow('mask', mask)
    return orig


def prepare(img):  # main function
    img_copy = img.copy()
    img = cv2.GaussianBlur(img, (9, 9), 10)  # blur the image
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]  # remove all pixels which brightness is less then 100
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    contours, _ = cv2.findContours(gray, 1, 2)
    biggest = contours[1]
    for c in contours:  # find the biggest contour (that's eye)
        if cv2.contourArea(c) > cv2.contourArea(biggest):
            biggest = c
    M = cv2.moments(biggest)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])  # find center of eye
    line = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[cY:cY + 1, 0:img.shape[1]]  # get pixel line on center of the eye
    start, end = -1, -1
    for i in range(len(line[0]) - 1):  # find X radius
        if line[0][i] == 0 and line[0][i + 1] == 255:
            start = i
            break
    for i in range(start, len(line[0]) - 1):
        if line[0][i + 1] == 0 and line[0][i] == 255:
            end = i
            break
    mask = np.zeros_like(img_copy)  # create mask
    rX = int((end - start) / 2)
    rY = int(rX / 2)  # find Y radius
    cv2.ellipse(mask, (cX, cY), (rX, rY), 360, 0, 360, (255,) * 3, -1)  # draw eye ellipse on the mask
    masked = cv2.bitwise_and(img_copy, mask)  # apply mask
    img_copy = get_row(masked, img_copy, [rX, rY], [cX, cY])  # stat find glands on the wrapped image
    return img_copy


def draw_picks(mask, a, y, f):  # just drawing on a mask
    peaks = find_peaks(a)
    peaks.remove(0)
    for i in range(len(a)):
        if i in peaks:
            mask[y][i + f] = (255, 255, 255)
    return mask


image = cv2.resize(cv2.imread(name), (742, 445))  # load and prepare the original image
cv2.imshow('out', prepare(image))  # show a result

while True:  # wait 'till you press 'q' and then destroy the programm
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
