# This module is used for Data Preprocessing.
# Here we import 5 photos for each person, and transform them into 100 photos for each photo
# Hence, totally 500 photos for each person.

try:
    import os
except ImportError:
    import OS as os

try:
    import cv2
except ImportError:
    print'Please install OPENCV to proceed.'
    exit(0)

import numpy as np
from random import randint

def larger(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(0, 24)

    pts1 = np.float32([[num, num], [cols - num, num], [num, rows - num], [cols - num, rows - num]])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst

def smaller(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(0, 24)

    pts1 = np.float32([[num, num], [cols - num, num], [num, rows - num], [cols - num, rows - num]])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts2, pts1)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst

def lighter(img):
    # copy the basic picture, avoid the change of the basic one
    dst = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    # get the number of rows and cols of picture
    rows, cols = dst.shape[:2]
    # take a random number to use
    num = randint(20,50)

    for xi in xrange(0, cols):
        for xj in xrange(0, rows):
            for i in range(0, 3):
                if dst[xj, xi, i] <= 255 - num:
                    dst[xj, xi, i] = int(dst[xj, xi, i] + num)
                else:
                    dst[xj, xi, i] = 255
    return dst

def darker(img):
    # copy the basic picture, avoid the change of the basic one
    dst = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(20,50)

    for xi in xrange(0, cols):
        for xj in xrange(0, rows):
            for i in range(0, 3):
                if dst[xj, xi, i] >= num:
                    dst[xj, xi, i] = int(dst[xj, xi, i] - num)
                else:
                    dst[xj, xi, i] = 0
    return dst

def moveright(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)
    M = np.float32([[1,0,num],[0,1,0]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def moveleft(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)
    M = np.float32([[1,0,-num],[0,1,0]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def movetop(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)
    M = np.float32([[1,0,0],[0,1,-num]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def movebot(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)
    M = np.float32([[1,0,0],[0,1,num]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def turnright(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(3,6)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -num, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def turnleft(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(3,6)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), num, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def changeandsave(name,time,choice,img,i):
    # the new name of changed picture is "changed?.png",? means it's the ?-th picture changed
    name = './changedphoto/' + str(name) + '/' +str(time) + '_changed' + str(i) + '.png'
    # do different changes by the choice
    if choice == 1:
        newimg = larger(img)
    elif choice == 2:
        newimg = smaller(img)
    elif choice == 3:
        newimg = lighter(img)
    elif choice == 4:
        newimg = darker(img)
    elif choice == 5:
        newimg = moveright(img)
    elif choice == 6:
        newimg = moveleft(img)
    elif choice == 7:
        newimg = movetop(img)
    elif choice == 8:
        newimg = movebot(img)
    elif choice == 9:
        newimg = turnleft(img)
    elif choice == 10:
        newimg = turnright(img)
    # save the new picture
    cv2.imwrite(name, newimg)


# take Abhi's 5 photos, change each photo into 100 photos, so totally 500 for Abhi
for j in range(1,6):
    img = cv2.imread('./abhi_' + str(j) +'.jpg',1)
    # for cycle to make change randomly 100 times
    # (1,n), n for n-1 photos, (1,10), after change, 9 photos
    for i in range(1,101):
        # take a random number as the choice
        choice = randint(1,10)
        changeandsave('abhi',j,choice,img,i)

# then the same for adi's photos
for j in range(1,6):
    img = cv2.imread('./adi_' + str(j) +'.jpg',1)
    # for cycle to make change randomly 100 times
    # (1,n), n for n-1 photos, (1,10), after change, 9 photos
    for i in range(1,101):
        # take a random number as the choice
        choice = randint(1,10)
        changeandsave('adi',j,choice,img,i)

# then the same for charu's photos
for j in range(1,6):
    img = cv2.imread('./charu_' + str(j) +'.jpg',1)
    # for cycle to make change randomly 100 times
    # (1,n), n for n-1 photos, (1,10), after change, 9 photos
    for i in range(1,101):
        # take a random number as the choice
        choice = randint(1,10)
        changeandsave('charu',j,choice,img,i)

# then the same for kru's photos
for j in range(1,6):
    img = cv2.imread('./kru_' + str(j) +'.jpg',1)
    # for cycle to make change randomly 100 times
    # (1,n), n for n-1 photos, (1,10), after change, 9 photos
    for i in range(1,101):
        # take a random number as the choice
        choice = randint(1,10)
        changeandsave('kru',j,choice,img,i)

# End of module