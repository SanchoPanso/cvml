import os
import glob
import cv2


img_paths = glob.glob('./photo*')
print(img_paths)

for path in img_paths:
    img = cv2.imread(path)
    ret, mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 250, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    counter = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < 30:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        counter += 1
    print(counter)
    
    #cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    
    cv2.imshow('img', cv2.resize(img, (400, 400)))
    cv2.imshow('mask', cv2.resize(mask, (400, 400)))
    cv2.waitKey()
    