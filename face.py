import math
import face_recognition
from PIL import Image
import numpy as np
import cv2 as cv
import os

path = os.path.abspath('.')
path += '\\student\\'
is_exists = os.path.exists(path)
if not is_exists:
    os.mkdir(path)
# is_file_exists = os.path.exists(path+'studentList.txt')
# if not is_file_exists:
#     f = open(path + 'studentList.txt', 'w+')
# else:
#     f = open(path+'studentList.txt', 'a+')
# a = list(f)

source_name = "person1.JPG"  # 第一张照片（用于创建所有同学的面部信息库）
new = []  # 后续会添加的照片（用于补充面部信息库）
sign_image = '2222.JPG'  # 用于签到的照片

image = face_recognition.load_image_file(source_name)


# 返回总人数和面部编码库
# 创建所有人的面部编码库，并把每个人的脸截取后存进当前文件夹中
def build_set(new, image):
    # 查找面部
    face_locations = face_recognition.face_locations(image)
    # 总人数
    num = len(face_locations)
    img = Image.open(source_name)
    for i in range(num):
        # 开始截取
        region = img.crop((face_locations[i][3], face_locations[i][0], face_locations[i][1], face_locations[i][2]))
        # 保存图片
        region.save(path + 'Student' + str(i) + '-test.PNG')
        # f.write('Student' + str(num - 1) + '-test.PNG'+'\n')
    # 查找面部编码
    list_of_face_encodings = face_recognition.face_encodings(image)
    for new_image in new:
        image2 = face_recognition.load_image_file(new_image)
        encoding2 = face_recognition.face_encodings(image2)
        locations2 = face_recognition.face_locations(image2)
        # 新图和原有列表作对比
        for i in range(len(encoding2)):
            ifsame = face_recognition.compare_faces(list_of_face_encodings, encoding2[i], tolerance=0.4)
            if True in ifsame:
                continue
            else:
                list_of_face_encodings.append(encoding2[i])  # 脸若不在数据里，则加进去
                num = num + 1
                img2 = Image.open(new_image)
                region2 = img2.crop((locations2[i][3], locations2[i][0], locations2[i][1], locations2[i][2]))
                # 保存图片
                region2.save(path + 'Student' + str(num - 1) + '.PNG')
                # f.write('Student' + str(num - 1) + '.PNG'+'\n')
    return num, list_of_face_encodings


# 输入照片名，面部编码库和总人数，输出签到情况
def sign_in(sign_image, list_of_face_encodings, total_num):
    sign_image_array = face_recognition.load_image_file(sign_image)
    sign_locations = face_recognition.face_locations(sign_image_array)
    sign_encodings = face_recognition.face_encodings(sign_image_array)
    sign_list = np.zeros(total_num)
    im = cv.imread(sign_image)
    num_new = 0
    for p in range(len(sign_encodings)):
        ifsame1 = face_recognition.compare_faces(list_of_face_encodings, sign_encodings[p], tolerance=0.6)
        face_dis = face_recognition.face_distance(list_of_face_encodings, sign_encodings[p])
        if True not in ifsame1:  # 如果此人没出现过
            img3 = Image.open(sign_image)
            region3 = img3.crop(
                (sign_locations[p][3], sign_locations[p][0], sign_locations[p][1], sign_locations[p][2]))
            # 保存图片
            cv.rectangle(im, (sign_locations[p][3], sign_locations[p][0]), (sign_locations[p][1], sign_locations[p][2]),
                         (0, 255, 0), 3)
            p_name1 = 'new student'
            if (sign_locations[p][0] > 10):
                cv.putText(im, p_name1, (sign_locations[p][3], sign_locations[p][0] - 6), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           0.8, (0, 255, 0))
            else:
                cv.putText(im, p_name1, (sign_locations[p][3], sign_locations[p][0] + 15),
                           cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
            region3.save(path + 'Student' + str(num_new) + '.PNG')
            # f.write('Student' + str(num_new) + '.PNG'+'\n')
            num_new = num_new + 1

        else:  # 如果此人出现过，找出与此人面部编码欧氏距离最近的一个签到
            min_dis = 10
            for j in range(len(ifsame1)):
                if (ifsame1[j] == True) and (face_dis[j] < min_dis):
                    min_dis = face_dis[j]
            for k in range(len(ifsame1)):
                if (min_dis == face_dis[k]):
                    sign_list[k] = 1
                    cv.rectangle(im, (sign_locations[p][3], sign_locations[p][0]),
                                 (sign_locations[p][1], sign_locations[p][2]), (0, 255, 0), 3)
                    p_name2 = 'Student' + str(k)
                    if (sign_locations[p][0] > 10):
                        cv.putText(im, p_name2, (sign_locations[p][3], sign_locations[p][0] - 6),
                                   cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
                    else:
                        cv.putText(im, p_name2, (sign_locations[p][3], sign_locations[p][0] + 15),
                                   cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
    cv.namedWindow('result', 0)
    cv.imshow('result', im)
    is_empty = 1
    print('应到' + str(len(list_of_face_encodings)) + '人')
    print('实到' + str(len(sign_encodings)) + '人')
    list_notin = []

    for n in range(len(sign_list)):
        if sign_list[n] == 0 and is_empty == 1:
            list_notin.append(n)
            is_empty = 0
            print(str(n) + "号同学缺席")
            continue
        if sign_list[n] == 0 and is_empty == 0:
            list_notin.append(n)
            print(str(n) + "号同学缺席")
    return list_notin


# 这里是想把所有缺席同学照片显示到一张图上
def print_absent(absent_list):
    absent_num = len(absent_list)
    if absent_num > 0:
        rowNum = math.ceil(absent_num ** 0.5)
        row_img = []
        index = 0
        colNum = math.ceil(absent_num / rowNum)
        for n in range(colNum):
            img_list = []
            for i in range(rowNum):
                if index in range(len(absent_list)):
                    img = cv.imread(path + 'Student' + str(absent_list[index]) + '.PNG')
                    # img = cv.imread(path + a[i])
                    new_img = cv.resize(img, (100, 100))
                    img_list.append(new_img)
                    index += 1
                else:
                    img = np.zeros((100, 100, 3), np.uint8)
                    img_list.append(img)
            row_img.append(cv.hconcat(img_list))
        if colNum != 1:
            final_img = cv.vconcat(row_img)
        else:
            final_img = row_img[0]
        cv.namedWindow("Who is absent?", 0)
        cv.imshow("Who is absent?", final_img)
        cv.waitKey(0)
        cv.destroyAllWindows()


numbers, encodings = build_set(new, image)

absent_list = sign_in(sign_image, encodings, numbers)

print_absent(absent_list)

# f.close()
