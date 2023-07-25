from io import StringIO
from pathlib import Path
import streamlit as st
import time
from detect import detect
import os
import sys
import argparse
from PIL import Image

############################
from FaceModel.Facemodel import predict
############################
##########图片分类##########
from ImageClassificationModel.clf import predict
############################



def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


# if __name__ == '__main__':
def myMain():
    st.title('✨深度学习模型部署：实现多功能')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    source = ("🧩图片检测", "🎈视频检测","⭐人脸对比","✨图片分类")
    source_index = st.sidebar.selectbox("选择输入", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "上传图片", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.convert('RGB')
                picture = picture.save(f'data/images/{uploaded_file.name}')
                opt.source = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False
    elif source_index == 1:
        uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'data/videos/{uploaded_file.name}'
        else:
            is_valid = False
    ######################################################################
    elif source_index == 2:##### 人脸对比：判断是否是同一个人
        col1, col2 = st.columns(2)
        file_up1 = st.sidebar.file_uploader("Upload an image", type="jpg", key="file_up1")
        file_up2 = st.sidebar.file_uploader("Upload an image", type="jpg", key="file_up2")

        option2 = st.multiselect(
            'you can select two images',
            ['a', 'b'])

        # file_up1 = st.file_uploader("Upload an image", type="jpg",key="file_up1")
        # file_up2 = st.file_uploader("Upload an image", type="jpg",key="file_up2")
        is_valid = False
        if file_up1 is None or file_up2 is None:

            if option2 == ["a", 'b']:
                with col1:
                    image1 = Image.open(r".\FaceModel\image\a.jpg")
                    file_up1 = r".\FaceModel\image\a.jpg"
                    st.image(image1, caption='Uploaded Image.', use_column_width=True)

                with col2:
                    image2 = Image.open(r".\FaceModel\image\b.jpg")
                    file_up2 = r".\FaceModel\image\b.jpg"

                    st.image(image2, caption='Uploaded Image.', use_column_width=True)
                    st.write("")
                if st.button("Submit", use_container_width=True):
                    st.write("Just a second...")
                    probability = predict(image1, image2)
                    print(probability)
                    st.success('successful prediction')
                    st.write("两张图片是同一个人的概率是Prediction：", probability.item())
                    st.balloons()


        else:
            with col1:
                image1 = Image.open(file_up1)
                st.image(image1, caption='Uploaded Image.', use_column_width=True)
            with col2:
                image2 = Image.open(file_up2)
                st.image(image2, caption='Uploaded Image.', use_column_width=True)
                st.write("")
            if st.button("Submit"):
                st.write("Just a second...")
                probability = predict(image1, image2)
                st.success('successful prediction')
                st.write("两张图片是同一个人的概率是Prediction：", probability.item())
                st.balloons()
    else:  #######################################图片分类##################################
        is_valid = False
        option = st.selectbox(
            'Choose the model you want to use?',
            ('resnet50', 'resnet101', 'densenet121', 'shufflenet_v2_x0_5', 'mobilenet_v2'))
        ""
        option2 = st.selectbox(
            'you can select some image',
            ('image_dog', 'image_snake'))

        file_up = st.sidebar.file_uploader("Upload an image", type="jpg")
        if file_up is None:
            if option2 == "image_dog":
                image = Image.open("./ImageClassificationModel/image/dog.jpg")
                file_up = "./ImageClassificationModel/image/dog.jpg"
            else:
                image = Image.open("./ImageClassificationModel/image/snake.jpg")
                file_up = "./ImageClassificationModel/image/snake.jpg"
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Just a second...")
            labels, fps = predict(file_up, option)

            # print out the top 5 prediction labels with scores
            st.success('successful prediction')
            st.write("前五名预测类别及概率如下：")
            for i in labels:
                st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])

            # print(t2-t1)
            # st.write(float(t2-t1))
            # st.write("")
            # st.metric("", "FPS:   " + str(fps))

        else:
            image = Image.open(file_up)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Just a second...")
            labels, fps = predict(file_up, option)

            # print out the top 5 prediction labels with scores
            st.success('successful prediction')
            st.write("前五名预测类别及概率如下：")
            for i in labels:
                st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])

            # print(t2-t1)
            # st.write(float(t2-t1))
            # st.write("")
            # st.metric("", "FPS:   " + str(fps))

    ######################################################################
    if is_valid:
        print('valid')
        if st.button('开始检测'):

            detect(opt)

            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))

                    st.balloons()
            else :
                with st.spinner(text='Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        st.video(str(Path(f'{get_detection_folder()}') / vid))

                    st.balloons()
            # else:
            #
            #     if st.button("Submit", use_container_width=True):
            #         st.write("Just a second...")
            #         probability = predict(image1, image2)
            #         print(probability)
            #         st.success('successful prediction')
            #         st.write("两张图片是同一个人的概率是Prediction：", probability.item())
            #         st.balloons()
