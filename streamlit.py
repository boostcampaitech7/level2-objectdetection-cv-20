import streamlit as st
import os
import random
import json
from PIL import Image, ImageDraw, ImageFont

def load_image(image_path):
    return Image.open(image_path)

def class_color(class_name, color_map):
    if class_name not in color_map:
        color_map[class_name] = "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
    return color_map[class_name]

def draw_bbox(image, bbox, class_name, format='pascal', color_map={}):
    draw = ImageDraw.Draw(image)
    if format == 'pascal':
        xmin, ymin, xmax, ymax = bbox
    elif format == 'coco':
        x, y, w, h = bbox
        xmin, ymin, xmax, ymax = x, y, x+w, y+h

    color = class_color(class_name, color_map)
    
    # Draw rectangle
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=10)
    
    # Use a larger font size
    font_size = 40  # 폰트 크기 설정
    try:
        # Try to use a TrueType font
        font = ImageFont.truetype("./yookyung/ARIAL.TTF", font_size) 
    except IOError:
        # If the TrueType font is not available, use the default font
        font = ImageFont.load_default()
    
    # Get text size
    text_bbox = draw.textbbox((0, 0), class_name, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    draw.rectangle([xmin, ymin, xmin + text_width, ymin + text_height], fill=color)
    draw.text((xmin, ymin), class_name, fill="black", font=font)
    
    return image


def visualize_image(image_path, annotations, format, color_map):
    original_image = load_image(image_path)
    bbox_image = original_image.copy()

    image_filename = os.path.basename(image_path)
    if format == 'coco':
        image_id = next((img['id'] for img in annotations['images'] if os.path.basename(img['file_name']) == image_filename), None)
        if image_id is not None:
            bboxes_and_classes = [(ann['bbox'], next(cat['name'] for cat in annotations['categories'] if cat['id'] == ann['category_id'])) 
                                  for ann in annotations['annotations'] if ann['image_id'] == image_id]
        else:
            bboxes_and_classes = []
    else:  # Pascal VOC format
        bboxes_and_classes = [(bbox, class_name) for class_name, bboxes in annotations[image_filename].items() for bbox in bboxes]

    for bbox, class_name in bboxes_and_classes:
        bbox_image = draw_bbox(bbox_image, bbox, class_name, format, color_map)

    return original_image, bbox_image

st.title('Visualize Image and Bounding Box')

# 1. 큰 폴더 디렉토리 입력 받기
default_dir = st.text_input("Enter the main directory path:", value="")

if default_dir and os.path.exists(default_dir):
    # 2. 이미지 폴더 선택
    image_folders = [f for f in os.listdir(default_dir) if os.path.isdir(os.path.join(default_dir, f))]
    selected_folder = st.selectbox("Select the image folder:", image_folders)

    # 3. JSON 파일 선택
    json_files = [f for f in os.listdir(default_dir) if f.endswith('.json')]
    selected_json = st.selectbox("Select the JSON file:", json_files)

    if selected_folder and selected_json:
        # JSON 파일 로드
        with open(os.path.join(default_dir, selected_json), 'r') as f:
            annotations = json.load(f)

        # 이미지 파일 목록 가져오기 (오름차순 정렬)
        image_files = sorted([f for f in os.listdir(os.path.join(default_dir, selected_folder)) if f.endswith(('.jpg', '.jpeg', '.png'))],
                     key=lambda x: int(x.split('.')[0]))
        # 세션 상태 초기화
        if 'current_image_index' not in st.session_state:
            st.session_state.current_image_index = 0

        # 4. 이미지 선택을 위한 드롭다운 메뉴
        selected_image_index = st.selectbox("Select an image to visualize:", 
                                    range(len(image_files)), 
                                    format_func=lambda x: f"{image_files[x]}",
                                    index=st.session_state.current_image_index)
        
        # 드롭다운 선택 시 세션 상태 업데이트
        if selected_image_index != st.session_state.current_image_index:
            st.session_state.current_image_index = selected_image_index

        # 색상 맵 초기화
        color_map = {}

        # JSON 형식 확인
        format = 'coco' if 'images' in annotations else 'pascal'

        # 이미지 시각화
        image_path = os.path.join(default_dir, selected_folder, image_files[st.session_state.current_image_index])
        original_image, bbox_image = visualize_image(image_path, annotations, format, color_map)

        # 이미지 표시
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(bbox_image, caption="Image with Bounding Boxes", use_column_width=True)

        # 네비게이션 버튼
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Previous Image") and st.session_state.current_image_index > 0:
                st.session_state.current_image_index -= 1
        with col3:
            if st.button("Next Image") and st.session_state.current_image_index < len(image_files) - 1:
                st.session_state.current_image_index += 1

         # 클래스 색상 범례 표시
        st.subheader("Class Color Legend")
        legend_cols = st.columns(5)
        for i, (class_name, color) in enumerate(color_map.items()):
            legend_cols[i % 5].color_picker(class_name, color, disabled=True)


elif default_dir:
    st.error(f'{default_dir} does not exist.')
