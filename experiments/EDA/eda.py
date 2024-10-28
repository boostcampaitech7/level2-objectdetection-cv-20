# from eda import EDAprint
import json
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import seaborn as sns
import numpy as np
import pandas as pd

class EDAprint():
    def __init__(self): 
            # JSON 파일 경로
            self.image_dir = '../../dataset/'
            self.json_file_path = '../../dataset/train.json'

            # JSON 파일 읽기
            with open(self.json_file_path, 'r') as f:
                self.data = json.load(f)

            # 클래스별 이미지 크기 정보를 저장할 딕셔너리
            self.class_image_sizes = defaultdict(list)
            self.class_object_sizes = defaultdict(list)
            self.class_aspect_ratios = defaultdict(list)
            self.class_average_sizes = defaultdict(list)
            self.class_counts = defaultdict(int)
            self.extreme_objects={}
            self.image_classes = defaultdict(set)
            self.image_object_counts = defaultdict(int)
            self.class_areas = defaultdict(list)
            
            # 이미지 정보와 어노테이션 정보 추출            
            self.annotations = self.data['annotations']
            self.images = {image['id']: (image['width'], image['height']) for image in self.data['images']}
            self.categories = {category['id']: category['name'] for category in self.data['categories']}
            self.images_info = {image['id']: image['file_name'] for image in self.data['images']}
            
            # 각 어노테이션에 대해 객체 크기와 클래스를 매칭
            for annotation in self.annotations:
                image_id = annotation['image_id']
                category_id = annotation['category_id']
                bbox = annotation['bbox']  # [x, y, width, height]
                width, height = bbox[2], bbox[3]
                aspect_ratio = width / height
                class_name = self.categories[category_id]
                self.class_object_sizes[class_name].append((width, height))
                self.class_aspect_ratios[class_name].append(aspect_ratio)
                self.class_counts[class_name] += 1
                self.image_classes[image_id].add(class_name)
                area = width * height
                self.class_areas[class_name].append(np.sqrt(area))
                if class_name not in self.extreme_objects:
                    self.extreme_objects[class_name] = {'min': None, 'avg': None, 'max': None, 'areas': []}
                
                self.extreme_objects[class_name]['areas'].append((area, annotation))
                self.image_object_counts[image_id] += 1
            for class_name, sizes in self.class_object_sizes.items():
                for width, height in sizes:
                    average_size = (width + height) / 2
                    self.class_average_sizes[class_name].append(average_size)


            self.colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_counts)))
            self.allsizes = list(self.class_object_sizes.values())
            self.widthMax = 0.
            self.heightMax = 0.
            for a in self.allsizes:
                for b in a:
                    if b[0] > self.widthMax:
                        self.widthMax = b[0]
                    if b[1] > self.heightMax:
                        self.heightMax = b[1]

    def printAll(self):
        print("printNumberofObjectsperClass")
        self.printNumberofObjectsperClass()   
        print("printAverageObjectSizeDistribution")
        self.printAverageObjectSizeDistribution()
        print("printMAMlength")
        self.printMAMlength()
        print("printWidthHeightInfo")
        self.printWidthHeightInfo()
        print("printCombinedObjectSizeScatter")
        self.printCombinedObjectSizeScatter()
        print("printNumberofObjectsperImage")
        self.printNumberofObjectsperImage()
        print("printClassCoOccurrenceSimilarity")
        self.printClassCoOccurrenceSimilarity()
        print("printBoxplot")
        self.printBoxplot()        
        print("printMAMobjectsperClass")
        self.printMAMobjectsperClass()

    def printNumberofObjectsperClass(self):
         # 객체 수 리스트 생성
        class_names = list(self.class_counts.keys())
        object_counts = list(self.class_counts.values())

        # 막대 그래프 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(class_names, object_counts,color=self.colors,label=class_names)

        # x축 레이블 회전
        plt.xticks(rotation=45, ha='right')

        # 그래프 꾸미기
        plt.title('Number of Objects per Class', fontsize=16)
        plt.xlabel('Class Name', fontsize=14)
        plt.ylabel('Number of Objects', fontsize=14)

        # 각 막대 위에 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}',
                    ha='center', va='bottom')

        # 범례 추가
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 통계 출력
        total_objects = sum(object_counts)
        print(f"Total number of objects: {total_objects}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Class with most objects: {max(self.class_counts, key=self.class_counts.get)} ({max(object_counts)})")
        print(f"Class with least objects: {min(self.class_counts, key=self.class_counts.get)} ({min(object_counts)})")

    def printWidthHeightInfo(self):    

        fig, axes = plt.subplots(len(self.class_object_sizes), 3, figsize=(20, 20))
        if len(self.class_object_sizes) == 1:
            axes = [axes]

        


        for i, (class_name) in enumerate(self.class_object_sizes.keys()):
            sizes = self.class_object_sizes[class_name]
            aspect_ratios = self.class_aspect_ratios[class_name]
            
            # 객체 크기 산점도
            widths, heights = zip(*sizes)
            axes[i][0].scatter(widths, heights, alpha=0.5, color=self.colors[i % len(self.colors)], label=class_name, s=5)
            axes[i][0].set_title(f'Object Size Scatter for {class_name}')
            axes[i][0].set_xlabel('Width')
            axes[i][0].set_ylabel('Height')
            axes[i][0].set_xlim(0,self.widthMax)
            axes[i][0].set_ylim(0,self.heightMax)
            
        for i, (class_name, average_sizes) in enumerate(self.class_average_sizes.items()):
            bins = np.arange(0, max(average_sizes) + 5, 5)
            hist, bin_edges = np.histogram(average_sizes, bins=bins)
            # KDE 플롯
            axes[i][1].hist(average_sizes, bins=int(len(average_sizes)/5), alpha=0.5, color=self.colors[i % len(self.colors)], density=True, label='Histogram')
            sns.kdeplot(average_sizes, ax=axes[i][1], fill=True, color=self.colors[i % len(self.colors)])
            
            axes[i][1].set_title(f'Average Object Size Distribution for {class_name}')
            axes[i][1].set_xlabel('Average Size (Width + Height) / 2')
            axes[i][1].set_ylabel('Density')    
            axes[i][1].set_xlim(0,self.widthMax)
         # 누적합 계산 및 막대그래프
            cumulative_sum = np.cumsum(hist)
            axes[i][2].bar(bin_edges[:-1], cumulative_sum, width=5, alpha=0.7, color=self.colors[i % len(self.colors)])
            axes[i][2].set_title(f'Cumulative Average Object Size for {class_name}')
            axes[i][2].set_xlabel('Average Size (Width + Height) / 2')
            axes[i][2].set_ylabel('Cumulative Frequency')
            axes[i][2].set_xlim(0,self.widthMax)
        plt.tight_layout()
        plt.show()

    def printCombinedObjectSizeScatter(self):

        plt.figure(figsize=(8, 8))
        for i, class_name in enumerate(self.class_object_sizes.keys()):
            sizes = self.class_object_sizes[class_name]
            widths, heights = zip(*sizes)
            plt.scatter(widths, heights, alpha=0.5, color=self.colors[i % len(self.colors)], label=class_name, s=5)

        plt.title('Combined Object Size Scatter')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.legend()
        plt.xlim(0,self.widthMax)
        plt.ylim(0,self.heightMax)

        plt.tight_layout()
        plt.show()

    def printNumberofObjectsperImage(self):
        # 객체 수 리스트 생성 및 정렬
        object_counts = sorted(self.image_object_counts.values())
        y_values = [max(object_counts), np.median(object_counts), min(object_counts)]

            
        # 막대 그래프 생성
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(object_counts)), object_counts)
        plt.title('Number of Objects per Image (Sorted)')
        plt.xlabel('Image Index (Sorted by Object Count)')
        plt.ylabel('Number of Objects')
        plt.yscale('log')
        plt.tight_layout()

        for y_val in y_values:
            # 수평선 그리기
            plt.axhline(y=y_val, color='r', linestyle='--')
            
            # 선 끝에 텍스트 추가
            plt.text(len(object_counts), y_val, f'{y_val:.2f}', 
                    verticalalignment='bottom', horizontalalignment='right')
        # y축 범위 설정 (선택적)
        max_y = max(object_counts)


        # 그리드 추가
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.show()

        # 기본 통계 출력
        print(f"Total number of images: {len(self.image_object_counts)}")
        print(f"Average number of objects per image: {sum(object_counts) / len(object_counts):.2f}")
        print(f"Maximum number of objects in an image: {max(object_counts)}")
        print(f"Minimum number of objects in an image: {min(object_counts)}")
        print(f"Median number of objects in an image: {object_counts[len(object_counts)//2]}")
    
    def printClassCoOccurrenceSimilarity(self):
        # 클래스별 등장 횟수 계산
            class_occurrences = defaultdict(int)
            for classes in self.image_classes.values():
                for class_name in classes:
                    class_occurrences[class_name] += 1

            # 클래스 간 Jaccard 유사도 계산
            class_names = list(class_occurrences.keys())
            jaccard_matrix = np.zeros((len(class_names), len(class_names)))

            for i, class1 in enumerate(class_names):
                for j, class2 in enumerate(class_names):
                    if i != j:
                        intersection = sum(1 for classes in self.image_classes.values() if class1 in classes and class2 in classes)
                        union = class_occurrences[class1] + class_occurrences[class2] - intersection
                        jaccard_matrix[i][j] = intersection / union if union > 0 else 0
                    else:
                        jaccard_matrix[i][j] = 1  # 자기 자신과의 유사도는 1

            # 데이터프레임으로 변환
            jaccard_df = pd.DataFrame(jaccard_matrix, index=class_names, columns=class_names)

            # 데이터 준비
            data = jaccard_df.values
            labels = jaccard_df.index

            # 그래프 크기 설정
            fig, ax = plt.subplots(figsize=(12, 10))

            # 히트맵 그리기
            im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

            # 컬러바 추가
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Jaccard Similarity", rotation=-90, va="bottom")

            # 축 레이블 설정
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # x축 레이블 회전
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # 각 셀에 텍스트 추가
            for i in range(len(labels)):
                for j in range(len(labels)):
                    text = ax.text(j, i, f"{data[i, j]:.2f}",
                                ha="center", va="center", color="black", fontsize=8)

            ax.set_title("Class Co-occurrence Similarity (Jaccard Index)")
            fig.tight_layout()
            plt.show()

    def printBoxplot(self):
        # 시각화
        fig, axes = plt.subplots(len(self.class_object_sizes), 2, figsize=(15, 20))
        if len(self.class_object_sizes) == 1:
            axes = [axes]

        for i, (class_name) in enumerate(self.class_object_sizes.keys()):
            sizes = self.class_object_sizes[class_name]
            aspect_ratios = self.class_aspect_ratios[class_name]
            
            # 객체 크기 박스플롯
            widths, heights = zip(*sizes)
            axes[i][0].boxplot([widths, heights], labels=['Width', 'Height'])
            axes[i][0].set_title(f'Object Size Boxplot for {class_name}')
            axes[i][0].set_ylabel('Size')
            
            # 종횡비 박스플롯
            axes[i][1].boxplot(aspect_ratios)
            axes[i][1].set_title(f'Aspect Ratio Boxplot for {class_name}')
            axes[i][1].set_ylabel('Aspect Ratio')
            axes[i][1].set_xticklabels(['Aspect Ratio'])

        plt.tight_layout()
        plt.show()
    
    def printAverageObjectSizeDistribution(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        for class_name, average_sizes in self.class_average_sizes.items():
            # 5단위로 묶기 위해 히스토그램의 bin 설정
            bins = np.arange(0, max(average_sizes) + 5, 5)
            ax.hist(average_sizes, bins=bins, alpha=0.5, label=class_name)

        ax.set_title('Average Object Size Distribution')
        ax.set_xlabel('Average Size (Width + Height) / 2')
        ax.set_ylabel('Frequency')
        ax.legend()

        plt.tight_layout()
        plt.show()

    def printMAMobjectsperClass(self):
        # 각 클래스별로 가장 작은, 평균, 큰 객체 찾기
        for class_name, obj_info in self.extreme_objects.items():
            sorted_areas = sorted(obj_info['areas'], key=lambda x: x[0])
            min_obj = sorted_areas[0][1]
            max_obj = sorted_areas[-1][1]
            avg_obj = sorted_areas[len(sorted_areas) // 2][1]
            
            self.extreme_objects[class_name]['min'] = min_obj
            self.extreme_objects[class_name]['avg'] = avg_obj
            self.extreme_objects[class_name]['max'] = max_obj

        # 이미지 출력
        for class_name, objs in self.extreme_objects.items():
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for j, obj_type in enumerate(['min', 'avg', 'max']):
                obj_info = objs[obj_type]
                image_id = obj_info['image_id']
                bbox = obj_info['bbox']
                image_path = self.image_dir + self.images_info[image_id]

                # 이미지 열기 및 바운딩 박스 그리기
                image = Image.open(image_path)
                axes[j].imshow(image)
                axes[j].add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none'))
                
                # 화살표 추가
                axes[j].annotate(f'{obj_type.capitalize()} Object', xy=(bbox[0]+bbox[2], bbox[1]), xytext=(bbox[0]+bbox[2]+30, bbox[1]-50),
                                arrowprops=dict(facecolor='yellow'))
                
                # 크기 정보 추가
                axes[j].text(0.5, -0.1, f'Size: {bbox[2]}x{bbox[3]}', ha='center', va='center', transform=axes[j].transAxes)
                
                axes[j].set_title(f'{obj_type.capitalize()} Object for {class_name}')
                axes[j].axis('off')
            
            plt.tight_layout()
            plt.show()

    def printMAMlength(self):
        class_stats = {}
        for class_name, areas in self.class_areas.items():
            class_stats[class_name] = {
                'min': min(areas),
                'avg': np.mean(areas),
                'max': max(areas)
            }

        # 시각화
        fig, ax = plt.subplots(figsize=(10, 6))

        class_names = list(class_stats.keys())
        x = np.arange(len(class_names))

        min_values = [stats['min'] for stats in class_stats.values()]
        avg_values = [stats['avg'] - stats['min'] for stats in class_stats.values()]
        max_values = [stats['max'] - stats['avg'] for stats in class_stats.values()]

        # 누적 막대 그래프 생성
        ax.bar(x, min_values, label='Min', alpha=0.7, color=self.colors[0])
        ax.bar(x, avg_values, bottom=min_values, label='Avg', alpha=0.7, color=self.colors[1])
        ax.bar(x, max_values, bottom=[sum(x) for x in zip(min_values, avg_values)], label='Max', alpha=0.7, color=self.colors[2])

        ax.set_ylabel('Square Root of Area')
        ax.set_title('Minimum, Average, and Maximum Square Root of Areas by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()

        # y축을 로그 스케일로 설정 (면적 차이가 큰 경우 유용)
        #ax.set_yscale('log')

        plt.tight_layout()
        plt.show()

        # 통계 출력
        for class_name, stats in class_stats.items():
            print(f"{class_name}:")
            print(f"  Min: {stats['min']:.2f}")
            print(f"  Avg: {stats['avg']:.2f}")
            print(f"  Max: {stats['max']:.2f}")