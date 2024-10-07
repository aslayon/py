import cv2
import torch
import numpy as np
import json
from pathlib import Path

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 비디오 파일 열기
video_path = 'yolodetecting/test.mp4'
cap = cv2.VideoCapture(video_path)

# 결과를 저장할 디렉토리 생성
output_dir = Path('output_data')
output_dir.mkdir(exist_ok=True)

frame_count = 0
whole_frame_data = []
while cap.isOpened(): # mp4 끝나기 전까지.
    ret, frame = cap.read() # 읽기 성공 여부, 프레임(이미지 데이터)
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR -> RGB 로 변환

    # YOLOv5로 객체 탐지
    results = model(frame_rgb) # 해당 프레임 객체 인식.

    # 탐지 결과를 리스트로 변환
    detections = results.pred[0].cpu().numpy()
    print(results.pred[0].cpu().numpy())
    # 프레임별 데이터를 저장할 리스트
    frame_data = []

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        obj = { 
            'class': int(cls),
            'confidence': float(conf),  #신뢰도는 일단 보류.
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)]  # 중심 좌표 추가
        }
        frame_data.append(obj)

    '''# 프레임 데이터를 JSON 파일로 저장
    with open(output_dir / f'frame_{frame_count:06d}.json', 'w') as f:
        json.dump(frame_data, f)

    results.save(save_dir = output_dir / f'frame_{frame_count:06d}.jpg') # 이미지 저장'''

    whole_frame_data.append(frame_data)

    frame_count += 1

cap.release()
#print(f"처리 완료: {frame_count} 프레임이 저장되었습니다.")

#  0프레임의 결과를 보면, 핸드폰을 book 으로 인식. 클래스 번호로 인식하는 건 효과가 없을 것으로 예상. 

# 객체 추적을 위해, 탐지한 객체의 중심 좌표를 활용하여 , 다음 프레임과의 차이가 가장 적은 객체를 동일 객체로 판단.
class_trace = []


for i, frame_data in enumerate(whole_frame_data):
    if i == 0:
        # 첫 프레임의 객체들을 class_trace에 추가

        for data in frame_data:
            tmp = []
            tmp.append(data)
            class_trace.append(tmp)
    else:
        
        for data in frame_data:
            flag = -1
            min_diff = float('inf')
            for j, class_t in enumerate(class_trace):
                #두 중심 좌표의 차이
                current_center = np.array(data['center'])
                previous_center = np.array(class_t[-1]['center']) # 탐지한 객체들의 가장 최근 위치와 비교

                diff = np.linalg.norm(current_center - previous_center) #피타고라스 정리( 사이 거리 구하기 ) 유클리드 거리 계산

                if diff < min_diff:
                    min_diff = diff
                    flag = j

            if flag != -1 and min_diff < 55:  # 차이가 65 이하인 경우만
                class_trace[flag].append(data)
            else:
                tmp = []
                tmp.append(data)
                class_trace.append(tmp)



for i,class_t in enumerate(class_trace):
    with open(output_dir / f'class_{i:06d}_tracing.json', 'w') as f:
        json.dump(class_t, f)
                
#해당 데이터들로는 진위여부를 파악하기 힘듬. 길찾기 프로그램 처럼, 해당 객체의 이동경로만 이루어져 있음.

#따라서, 97개의 프레임이 존재하므로, [97(프레임 수)][100(예상 객체 최대 수)] 로 class_trace 를 지정하고, 다시 class_trace 를 순회하며 일치하는 객체 확인,


class_trace_by_frame = [[] for _ in range(97)]

for i, frame_data in enumerate(whole_frame_data):
    

    for data in frame_data:
        
        for j,class_t in enumerate(class_trace):
            for k,data_tr in enumerate(class_t):
                if data_tr == data:
                    data['class'] = j
                    class_trace_by_frame[i].append(data)



video_path = 'yolodetecting/test.mp4'
cap = cv2.VideoCapture(video_path)

# 결과를 저장할 디렉토리 생성
output_dir = Path('output_data')
output_dir.mkdir(exist_ok=True)


# 색상 맵 생성 (클래스별로 다른 색상 사용)
color_map = np.random.randint(0, 255, size=(100, 3))  # 100개의 클래스까지 지원

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count < len(class_trace_by_frame):
        for obj in class_trace_by_frame[frame_count]:
            # bbox 가져오기
            x1, y1, x2, y2 = obj['bbox']
            class_id = obj['class']
            
            # 바운딩 박스 그리기
            color = tuple(map(int, color_map[class_id % 100]))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 클래스 이름과 ID 추가
            label = f'Class: {class_id}'
            cv2.putText(frame, label, (int(x1), int(y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 결과 이미지 저장
        cv2.imwrite(str(output_dir / f'frame_{frame_count:06d}_bbox.jpg'), frame)

    frame_count += 1

cap.release()
print(f"프레임별 이미지 저장 완료: {frame_count} 프레임")