## Real Time Face Alignment 개발

face alignment란 얼굴 특징을 트래킹하는 AI학습 시스템

[face-alignment](https://github.com/1adrianb/face-alignment)

Tensorflow 딥러닝을 활용

### Plan

1주차 : 실습 환경 셋팅 밑 오픈 소스 실습

* 환경 셋팅 밑 [오픈 소스 실행](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)
* 



### 프로젝트의 필요성 : 학습과 유지보수

* 기존의 face alignment를 그냥 사용하면 필요에 따라 소스를 수정하기 어려움
* 

### stack

* python
* openCV, dlib
* 

## facial landmarks detection process

* Step #1: Localize the face in the image
* Step #2: Detect the key facial structures on the face ROI
  



## AI가 사람의 얼굴을 인식하는 과정

1. detection : 이미지에서 '얼굴'에 해당하는 부분을 찾는 것
2. alignment : 눈, 코, 입, 귀, 턱 등 주요한 특징 부위들을 표시하는 단계. 얼굴의 각도 등을 조정하기 위한 준비
3. normalization : 얼굴이 돌아가 있는 등의 경우 이를 수직의 상태로 돌리는 표준화 과정
4. recogintion : 표준화된 얼굴 사진과 DB 속 얼굴 사진을 비교하여 인식하는 단계



