

# ✋<b>FingerNumber_classifier ✋
### 👀프로젝트 진행에 대한 자세한 사항은 `프로젝트 진행내역.md`에 상세히 서술되어있습니다! 참고바랍니다.
### 👀코드와 실행 결과는 `FingerNumber_classifier.ipynb`을 통해 바로 확인할 수 있습니다.
  
## <b>프로젝트 설명
- 중학교 시절, 중국어 시간에 손가락으로 1부터 10까지 표기하는 방법을 배운 기억이 있다. 최근 대학 수업에서 mnist를 분류하는 코드를 작성했는데, 본인이 직접 데이터를 준비하고 훈련시켜보고 싶어 프로젝트를 진행하게 되었다. 
  <img width="566" alt="image" src="https://user-images.githubusercontent.com/46768743/142754273-66f3c4ec-bf66-4a6d-b553-f92c46081f79.png">
  - `이미지 출처: 시사중국어사 네이버 공식 블로그` https://blog.naver.com/chinasisa/221482581302

- 손가락 숫자 사진을 1부터 10까지 각각 30장을 준비하여 이를 데이터셋으로 갖는다.
  - <img width="300" alt="image" src="https://user-images.githubusercontent.com/46768743/142755240-0e1a3570-e740-434a-b3e5-99a1cc9eb99d.png"> <img width="300" alt="image" src="https://user-images.githubusercontent.com/46768743/142755325-8dd0e747-12f8-4458-96fc-36a9dde59f0c.png">



## ~~<b>사용 방법 `(CPU를 사용할 경우)` 📖~~ 일단 CPU버전은 여기까지 진행하며, GPU를 우선으로 개발을 진행한 후 CPU에 맞게 다시 재 구현 할 예정입니다.
  - 업로드 된 `dataSet.zip` 파일과 `FingerNumber_classifier.py`파일을 <b>같은 경로</b>에 저장하여 코드를 실행합니다.
    - MacBook Air m1노트북, Pycharm 환경, CPU사용 기준 실행시 학습시간은 15에포크 기준 `1시간 40분`정도 소요됩니다.
    - 현재 코드는 15에포크로 되어있습니다. 좋은 성능을 얻기 위해서는 에포크를 늘려야합니다. 15에포크의 정확도는 약 ~~79% (배치사이즈 4 기준)로 낮은 성능을 보입니다.~~ 이는 검증셋에 대한 정확도입니다.
      - 자세한 내용은 `프로젝트 진행내역.md`를 참고하시면 됩니다.
    - 사용되는 모듈들은 반드시 설치되어있어야합니다.
  - 압축 파일은 yi(1) 부터 shi(10)까지 각각의 폴더에 30장씩 준비되어있습니다.
  - 압축을 풀 때 dataSet폴더를 생성하여 그 안에 풀어야합니다.
    - ![image](https://user-images.githubusercontent.com/46768743/142754349-61194eb9-87ba-4ab0-9a20-a8bf777c8fce.png)
  - `FingerNumber_classifier.py`파일을 실행합니다.
  
## <b>사용 방법 `(GPU를 사용할 경우)` 📖
  - CUDA 를 사용하기 위해서는 NVIDIA GPU가 필요합니다. 따라서 구글 코랩으로 진행합니다.
    - GPU로 실행시 1 에포크당 대략 12초가 걸리며, 100에포크 기준 20분 정도 소요됩니다.
  - 업로드 된 `dataSet.zip` 파일과 `FingerNumber_classifier.ipynb`파일을 다운로드 받습니다.
  - 본인의 구글 드라이브(기본경로)에 FingerNumber_classifier_project 폴더를 생성합니다.
  - FingerNumber_classifier_project폴더에 dataSet 폴더를 생성합니다.
  - dataSet폴더 안에 다운로드 받은 dataSet.zip을 압축 해제합니다.
  - 다시 `구글 코랩`으로 돌아옵니다.
  - 다운로드 받은 `FingerNumber_classifier.ipynb`를 열어줍니다.
  - 구글 코랩의 상단부에 런타임 -> 런타임 유형 변경 -> 'None' 에서 GPU로 변경합니다.
  - <img width="446" alt="image" src="https://user-images.githubusercontent.com/46768743/142857000-db20c982-3364-477c-ba89-f8ecfffe595c.png">
  <br> - 위 그림 처럼 왼쪽 폴더 모양 아이콘을 클릭한 후 2번으로 표시된 아이콘을 누르면 마운트가 진행됩니다.
  - <img width="637" alt="image" src="https://user-images.githubusercontent.com/46768743/142857247-61e38577-3dbd-4d37-9abc-82506366e35b.png">
  <br> - 최종적으로 위와같은 이미지와 동일한 구조가 되어야 합니다.
  - 런타임이 끊길 경우 재 연결 후, 재 마운트를 해야합니다.
  - 코드를 실행합니다. (Ctrl + Enter)

  
## <b>사용 예제
- 여건이 된다면 실시간으로 손 모양을 인식하여 즉시 어느 숫자를 가리키는지 출력해주는 것이 최종 목표입니다.
  
## <b>개발 환경
- 언어: Python 3
- 환경: PyCharm CE
- 기간: 2021.11.09 ~ 2021.12.08

## <b>업데이트 내역
- v 0.1 : 정상적인 학습이 이루어지는 코드 작성완료
- v 0.2 : GPU사용을 위해 구글 코랩화 완료, 테스트 모델로 최증 검증 코드 추가
- v 0.3 : 개발중..
  - 새로운 사진에 대해 어떤 숫자를 가리키고 있는지 알려주는 기능을 개발합니다.
## 코드 참고사항
  - 이 코드는 영상처리와 딥러닝 과목의 10주차 과제 코드를 참고하여 작성하였습니다. 원본의 출처를 알아내는 대로 즉시 명시하도록 하겠습니다.
