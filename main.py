from FingerNumber_classifier import run

while True:
    print("FingerNumber_Classifier")
    print("[0] 모델 새로 학습하기")
    print("[1] 학습된 모델로 새로운 사진 분류해보기")
    print("[any] 종료")

    menu = int(input("숫자를 입력해주세요: "))

    if menu == 0:
        print("모델을 새로 학습합니다.")
        print("15 에포크 기준 정확도는 79%입니다.")
        epochNum = int(input("원하는 에포크 수를 입력해주세요: "))
        print(f"{epochNum}으로 에포크를 설정하였습니다.")
        print("모델 학습을 시작합니다.")
        run(n_epochs=epochNum)

    elif menu == 1:
        print("분류를 진행합니다.")
        file_path = input("파일의 [절대경로]를 입력해주세요: ")

        pass
    else:
        print('프로그램을 종료합니다.')
        break