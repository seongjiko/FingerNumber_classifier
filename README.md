

# โ<b>FingerNumber_classifier โ
### ๐ํ๋ก์ ํธ ์งํ์ ๋ํ ์์ธํ ์ฌํญ์ `ํ๋ก์ ํธ ์งํ๋ด์ญ.md`์ ์์ธํ ์์ ๋์ด์์ต๋๋ค! ์ฐธ๊ณ ๋ฐ๋๋๋ค.
### ๐์ฝ๋์ ์คํ ๊ฒฐ๊ณผ๋ `FingerNumber_classifier.ipynb`์ ํตํด ๋ฐ๋ก ํ์ธํ  ์ ์์ต๋๋ค.
  
## <b>ํ๋ก์ ํธ ์ค๋ช
- ์คํ๊ต ์์ , ์ค๊ตญ์ด ์๊ฐ์ ์๊ฐ๋ฝ์ผ๋ก 1๋ถํฐ 10๊น์ง ํ๊ธฐํ๋ ๋ฐฉ๋ฒ์ ๋ฐฐ์ด ๊ธฐ์ต์ด ์๋ค. ์ต๊ทผ ๋ํ ์์์์ mnist๋ฅผ ๋ถ๋ฅํ๋ ์ฝ๋๋ฅผ ์์ฑํ๋๋ฐ, ๋ณธ์ธ์ด ์ง์  ๋ฐ์ดํฐ๋ฅผ ์ค๋นํ๊ณ  ํ๋ จ์์ผ๋ณด๊ณ  ์ถ์ด ํ๋ก์ ํธ๋ฅผ ์งํํ๊ฒ ๋์๋ค. 
  <img width="566" alt="image" src="https://user-images.githubusercontent.com/46768743/142754273-66f3c4ec-bf66-4a6d-b553-f92c46081f79.png">
  - `์ด๋ฏธ์ง ์ถ์ฒ: ์์ฌ์ค๊ตญ์ด์ฌ ๋ค์ด๋ฒ ๊ณต์ ๋ธ๋ก๊ทธ` https://blog.naver.com/chinasisa/221482581302

- ์๊ฐ๋ฝ ์ซ์ ์ฌ์ง์ 1๋ถํฐ 10๊น์ง ๊ฐ๊ฐ 30์ฅ์ ์ค๋นํ์ฌ ์ด๋ฅผ ๋ฐ์ดํฐ์์ผ๋ก ๊ฐ๋๋ค.
  - <img width="300" alt="image" src="https://user-images.githubusercontent.com/46768743/142755240-0e1a3570-e740-434a-b3e5-99a1cc9eb99d.png"> <img width="300" alt="image" src="https://user-images.githubusercontent.com/46768743/142755325-8dd0e747-12f8-4458-96fc-36a9dde59f0c.png">



## ~~<b>์ฌ์ฉ ๋ฐฉ๋ฒ `(CPU๋ฅผ ์ฌ์ฉํ  ๊ฒฝ์ฐ)` ๐~~ ์ผ๋จ CPU๋ฒ์ ์ ์ฌ๊ธฐ๊น์ง ์งํํ๋ฉฐ, GPU๋ฅผ ์ฐ์ ์ผ๋ก ๊ฐ๋ฐ์ ์งํํ ํ CPU์ ๋ง๊ฒ ๋ค์ ์ฌ ๊ตฌํ ํ  ์์ ์๋๋ค.
  - ์๋ก๋ ๋ `dataSet.zip` ํ์ผ๊ณผ `FingerNumber_classifier.py`ํ์ผ์ <b>๊ฐ์ ๊ฒฝ๋ก</b>์ ์ ์ฅํ์ฌ ์ฝ๋๋ฅผ ์คํํฉ๋๋ค.
    - MacBook Air m1๋ธํธ๋ถ, Pycharm ํ๊ฒฝ, CPU์ฌ์ฉ ๊ธฐ์ค ์คํ์ ํ์ต์๊ฐ์ 15์ํฌํฌ ๊ธฐ์ค `1์๊ฐ 40๋ถ`์ ๋ ์์๋ฉ๋๋ค.
    - ํ์ฌ ์ฝ๋๋ 15์ํฌํฌ๋ก ๋์ด์์ต๋๋ค. ์ข์ ์ฑ๋ฅ์ ์ป๊ธฐ ์ํด์๋ ์ํฌํฌ๋ฅผ ๋๋ ค์ผํฉ๋๋ค. 15์ํฌํฌ์ ์ ํ๋๋ ์ฝ ~~79% (๋ฐฐ์น์ฌ์ด์ฆ 4 ๊ธฐ์ค)๋ก ๋ฎ์ ์ฑ๋ฅ์ ๋ณด์๋๋ค.~~ ์ด๋ ๊ฒ์ฆ์์ ๋ํ ์ ํ๋์๋๋ค.
      - ์์ธํ ๋ด์ฉ์ `ํ๋ก์ ํธ ์งํ๋ด์ญ.md`๋ฅผ ์ฐธ๊ณ ํ์๋ฉด ๋ฉ๋๋ค.
    - ์ฌ์ฉ๋๋ ๋ชจ๋๋ค์ ๋ฐ๋์ ์ค์น๋์ด์์ด์ผํฉ๋๋ค.
  - ์์ถ ํ์ผ์ yi(1) ๋ถํฐ shi(10)๊น์ง ๊ฐ๊ฐ์ ํด๋์ 30์ฅ์ฉ ์ค๋น๋์ด์์ต๋๋ค.
  - ์์ถ์ ํ ๋ dataSetํด๋๋ฅผ ์์ฑํ์ฌ ๊ทธ ์์ ํ์ด์ผํฉ๋๋ค.
    - ![image](https://user-images.githubusercontent.com/46768743/142754349-61194eb9-87ba-4ab0-9a20-a8bf777c8fce.png)
  - `FingerNumber_classifier.py`ํ์ผ์ ์คํํฉ๋๋ค.
  
## <b>์ฌ์ฉ ๋ฐฉ๋ฒ `(GPU๋ฅผ ์ฌ์ฉํ  ๊ฒฝ์ฐ)` ๐
  - CUDA ๋ฅผ ์ฌ์ฉํ๊ธฐ ์ํด์๋ NVIDIA GPU๊ฐ ํ์ํฉ๋๋ค. ๋ฐ๋ผ์ ๊ตฌ๊ธ ์ฝ๋ฉ์ผ๋ก ์งํํฉ๋๋ค.
    - GPU๋ก ์คํ์ 1 ์ํฌํฌ๋น ๋๋ต 12์ด๊ฐ ๊ฑธ๋ฆฌ๋ฉฐ, 100์ํฌํฌ ๊ธฐ์ค 20๋ถ ์ ๋ ์์๋ฉ๋๋ค.
  - ์๋ก๋ ๋ `dataSet.zip` ํ์ผ๊ณผ `FingerNumber_classifier.ipynb`ํ์ผ์ ๋ค์ด๋ก๋ ๋ฐ์ต๋๋ค.
  - ๋ณธ์ธ์ ๊ตฌ๊ธ ๋๋ผ์ด๋ธ(๊ธฐ๋ณธ๊ฒฝ๋ก)์ FingerNumber_classifier_project ํด๋๋ฅผ ์์ฑํฉ๋๋ค.
  - FingerNumber_classifier_projectํด๋์ dataSet ํด๋๋ฅผ ์์ฑํฉ๋๋ค.
  - dataSetํด๋ ์์ ๋ค์ด๋ก๋ ๋ฐ์ dataSet.zip์ ์์ถ ํด์ ํฉ๋๋ค.
  - ๋ค์ `๊ตฌ๊ธ ์ฝ๋ฉ`์ผ๋ก ๋์์ต๋๋ค.
  - ๋ค์ด๋ก๋ ๋ฐ์ `FingerNumber_classifier.ipynb`๋ฅผ ์ด์ด์ค๋๋ค.
  - ๊ตฌ๊ธ ์ฝ๋ฉ์ ์๋จ๋ถ์ ๋ฐํ์ -> ๋ฐํ์ ์ ํ ๋ณ๊ฒฝ -> 'None' ์์ GPU๋ก ๋ณ๊ฒฝํฉ๋๋ค.
  - <img width="446" alt="image" src="https://user-images.githubusercontent.com/46768743/142857000-db20c982-3364-477c-ba89-f8ecfffe595c.png">
  <br> - ์ ๊ทธ๋ฆผ ์ฒ๋ผ ์ผ์ชฝ ํด๋ ๋ชจ์ ์์ด์ฝ์ ํด๋ฆญํ ํ 2๋ฒ์ผ๋ก ํ์๋ ์์ด์ฝ์ ๋๋ฅด๋ฉด ๋ง์ดํธ๊ฐ ์งํ๋ฉ๋๋ค.
  - <img width="637" alt="image" src="https://user-images.githubusercontent.com/46768743/142857247-61e38577-3dbd-4d37-9abc-82506366e35b.png">
  <br> - ์ต์ข์ ์ผ๋ก ์์๊ฐ์ ์ด๋ฏธ์ง์ ๋์ผํ ๊ตฌ์กฐ๊ฐ ๋์ด์ผ ํฉ๋๋ค.
  - ๋ฐํ์์ด ๋๊ธธ ๊ฒฝ์ฐ ์ฌ ์ฐ๊ฒฐ ํ, ์ฌ ๋ง์ดํธ๋ฅผ ํด์ผํฉ๋๋ค.
  - ์ฝ๋๋ฅผ ์คํํฉ๋๋ค. (Ctrl + Enter)

  
## <b>์ฌ์ฉ ์์ 
- ์ฌ๊ฑด์ด ๋๋ค๋ฉด ์ค์๊ฐ์ผ๋ก ์ ๋ชจ์์ ์ธ์ํ์ฌ ์ฆ์ ์ด๋ ์ซ์๋ฅผ ๊ฐ๋ฆฌํค๋์ง ์ถ๋ ฅํด์ฃผ๋ ๊ฒ์ด ์ต์ข ๋ชฉํ์๋๋ค.
  
## <b>๊ฐ๋ฐ ํ๊ฒฝ
- ์ธ์ด: Python 3
- ํ๊ฒฝ: Google Colab
- ๊ธฐ๊ฐ: 2021.11.09 ~ 2021.12.08

## <b>์๋ฐ์ดํธ ๋ด์ญ
- v 0.1 : ์ ์์ ์ธ ํ์ต์ด ์ด๋ฃจ์ด์ง๋ ์ฝ๋ ์์ฑ์๋ฃ
- v 0.2 : GPU์ฌ์ฉ์ ์ํด ๊ตฌ๊ธ ์ฝ๋ฉํ ์๋ฃ, ํ์คํธ ๋ชจ๋ธ๋ก ์ต์ฆ ๊ฒ์ฆ ์ฝ๋ ์ถ๊ฐ
- v 0.3 : ์๋ก์ด ์ฌ์ง์ ๋ํด ์ด๋ค ์ซ์๋ฅผ ๊ฐ๋ฆฌํค๊ณ  ์๋์ง ์๋ ค์ฃผ๋ ๊ธฐ๋ฅ์ ๊ฐ๋ฐ์๋ฃ
  
- v 1.0 : ๊ฐ๋ฐ์ค,, ์ฑ๋ฅ ํฅ์๊ณผ ์ฝ๋ ์ต์ ํ
## ์ฝ๋ ์ฐธ๊ณ ์ฌํญ
  - ์ด ์ฝ๋๋ ์์์ฒ๋ฆฌ์ ๋ฅ๋ฌ๋ ๊ณผ๋ชฉ์ 10์ฃผ์ฐจ ๊ณผ์  ์ฝ๋๋ฅผ ์ฐธ๊ณ ํ์ฌ ์์ฑํ์์ต๋๋ค. ์๋ณธ์ ์ถ์ฒ๋ฅผ ์์๋ด๋ ๋๋ก ์ฆ์ ๋ช์ํ๋๋ก ํ๊ฒ ์ต๋๋ค.
