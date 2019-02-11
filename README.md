# Practice-ML
機械学習の学習の為のリポジトリです、研究の成果物などもここにまとめます。

## GradThesisについて
### 概要
学部の卒業論文に関してのコードをまとめてあります。  
論文：https://drive.google.com/file/d/1r56vpUBB4srZacMp1zR5HGXoY7XURfbM/view?usp=sharing  
論文中の手法1,2,3,4, 提案手法をそれぞれjupyter notebookとして
- LSGAN_iterNONE.ipynb
- LSGAN_iterFixedAdam.ipynb
- LSGAN_iterFixedAdam_leakyReLU.ipynb
- LSGAN_iterFixedAdam_Leaky_Tanh.ipynb
- LSGAN_iterFULL.ipynb
でそれぞれまとめてあります。

類似塩基配列の生成に関しては
- LSGAN_ver3.ipynb
を実行、生成器・識別器のモデルは
- seq_net.py
- 
になります。

### 環境
- framework: PyTorch 1.0.0

GPUを使用する場合
- CUDA: Toolkit 10.0
- cuDNN: v7

Google colabolatoryで動かす場合はnotebook上の指定のセルを実行してください。
