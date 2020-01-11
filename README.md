# Practice-ML

機械学習の学習の為のリポジトリです、研究の成果物などもここにまとめます。

## Grad_Thesis について

### 概要

学部の卒業論文に関しての再現のためのコードをまとめてあります。  
論文：[2LGAN による結合因子 CTCF の類似結合部位の生成](https://drive.google.com/file/d/1r56vpUBB4srZacMp1zR5HGXoY7XURfbM/view?usp=sharing)  
論文中の手法 1,2,3,4, 提案手法をそれぞれ jupyter notebook として

- LSGAN_iterNONE.ipynb
- LSGAN_iterFixedAdam.ipynb
- LSGAN_iterFixedAdam_leakyReLU.ipynb
- LSGAN_iterFixedAdam_Leaky_Tanh.ipynb
- LSGAN_iterFULL.ipynb  
  でそれぞれまとめてあります。

類似塩基配列の生成に関しては

- LSGAN_ver3.ipynb  
  を実行、論文中の基盤ネットワーク・提案ネットワークの生成器・識別器のモデルはそれぞれ
- seq_net.py
- seq_net_ver2.py  
  になります。

### 環境

- framework: PyTorch 1.0.0

GPU を使用する場合

- CUDA: Toolkit 10.0
- cuDNN: v7  
  環境構築に関しては[こちら](https://qiita.com/Ric418/items/b73f929739df92079451)を参考にしてください。

Google colabolatory 上で動かす場合は notebook 上の指定のセルを実行してください。

## wine_quality

wine の品質を予測するタスク
