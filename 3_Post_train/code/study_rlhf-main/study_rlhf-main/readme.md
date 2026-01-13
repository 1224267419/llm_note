先手撕ppo

ppo难度最大

再去手撕其他方法

使用jupytext包的
jupytext --set-formats ipynb,py your_notebook.ipynb
方法将your_notebook.py 文件配对,保存 .ipynb 文件时，.py 文件会自动更新；反之，如果你编辑并保存 .py 文件，.ipynb 文件也会被更新
从而自动更新.py和ipynb的变化