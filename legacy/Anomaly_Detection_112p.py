#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys, os
import traceback
import glob
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# if kekbco-user01 python, delete below~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import paramiko

import subprocess
import datetime
import time
import threading
from scipy.optimize import minimize

# 20240226追加
import keras.backend
# 20240204追加
import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

import tkinter as tk # GUI用
from tkinter import ttk
import tkinter.messagebox
from PIL import Image, ImageTk, ImageOps  # 画像データ用

import epics # EPICS用
from epics import ca

# 20240221追加
import gc

# %matplotlib inline

# 1次元データを表示する関数(1個)
# x, y, x_limit, y_limit, si, clはリスト
def show_data1(x, y, title, x_label, y_label, x_limit, y_limit, y_legend, si, cl, tx, fc, al):
    # Figureを設定
    fig = plt.figure(figsize = (4.2, 3.5), tight_layout=True)
            
    # Axesを追加
    ax = fig.add_subplot(1, 1, 1)

    # Axesのタイトルの設定
    ax.set_title(title, fontsize = 9)

    # 軸ラベルの設定
    ax.set_xlabel(x_label, size = 12, weight = "light")
    ax.set_ylabel(y_label, size = 12, weight = "light")
    
    # x軸の目盛設定
    ax.set_xlim(x_limit[0], x_limit[1])
    # y軸の目盛設定
    ax.set_ylim(y_limit[0], y_limit[1])
    
    # 目盛方向を両側に設定
    # 目盛の長さを5ポイントに設定
    # 目盛と目盛ラベルの色をblackに設定
    ax.tick_params(direction = "inout", length = 5, colors = "black")
    
    # y軸の目盛を指数にする
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    
    # データをプロット
    if(y_legend[0] == 'Measured'):
        k = 1
    else:
        k = 0

    for i in range(k, len(x)):
        ax.scatter(x[i], y[i], s = si[i], marker = "o", color = cl[i], label = y_legend[i])
    
    # 凡例
    ax.legend(loc = "upper center", fontsize = 6)
    
    # テキスト
    ax.text(x_limit[1] * 0.05, y_limit[1] * 0.77, tx, fontsize = 6)
    
    # face color
    ax.set_facecolor(fc)
    ax.set_alpha(al)

# 1次元データを表示する関数(2個、3個)
# x, y, x_limit, y_limit, si, clはリスト
def show_data2(x, y, title, x_label, y_label, x_limit, y_limit, y_legend, si, cl, tx, fc, al):

    # Figureを設定
    if(len(x) == 3) or (len(x) == 4):
        fig, axes = plt.subplots(1, 2, figsize = (8, 3.5), tight_layout=True)
        plt.subplots_adjust(wspace = 0.3)
    if(len(x) == 5):
        fig, axes = plt.subplots(1, 3, figsize = (12, 3.5), tight_layout=True)
        plt.subplots_adjust(wspace = 0.3)
    
    # データをプロット
    for i in range(0, len(x)):
        if(len(x) == 3):
            if(i < 2): k = 0
            if(i == 2): k = 1
        if(len(x) == 4):
            if(i < 3): k = 0
            if(i == 3): k = 1
        if(len(x) == 5):
            if(i < 3): k = 0
            if(i == 3): k = 1
            if(i == 4): k = 2
        
        # Axesのタイトルの設定
        axes[k].set_title(title[i], fontsize = 9)

        # 軸ラベルの設定
        axes[k].set_xlabel(x_label[i], size = 10, weight = "light")
        axes[k].set_ylabel(y_label[i], size = 10, weight = "light")

        # x軸の目盛設定
        axes[k].set_xlim(x_limit[i])
        # y軸の目盛設定
        axes[k].set_ylim(y_limit[i])
    
        # 目盛方向を両側に設定
        # 目盛の長さを5ポイントに設定
        # 目盛と目盛ラベルの色をblack設定
        axes[k].tick_params(direction = "inout", length = 4, colors = "black")
    
        # y軸の目盛を指数にする
        axes[k].ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    
        axes[k].scatter(x[i], y[i], s = si[i], marker = "o", color = cl[i], label = y_legend[i])
    
        # 凡例
        axes[k].legend(loc = "upper center", fontsize = 6)
        
        # テキスト
        if(i == 1):
            axes[k].text(x_limit[i][1] * 0.3, y_limit[i][1] * 0.77, tx, fontsize = 6)
    
        # face color
        axes[k].set_facecolor(fc)
        axes[k].set_alpha(al)

# 20240330
# 回帰曲線の解析解を計算する関数(1次元、Logsitic)
# x0、x1は調査変数値(ベクトル)、tはターゲット値(ベクトル)
def fit_line(x0, x1, t): 
    mx = np.mean(x0)
    mt = np.mean(t)
    mtx = np.mean(t * x0)
    mxx = np.mean(x0 * x0)
    
    # 各係数の計算
    w0 = (mtx - mt * mx) / (mxx - mx**2)
    w1 = 0.
    w2 = mt - w0 * mx
    
    # 各係数をベクトルにして返す
    return np.array([w0, w1, w2]) 

# 回帰曲線の解析解を計算する関数(2次元、Logsitic)
# x0、x1は調査変数値(ベクトル)、tはターゲット値(ベクトル)
def fit_plane(x0, x1, t): 
    c_tx0 = np.mean(t * x0) - np.mean(t) * np.mean(x0)
    c_tx1 = np.mean(t * x1) - np.mean(t) * np.mean(x1)
    c_x0x1 = np.mean(x0 * x1) - np.mean(x0) * np.mean(x1)
    v_x0 = np.var(x0)
    v_x1 = np.var(x1)
    
    # 各係数の計算
    w0 = (c_tx1 * c_x0x1 - v_x1 * c_tx0) / (c_x0x1**2 - v_x0 *v_x1)
    w1 = (c_tx0 * c_x0x1 - v_x0 * c_tx1) / (c_x0x1**2 - v_x0 *v_x1)
    w2 = -w0 * np.mean(x0) - w1 * np.mean(x1) + np.mean(t)
    
    # 各係数をベクトルにして返す
    return np.array([w0, w1, w2])

# 回帰曲線の解を勾配法で計算する関数(2次元、制限付きで計算する)
# w_1は初期値、x0、x1は調査変数値(ベクトル)、tはターゲット値(ベクトル)
def fit_plane_num(w_1, x0, x1, t): 
    # 制限
    if w_1[0] < 0.: w_1[0] = 0.
    if w_1[1] < 0.: w_1[1] = 0.
    if w_1[2] < 1.e-9: w_1[2] = 1.e-9
        
    alpha1 = 1.e-7 # w0の学習率
    alpha2 = 1.e-12 # w1の学習率
    alpha3 = 1.e-1 # w2の学習率
    tau_max = 2000 # 繰り返しの最大数
    eps = 5.e-9 # 繰り返しをやめる勾配の絶対値の閾値
    
    # 各係数の計算
    for tau in range(1, tau_max):
        dmse = dmse_plane(x0, x1, t, w_1)
        
        w_1[0] = w_1[0] - alpha1 * dmse[0]
        w_1[1] = w_1[1] - alpha2 * dmse[1]
        w_1[2] = w_1[2] - alpha3 * dmse[2]
        if w_1[0] < 0.: w_1[0] = 0.
        if w_1[1] < 0.: w_1[1] = 0.
        if w_1[2] < 1.e-9: w_1[2] = 1.e-9
        
        # forループの終了判定
        if max(np.absolute(dmse)) < eps:
            break
    
    # 結果
    w0 = w_1[0]
    w1 = w_1[1]
    w2 = w_1[2]
    
    # 各係数をベクトルにして返す
    return np.array([w0, w1, w2]), dmse, tau

# 2次元の平均二乗誤差(MSE)の勾配を計算する関数：x0、x1は調査変数値(ベクトル)、
# tはターゲット値(ベクトル)、wは係数(ベクトル)
def dmse_plane(x0, x1, t, w):
    y = w[0] * x0 +w[1] * x1 + w[2]
    d_w0 = 2 * np.mean((y-t) * x0)
    d_w1 = 2 * np.mean((y-t) * x1)
    d_w2 = 2 * np.mean(y-t)
    
    # 各項の勾配を返す
    return d_w0, d_w1, d_w2 

# 20240330
# 線のMSEを計算する関数:Mean Squared Error 平均二乗誤差
def mse_line(x0, x1, t, w):
    y_r =w[0] * x0 + w[2]
    # 最小圧力は3.E-8 Paとする
    y = np.maximum(y_r, 3.e-8)
    # 平均二乗誤差
    mse = np.mean((y-t)**2)
    # 最大二乗誤差
    maxse = np.max((y-t)**2)
    
    #平均二乗誤差と最大二乗誤差を返す
    return mse, maxse

# 面のMSEを計算する関数:Mean Squared Error 平均二乗誤差
def mse_plane(x0, x1, t, w):
    y_r =w[0] * x0 +w[1] * x1 + w[2]
    # 最小圧力は3.E-8 Paとする
    y = np.maximum(y_r, 3.e-8)
    # 平均二乗誤差
    mse = np.mean((y-t)**2)
    # 最大二乗誤差
    maxse = np.max((y-t)**2)
    
    #平均二乗誤差と最大二乗誤差を返す
    return mse, maxse

# モデルB(Tail)の関数
# wは係数(ベクトル)、xは調査変数値(ベクトル)
def model_B(x, w):
    # overflow対策
    a = np.minimum(-w[0] * x, 700)
    
    # y = w[2] * np.exp(-w[0] * x) + w[3] / (x + w[1])
    y = w[2] * np.exp(a) + w[3] / (x + w[1])
    
    return y

# 20240320変更
# モデルC(Tail)の関数
# wは係数(ベクトル)、xは調査変数値(ベクトル)
def model_C(x, w):
    # overflow対策
    a = np.minimum(-abs(w[1]) * x, 700)
    
    # モデルの関数 20240117変更
    # y = w[0] * np.exp(a) + w[2] / (x + w[3]) + w[4]
    # y = w[0] * np.exp(a) + abs(w[2]) / (x + 1) + w[3]
    # y = w[0] * np.exp(a) + abs(w[2]) / (x + 1)
    y = abs(w[0]) * np.exp(a) + abs(w[2]) / (x + 1) + abs(w[3])
    
    return y

# 20240320追加 DIFの時
# モデルC_dif(Tail)の関数
# wは係数(ベクトル)、xは調査変数値(ベクトル)
def model_C_dif(x, w):
    # overflow対策
    a = np.minimum(-w[1] * x, 700)
    
    # モデルの関数 20240117変更
    # y = w[0] * np.exp(a) + w[2] / (x + w[3]) + w[4]
    # y = w[0] * np.exp(a) + abs(w[2]) / (x + 1) + w[3]
    # y = w[0] * np.exp(a) + abs(w[2]) / (x + 1)
    y = w[0] * np.exp(a) + w[2] / (x + 1) + w[3]
    
    return y

# モデルB(Tail)のMSEを計算する関数
# wは係数、xは調査変数値(ベクトル)、tはターゲット値(ベクトル)
def mse_model_B(w, x, t):
    y = model_B(x, w)
    # MSEの計算
    # overflow対策
    a = np.minimum(np.abs(y-t), 1.e+150)
    mse = np.mean(a**2)
    # mse = np.mean((y-t)**2)
    
    return mse # MSEを返す

# 20240120変更
# モデルC(Tail)のMSEを計算する関数
# wは係数、xは調査変数値(ベクトル)、tはターゲット値(ベクトル)
def mse_model_C(w, x, t):
    y = model_C(x, w)
    # MSEの計算
    # overflow対策
    a = np.minimum(np.abs(y-t), 1.e+150)
    mse = np.mean(a**2)
    # mse = np.mean((y-t)**2)
    
    return mse # MSEを返す

# 20240320追加 DIFの時
# モデルC_dif(Tail)のMSEを計算する関数
# wは係数、xは調査変数値(ベクトル)、tはターゲット値(ベクトル)
def mse_model_C_dif(w, x, t):
    y = model_C_dif(x, w)
    # MSEの計算
    # overflow対策
    a = np.minimum(np.abs(y-t), 1.e+150)
    mse = np.mean(a**2)
    # mse = np.mean((y-t)**2)
    
    return mse # MSEを返す

# モデルB(Tail)の最大SEを計算する関数
def maxse_model_B(w, x, t):
    y = model_B(x, w)
    # 最大SEの計算
    maxse = np.max((y-t)**2)

    return maxse # MaxSEを返す

# 20240120変更
# モデルC(Tail)の最大SEを計算する関数
def maxse_model_C(w, x, t):
    y = model_C(x, w)
    # 最大SEの計算
    maxse = np.max((y-t)**2)

    return maxse # MaxSEを返す

# 20240320追加 DIFの時
# モデルC_dif(Tail)の最大SEを計算する関数
def maxse_model_C_dif(w, x, t):
    y = model_C_dif(x, w)
    # 最大SEの計算
    maxse = np.max((y-t)**2)

    return maxse # MaxSEを返す

# モデルB(Tail)の係数を最適化する関数
# w_1は初期値、xは調査変数値(ベクトル)、tはターゲット値(ベクトル)
def fit_model_B(w_1, x, t):
    # minimizeを使う
    res1 = minimize(mse_model_B, w_1, args = (x, t), method = "powell")
    
    return res1.x

# 20240120変更
# モデルC(Tail)の係数を最適化する関数
# w_1は初期値、xは調査変数値(ベクトル)、tはターゲット値(ベクトル)
def fit_model_C(w_1, x, t):
    # minimizeを使う
    res1 = minimize(mse_model_C, w_1, args = (x, t), method = "powell", tol = 1.e-5)
    # res1 = minimize(mse_model_C, w_1, args = (x, t), method = "powell")

    return res1.x

# 20240320追加 DIFの時
# モデルC_dif(Tail)の係数を最適化する関数
# w_1は初期値、xは調査変数値(ベクトル)、tはターゲット値(ベクトル)
def fit_model_C_dif(w_1, x, t):
    # minimizeを使う
    res1 = minimize(mse_model_C_dif, w_1, args = (x, t), method = "powell", tol = 1.e-3)
    # res1 = minimize(mse_model_C_dif, w_1, args = (x, t), method = "powell")

    return res1.x

# FNN (Forward Neural Network)
def FNN(wv, M, K, x):
    N, D = x.shape # 入力次元　今はD = 2、M = 2、K = 3
    w = wv[:M * (D + 1)] # 中間層ニューロンへの重み
    w = w.reshape(M, (D + 1))
    v = wv[M * (D + 1):] # 出力層ニューロンへの重み
    v = v.reshape(K, (M + 1))
    b = np.zeros((N, M + 1)) # 中間層ニューロンへの入力総和
    z = np.zeros((N, M + 1)) # 中間層ニューロンの出力
    a = np.zeros((N, K)) # 出力層ニューロンの入力総和
    y = np.zeros((N, K)) # 出力層ニューロンの出力
    for n in range(N):
        # 中間層の計算
        for m in range(M):
            # ダミーの1を列に加える
            b[n, m] = np.dot(w[m, :], np.r_[x[n, :], 1])
            z[n, m] = Sigmoid(b[n, m])
        # 出力層の計算
        z[n, M] = 1 # ダミーニューロン
        wkz = 0
        for k in range(K):
            a[n, k] = np.dot(v[k, :], z[n, :])
            wkz = wkz + np.exp(a[n, k])
        for k in range(K):
            y[n, k] = np.exp(a[n, k]) / wkz
            
    return y, a, z, b

# シグモイド関数 (Sigmoid)
def Sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    
    return y

# 20240204追加
# 対数にする関数(xはarray)
def funclog(x):
    y = np.copy(x)
    for i in range(len(x)):
        if (x[i] >= 0.):
            x1 = np.maximum(x[i], 1.e-12)
            y[i] = np.log10(x1 * 1.e12)
        elif (x[i] < 0):
            x1 = np.maximum(abs(x[i]), 1.e-12)
            y[i] = -np.log10(x1 * 1.e12)
    
    return y

# 20240204追加
# 対数にする関数(xは変数)
def funclog1(x):
    if (x >= 0.):
        x1 = np.maximum(x, 1.e-12)
        y = np.log10(x1 * 1.e12)
    elif (x < 0):
        x1 = np.maximum(abs(x), 1.e-12)
        y = -np.log10(x1 * 1.e12)
    
    return y

# 20240226追加
# 標準化する関数
def standard_n(n, axis = None, ddof = 0):
    # 平均値を計算
    mean_n = np.mean(n, axis = axis, keepdims = True)
    # 標準偏差を計算 ddof = 0なら標準偏差、ddof = 1なら不偏標準偏差
    std_n = np.std(n, axis = axis, keepdims = True, ddof = ddof)
    # 標準化の計算
    if(std_n == 0.):
        standard_n = (n - mean_n) / 1.0
    else:
        standard_n = (n - mean_n) / std_n
    
    return standard_n

# 20270227追加
# 標準化する関数1
def standard_n1(n,  me1, st1):
    
    # 標準化の計算
    if(st1 == 0.):
        standard_n = (n - me1) / 1.0
    else:
        standard_n = (n - me1) / st1
    
    return standard_n

# kekb-co-userに入り、kblogrdを使ってデータを読み出す
# Data_Paraはリスト,DT_Para、Mode_Paraは文字変数、List_Paraはリスト
def Get_kekbcouser(Ring_Name, Date_Para, DT_Para, Mode_Para, List_Para): 
    Error_ssh = 'none'
    
    Mode_Para_c = Mode_Para #20230816+c
    if(Mode_Para == 'CHK_Strg'):  #20230816+c
        Mode_Para_c = 'DIF_Strg'  #20230816+c
    elif(Mode_Para == 'HK_Tail'):  #20230816+c
        Mode_Para_c = 'DIF_Tail'  #20230816+c
    
    # if kekbco-user01 python, delete below~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    '''
   
    # まず、kekb-login.kek.jpに入る
    vm = paramiko.SSHClient()
    vm.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    vm.connect('kekb-login1.kek.jp', username='suetsugu', password='ys1002')
    
    #次に、kekb-co-userに入る
    vmtransport = vm.get_transport()
    dest_addr = ('kekb-co-user', 22) #edited#
    local_addr = ('kekb-login1.kek.jp', 22) #edited#
    try:
        vmchannel = vmtransport.open_channel("direct-tcpip", dest_addr, local_addr)
    except Exception as e:
        print(traceback.format_exc())
        print(str(e))
        Error_ssh = str(e)
        tk.messagebox.showinfo('SSH Error', '    Restart again     ')
        return Error_ssh
        
    jhost = paramiko.SSHClient()
    jhost.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    #jhost.load_host_keys('/home/osmanl/.ssh/known_hosts') #キーを使う時
    jhost.connect('kekb-co-user', username='suetsugu', password='ys1002', sock=vmchannel)
    
    #各側室についてループ
    for LC_Room in List_Para:
        for Date_Range in Date_Para:
            filename = LC_Room + '_' + Mode_Para_c + '_Data_' + Date_Range +'.txt'  #20230816+c
            if(LC_Room == 'BEAM'):
                filename = Ring_Name + '_' + Mode_Para_c + '_Beam_Data_' + Date_Range +'.txt'  #20230816+c
        
            #コマンドを作る
            CMD = 'sh ' + LC_Room + '.sh ' + Date_Range + DT_Para + ' ' + filename
            print('CMD: ', CMD, 'is running')
            print('Please wait for a while')
            
            # コマンドの実行
            stdin, stdout, stderr = jhost.exec_command(CMD) 

            # コマンド実行結果を変数に格納
            cmd_result = ''
            for line in stdout:
                cmd_result += line
    
            # 実行結果を出力
            print(cmd_result)

    # Close kekb-co-user、kekb-login1.kek.jpを抜ける
    jhost.close()
    vm.close()
    
    return Error_ssh

    '''

    # if kekbco-user01 python, use below~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #各側室についてループ
    for LC_Room in List_Para:
        for Date_Range in Date_Para:
            filename = LC_Room + '_' + Mode_Para_c + '_Data_' + Date_Range +'.txt'  #20230816+c
            if(LC_Room == 'BEAM'):
                filename = Ring_Name + '_' + Mode_Para_c + '_Beam_Data_' + Date_Range +'.txt'  #20230816+c
        
            #コマンドを作る
            CMD = 'sh ' + LC_Room + '.sh ' + Date_Range + DT_Para + ' ' + filename
            
            try:
                subprocess.run(CMD, shell = True)
            except Exception as e:
                print('subprocess.check_call() failed', CMD)
                print(str(e))
                Error_ssh = str(e)
                tk.messagebox.showinfo('CMD Error', '     Restart    ')
                return Error_ssh
                
    return Error_ssh
    
# SFTPを使って、kekb-co-userからローカルPCにデータを持ってくる
# Data_Paraはリスト,Mode_Paraは文字変数、List_Paraはリスト
def Get_SFTP(Ring_Name, Date_Para, Mode_Para, List_Para):
    
    # if kekbco-user01 python, just return~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    return
    
    Mode_Para_c = Mode_Para #20230816+c
    if(Mode_Para == 'CHK_Strg'):  #20230816+c
        Mode_Para_c = 'DIF_Strg'  #20230816+c
    elif(Mode_Para == 'CHK_Tail'):  #20230816+c
        Mode_Para_c = 'DIF_Tail'  #20230816+c
        
    # SFTP接続先の設定
    HOST = 'kekb-user01.kek.jp'
    PORT = 22
    SFTP_USER = 'suetsugu'
    SFTP_PASSWORD = 'ys1002'

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    client.connect(HOST, port = PORT, username = SFTP_USER, password = SFTP_PASSWORD) # パスワード認証
    try:
        # SFTPセッション開始
        sftp_connection = client.open_sftp()

        for LC_Room in List_Para:
            for Date_Range in Date_Para:
                filename = LC_Room + '_' + Mode_Para_c + '_Data_' + Date_Range +'.txt' #20230816+c
                if(LC_Room == 'BEAM'):
                    filename = Ring_Name + '_' + Mode_Para_c + '_Beam_Data_' + Date_Range +'.txt' #20230816+c
    
                # ファイルのダウンロード
                sftp_connection.get(filename, filename)
                print('file: ', filename, ' is transfered')
    finally:
        client.close() # kekb-co-at1.kek.jpを抜ける
        
# kekb-co-userに入り、deldateのデータを消す
def Rmv_kekbcouser(Ring_Name, Last_Refd_i, Deletedate_dt0, Day_advance0):

    # if kekbco-user01 python, delete below ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    '''
    
    # まず、kekb-login.kek.jpに入る
    vm = paramiko.SSHClient()
    vm.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    vm.connect('kekb-login1.kek.jp', username='suetsugu', password='ys1002')
    
    #次に、kekb-co-userに入る
    vmtransport = vm.get_transport()
    dest_addr = ('kekb-co-user', 22) #edited#
    local_addr = ('kekb-login1.kek.jp', 22) #edited#
    vmchannel = vmtransport.open_channel("direct-tcpip", dest_addr, local_addr)
    
    jhost = paramiko.SSHClient()
    jhost.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    #jhost.load_host_keys('/home/osmanl/.ssh/known_hosts') #キーを使う時
    jhost.connect('kekb-co-user', username='suetsugu', password='ys1002', sock=vmchannel)

    # Last_Refd+Ref_Pd日から1~4日 20240315変更
    for k in range(1, 5):
        day_advance =  -(Day_advance0 + k)
        Deletedate_dt = Deletedate_dt0 + datetime.timedelta(days = day_advance)
        Deldate = '{:%Y%m%d}'.format(Deletedate_dt)
    
        #file名
        filename = Ring_Name + '*' + Deldate + '*.*'
        
        #コマンドを作る
        CMD = 'rm ' + filename
        print('CMD: ', CMD, 'is running')
        print('Please wait for a while')
            
        # コマンドの実行
        stdin, stdout, stderr = jhost.exec_command(CMD) 

        # コマンド実行結果を変数に格納
        cmd_result = ''
        for line in stdout:
            cmd_result += line
    
        # 実行結果を出力
        print(cmd_result)
        print(Day_advance0 + k, ' days old data were removed in kekb-co-user')
    
    # Last_Refd日から1~Last_Refd-2日 20240315変更
    for k in range(1, Last_Refd_i - 1):
        day_advance =  -(Last_Refd_i - k)
        Deletedate_dt = Deletedate_dt0 + datetime.timedelta(days = day_advance)
        Deldate = '{:%Y%m%d}'.format(Deletedate_dt)
    
        #file名
        filename = Ring_Name + '*' + Deldate + '*.*'
        
        #コマンドを作る
        CMD = 'rm ' + filename
        print('CMD: ', CMD, 'is running')
        print('Please wait for a while')
            
        # コマンドの実行
        stdin, stdout, stderr = jhost.exec_command(CMD) 

        # コマンド実行結果を変数に格納
        cmd_result = ''
        for line in stdout:
            cmd_result += line
    
        # 実行結果を出力
        print(cmd_result)
        print(Last_Refd_i - k, ' days old data were removed in kekb-co-user')

    # Close kekb-co-user、kekb-login1.kek.jpを抜ける
    jhost.close()
    vm.close()
     
    return
    
    '''
    
    # if kekbco-user01 python, use below ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # Last_Refd+Ref_Pd日から1~4日 20240315変更
    for k in range(1, 5):
        day_advance =  -(Day_advance0 + k)
        Deletedate_dt = Deletedate_dt0 + datetime.timedelta(days = day_advance)
        Deldate = '{:%Y%m%d}'.format(Deletedate_dt)
        
        #file名
        filename = Ring_Name + '*' + Deldate + '*.txt'
        
        #コマンドを作る
        CMD = 'rm ' + filename
 
        try:
            subprocess.run(CMD, shell = True)
        except Exception as e:
            print('subprecess.check_call() failed', CMD)
            print(str(e))
        
        #file名
        filename = Ring_Name + '*' + Deldate + '*.npy'
        
        #コマンドを作る
        CMD = 'rm ' + filename
 
        try:
            subprocess.run(CMD, shell = True)
        except Exception as e:
            print('subprecess.check_call() failed', CMD)
            print(str(e))
             
        #file名
        filename = Ring_Name + '*' + Deldate + '*.npz'
        
        #コマンドを作る
        CMD = 'rm ' + filename
 
        try:
            subprocess.run(CMD, shell = True)
        except Exception as e:
            print('subprecess.check_call() failed', CMD)
            print(str(e))
        
        print(Day_advance0 + k, ' days old data (txt, npy, npz) were removed in kekb-co-user')
    
    # Last_Refd日から1~Last_Refd-2日 20240315変更
    for k in range(1, Last_Refd_i-1):
        day_advance =  -(Last_Refd_i - k)
        Deletedate_dt = Deletedate_dt0 + datetime.timedelta(days = day_advance)
        Deldate = '{:%Y%m%d}'.format(Deletedate_dt)
        
        #file名
        filename = Ring_Name + '*' + Deldate + '*.txt'
        
        #コマンドを作る
        CMD = 'rm ' + filename
 
        try:
            subprocess.run(CMD, shell = True)
        except Exception as e:
            print('subprecess.check_call() failed', CMD)
            print(str(e))
        
        #file名
        filename = Ring_Name + '*' + Deldate + '*.npy'
        
        #コマンドを作る
        CMD = 'rm ' + filename
 
        try:
            subprocess.run(CMD, shell = True)
        except Exception as e:
            print('subprecess.check_call() failed', CMD)
            print(str(e))
            
        #file名
        filename = Ring_Name + '*' + Deldate + '*.npz'
        
        #コマンドを作る
        CMD = 'rm ' + filename
 
        try:
            subprocess.run(CMD, shell = True)
        except Exception as e:
            print('subprecess.check_call() failed', CMD)
            print(str(e))
            
        print(Last_Refd_i - k, ' days old data (txt, npy, npz) were removed in kekb-co-user')
        
    return

# Excel型の日付をkblogrd型の日付にする
def Convert_Excel_to_Kblogrd(excelform):
    mont = excelform[0:2]
    days = excelform[3:5]
    year = excelform[6:10]
    hour = excelform[11:13]
    mins = excelform[14:16]
    secs = excelform[17:20]
    
    dtime = datetime.datetime(int(year), int(mont), int(days), int(hour), int(mins), int(secs))
    kblogrdform = '{:%Y%m%d%H%M%S}'.format(dtime) # 書式変更
    
    return kblogrdform

# kblogrd型の日付をExcel型の日付にする
def Convert_Kblogrd_to_Excel(kbform):
    year = kbform[0:4]
    mont = kbform[4:6]
    days = kbform[6:8]
    hour = kbform[8:10]
    mins = kbform[10:12]
    secs = kbform[12:14]
    etime = mont + '/' + days + '/' + year + ' ' + hour + ':' + mins + ':' + secs 
    
    return etime

# kblogrd型の日付をdatetime型の日付にする
def Convert_Kblogrd_to_Dtime(kbform):
    year = kbform[0:4]
    mont = kbform[4:6]
    days = kbform[6:8]
    hour = kbform[8:10]
    mins = kbform[10:12]
    secs = kbform[12:14]
    dtime = datetime.datetime(int(year), int(mont), int(days), int(hour), int(mins), int(secs))
    
    return dtime

# Excel型の日付をdatetime型の日付にする
def Convert_Excel_to_Dtime(excelform):
    mont = excelform[0:2]
    days = excelform[3:5]
    year = excelform[6:10]
    hour = excelform[11:13]
    mins = excelform[14:16]
    secs = excelform[17:20]
    
    dtime = datetime.datetime(int(year), int(mont), int(days), int(hour), int(mins), int(secs))
    
    return dtime

# ファイルの行数を計算する 20240118 変更
def Cal_File_Row(Date_Range, DT_Para):
    stdstart = int(Date_Range[:14]) #　kblord形式
    stdend = int(Date_Range[15:]) # kblogrd形式
    
    stdstartd = Convert_Kblogrd_to_Dtime(str(stdstart)) # dtime形式
    stdendd = Convert_Kblogrd_to_Dtime(str(stdend)) #d time形式
    td = stdendd - stdstartd
    
    dtpara = float(DT_Para[1:])
    stdngyou = float(td.total_seconds()) / dtpara
    
    return stdngyou

# ArrayをArrayファイル(.npy)とTextファイル(.txt)に保存する。
def Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save):
    # Arrayのまま保存する
    np.save(Array_File_Name_to_Save, Array_to_Save)

    # Arrayをテキストファイルで保存する
    np.savetxt(Text_File_Name_to_Save, Array_to_Save, fmt = '%s', delimiter=',')
    
# 202401変更
# StrgモードのAbnormalファイルがあるかどうか確認し、なければ作る    
def Check_Strg_Abnormal_Result_File(Abnormal_Result_Strg_File_Name, Abnormal_Result_Strg_Text_File_Name):
    path = Abnormal_Result_Strg_File_Name
    is_file = os.path.isfile(path)
    if(is_file == False): # arrayのファイルがなかったら
        path = Abnormal_Result_Strg_Text_File_Name # textファイルを探す
        is_file = os.path.isfile(path)
        if(is_file == False): # csvのファイルもなかったら
            print(Abnormal_Result_Strg_File_Name, ' was newly created')
            Abnormal_Result_Strg_List = np.array([['Date_Range_DIF', 'Date_Range_STD', 'Abort_Timing', 'Record Name', 
                                                   'RMSE_cal', 'RMSE_std', 'Beam_Max', 'MaxRSE_std', 'Max_Pre', 'Max_Sel_Pre', 
                                                   'Avg_Sel_Pre', 'Max_Cal', 'Avg_Cal', 'RMSE_STEP_Max', 'STEP_Max_Avg', 
                                                   'RMSE_STEP_Mean', 'STEP_Mean_Avg', 'W0_std', 'W1_std', 'W2_std', 'W0_dif', 
                                                   'W1_dif', 'W2_dif', 'RMSE_dif', 'MaxRMSE_dif', 'Cause_no']])

            # arrayとして保存する
            np.save(Abnormal_Result_Strg_File_Name, Abnormal_Result_Strg_List)
        else: # textファイルがあったらtextファイルを読み込んでarrayファイルにする
            print(Abnormal_Result_Strg_File_Name, ' was created from text file')
            content =[]
            with open (Abnormal_Result_Strg_Text_File_Name, encoding = 'utf8', newline = '') as f:
                csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
                n_row = 0 #もとのデータの行数を求める変数：初期値
                for row in csvreader:
                    content = content + row #contentは1行のリストになる
                    n_row += 1 #もとのcsvファイル(リスト)の行数となる

            # もとのデータのリスト'content'をベクトルのデータ'data'にする
            data = np.array(content)
            #ベクトルのデータ'data'をn_row行、n_column列の行列"d"にする
            Abnormal_Result_Strg_List = data.reshape(n_row, -1)
            # print(Abnormal_Result_Strg_List)
            # arrayとして保存する
            np.save(Abnormal_Result_Strg_File_Name, Abnormal_Result_Strg_List)
    else: # arrayのファイルがあるなら
        Abnormal_Result_Strg_List = np.load(Abnormal_Result_Strg_File_Name)

    return Abnormal_Result_Strg_List

#202401変更
# StrgモードのNormalファイルがあるかどうか確認し、なければ作る    
def Check_Strg_Normal_Result_File(Normal_Result_Strg_File_Name, Normal_Result_Strg_Text_File_Name):
    path = Normal_Result_Strg_File_Name
    is_file = os.path.isfile(path)
    if(is_file == False): # arrayのファイルがなかったら
        path = Normal_Result_Strg_Text_File_Name # textファイルを探す
        is_file = os.path.isfile(path)
        if(is_file == False): # csvのファイルもなかったら
            print(Normal_Result_Strg_File_Name, ' was newly created')
            Normal_Result_Strg_List = np.array([['Date_Range_DIF', 'Date_Range_STD', 'Abort_Timing', 'Record Name', 'RMSE_cal', 
                                                 'RMSE_std','Beam_Max', 'MaxRSE_std', 'Max_Pre', 'Max_Sel_Pre', 
                                                 'Avg_Sel_Pre', 'Max_Cal', 'Avg_Cal', 'RMSE_STEP_Max', 'STEP_Max_Avg', 
                                                 'RMSE_STEP_Mean', 'STEP_Mean_Avg', 'W0_std', 'W1_std', 'W2_std', 'W0_dif', 
                                                 'W1_dif', 'W2_dif', 'RMSE_dif', 'MaxRMSE_dif', 'Cause_no']])

            # arrayとして保存する
            np.save(Normal_Result_Strg_File_Name, Normal_Result_Strg_List)
        else: # textファイルがあったらtextファイルを読み込んでarrayファイルにする
            print(Normal_Result_Strg_File_Name, ' was created from text file')
            content =[]
            with open (Normal_Result_Strg_Text_File_Name, encoding = 'utf8', newline = '') as f:
                csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
                n_row = 0 #もとのデータの行数を求める変数：初期値
                for row in csvreader:
                    content = content + row #contentは1行のリストになる
                    n_row += 1 #もとのcsvファイル(リスト)の行数となる
                # print('n_row =', n_row)
                # print(content)
            # もとのデータのリスト'content'をベクトルのデータ'data'にする
            data = np.array(content)
            #ベクトルのデータ'data'をn_row行、n_column列の行列"d"にする
            Normal_Result_Strg_List = data.reshape(n_row,-1)
            print(Normal_Result_Strg_List)
            # arrayとして保存する
            np.save(Normal_Result_Strg_File_Name, Normal_Result_Strg_List)
    else: # arrayのファイルがあるなら
        Normal_Result_Strg_List = np.load(Normal_Result_Strg_File_Name)

    return Normal_Result_Strg_List

# 202401変更
# TailモードのAbnormalファイルがあるかどうか確認し、なければ作る 
def Check_Tail_Abnormal_Result_File(Abnormal_Result_Tail_File_Name, Abnormal_Result_Tail_Text_File_Name):
    path = Abnormal_Result_Tail_File_Name
    is_file = os.path.isfile(path)
    if(is_file == False): # arrayのファイルがなかったら
        path = Abnormal_Result_Tail_Text_File_Name # textファイルを探す
        is_file = os.path.isfile(path)
        if(is_file == False): # textのファイルもなかったら
            print(Abnormal_Result_Tail_File_Name, ' was newly created')
            Abnormal_Result_Tail_List = np.array([['Date_Range_DIF', 'Date_Range_STD', 'Abort_Timing', 'Record Name', 'RMSE_cal', 
                                                   'RMSE_std','Beam_Max', 'MaxRSE_std', 'Max_Pre', 'Avg_Pre', 'Max_Cal', 
                                                   'Avg_Cal', 'Max_Raw_Pre', 'Avg_Raw_Pre', 'Max_STD_Raw_Pre', 'Avg_STD_Raw_Pre',
                                                   'W0_std', 'W1_std', 'W2_std', 'W3_std', 'W4_std', 'W0_dif', 'W1_dif', 'W2_dif',
                                                   'W3_dif', 'W4_dif', 'RMSE_dif', 'MaxRSE_dif', 'Cause_no']])
            
            # arrayとして保存する
            np.save(Abnormal_Result_Tail_File_Name, Abnormal_Result_Tail_List)
        else: # textファイルがあったらtextファイルを読み込んでarrayファイルにする
            print(Abnormal_Result_Tail_File_Name, ' was created from text file')
            content =[]
            with open (Abnormal_Result_Tail_Text_File_Name, encoding = 'utf8', newline = '') as f:
                csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
                n_row = 0 #もとのデータの行数を求める変数：初期値
                for row in csvreader:
                    content = content + row #contentは1行のリストになる
                    n_row += 1 #もとのcsvファイル(リスト)の行数となる
                # print('n_row =', n_row)
                # print(content)
            # もとのデータのリスト'content'をベクトルのデータ'data'にする
            data = np.array(content)
            #ベクトルのデータ'data'をn_row行、n_column列の行列"d"にする
            Abnormal_Result_Tail_List = data.reshape(n_row, -1)
            print(Abnormal_Result_Tail_List)
            # arrayとして保存する
            np.save(Abnormal_Result_Tail_File_Name, Abnormal_Result_Tail_List)
            
            # 20240223追加　ローカルリストの削除
            del content, data
    
    else: # arrayのファイルがあるなら
        Abnormal_Result_Tail_List = np.load(Abnormal_Result_Tail_File_Name)
    
    # メモリーの開放
    gc.collect()
    
    return Abnormal_Result_Tail_List

# 202401変更
# TailモードのNormalファイルがあるかどうか確認し、なければ作る 
def Check_Tail_Normal_Result_File(Normal_Result_Tail_File_Name, Normal_Result_Tail_Text_File_Name):
    path = Normal_Result_Tail_File_Name
    is_file = os.path.isfile(path)
    if(is_file == False): # arrayのファイルがなかったら
        path = Normal_Result_Tail_Text_File_Name # textファイルを探す
        is_file = os.path.isfile(path)
        if(is_file == False): # textのファイルもなかったら
            print(Normal_Result_Tail_File_Name, ' was created')
            Normal_Result_Tail_List = np.array([['Date_Range_DIF', 'Date_Range_STD', 'Abort_Timing', 'Record Name', 'RMSE_cal', 
                                                 'RMSE_std','Beam_Max', 'MaxRSE_std', 'Max_Pre', 'Avg_Pre', 'Max_Cal', 
                                                 'Avg_Cal', 'Max_Raw_Pre', 'Avg_Raw_Pre', 'Max_STD_Raw_Pre', 'Avg_STD_Raw_Pre',
                                                 'W0_std', 'W1_std', 'W2_std', 'W3_std', 'W4_std', 'W0_dif', 'W1_dif', 'W2_dif',
                                                 'W3_dif', 'W4_dif', 'RMSE_dif', 'MaxRSE_dif', 'Cause_no']])

            # arrayとして保存する
            np.save(Normal_Result_Tail_File_Name, Normal_Result_Tail_List)
        else: # textファイルがあったらtextファイルを読み込んでarrayファイルにする
            print(Normal_Result_Tail_File_Name, ' was created from text file')
            content =[]
            with open (Normal_Result_Tail_Text_File_Name, encoding = 'utf8', newline = '') as f:
                csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
                n_row = 0 #もとのデータの行数を求める変数：初期値
                for row in csvreader:
                    content = content + row #contentは1行のリストになる
                    n_row += 1 #もとのcsvファイル(リスト)の行数となる
                # print('n_row =', n_row)
                # print(content)
            # もとのデータのリスト'content'をベクトルのデータ'data'にする
            data = np.array(content)
            #ベクトルのデータ'data'をn_row行、n_column列の行列"d"にする
            Normal_Result_Tail_List = data.reshape(n_row,-1)
            print(Normal_Result_Tail_List)
            # arrayとして保存する
            np.save(Normal_Result_Tail_File_Name, Normal_Result_Tail_List)
            
            # 20240223追加　ローカルリストの削除
            del content, data
            
    else: # arrayのファイルがあるなら
        Normal_Result_Tail_List = np.load(Normal_Result_Tail_File_Name)

    # メモリーの開放
    gc.collect()
    
    return Normal_Result_Tail_List

# ==============================================================================================================

# Strgモードの選別用解析結果があるかどうか確認し、なければtextファイルから  
def Check_Strg_Class_Result_File(Strg_Class_Result_File_Name, Strg_Class_Result_Text_File_Name):
    path = Strg_Class_Result_File_Name
    is_file = os.path.isfile(path)
    if(is_file == False): # arrayのファイルがなかったら
        path = Strg_Class_Result_Text_File_Name # textファイルを探す
        is_file = os.path.isfile(path)
        if(is_file == False): # csvのファイルもなかったら
            print(Strg_Class_Result_File_Name, ' does not exist.')
            Strg_Class_Result_List = np.empty((0,4))
        else: # textファイルがあったらtextファイルを読み込んでarrayファイルにする
            print(Strg_Class_Result_File_Name, ' was created from text file')
            content =[]
            with open (Strg_Class_Result_Text_File_Name, encoding = 'utf8', newline = '') as f:
                csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
                for row in csvreader:
                    content = content + row #contentは1行のリストになる
            # もとのデータのリスト'content'をベクトルのデータ'data'にする
            data = np.array(content)
            Strg_Class_Result_List = data
            print(Strg_Class_Result_List)
            # arrayとして保存する
            np.save(Strg_Class_Result_File_Name, Strg_Class_Result_List)
    else: # arrayのファイルがあるなら
        Strg_Class_Result_List = np.load(Strg_Class_Result_File_Name)

    return Strg_Class_Result_List

# ==============================================================================================================

# Tailモードの選別用解析結果があるかどうか確認し、なければtextファイルから  
def Check_Tail_Class_Result_File(Tail_Class_Result_File_Name, Tail_Class_Result_Text_File_Name):
    path = Tail_Class_Result_File_Name
    is_file = os.path.isfile(path)
    if(is_file == False): # arrayのファイルがなかったら
        path = Tail_Class_Result_Text_File_Name # textファイルを探す
        is_file = os.path.isfile(path)
        if(is_file == False): # csvのファイルもなかったら
            print(Tail_Class_Result_File_Name, ' does not exist.')
            Tail_Class_Result_List = np.empty((0,4))
        else: # textファイルがあったらtextファイルを読み込んでarrayファイルにする
            print(Tail_Class_Result_File_Name, ' was created from text file')
            content =[]
            with open (Tail_Class_Result_Text_File_Name, encoding = 'utf8', newline = '') as f:
                csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
                for row in csvreader:
                    content = content + row #contentは1行のリストになる
            # もとのデータのリスト'content'をベクトルのデータ'data'にする
            data = np.array(content)
            Tail_Class_Result_List = data
            print(Tail_Class_Result_List)
            # arrayとして保存する
            np.save(Tail_Class_Result_File_Name, Tail_Class_Result_List)
    else: # arrayのファイルがあるなら
        Tail_Class_Result_List = np.load(Tail_Class_Result_File_Name)

    return Tail_Class_Result_List

# ==============================================================================================================

def Get_Fit_STD_Strg(List_Para, Method, Ref_Pd, CCG_n):
# 各側室の各レコードについて、蓄積中の圧力に対して、最小二乗誤差法でフィットする回帰曲線を計算する。
# MethodはFNN

    # エラーフラグリセット
    Error_f = 'none'
    
    # モード：ビーム蓄積時
    Mode_Para = 'STD_Strg'
    
    # リング名:LERかHER
    Ring_Name = List_Para[0][:3]
    
    # データの時間間隔
    DT_Para = 'd60'
    
    # ビーム電流とバンチ数のレコード
    if(Ring_Name == 'LER'):
        BEAMCURRENT = 'BMLDCCT:CURRENT'
        BUNCHNO = 'CGLINJ:BKSEL:NOB_SET'
    elif(Ring_Name == 'HER'):
        BEAMCURRENT = 'BMHDCCT:CURRENT'
        BUNCHNO = 'CGHINJ:BKSEL:NOB_SET'
    
    # STDのデータの時間範囲(ファイルから読み込む)
    path = Ring_Name + '_Date_Range_STD_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print('No Date_Range_STD_File')
        sys.exit()   
    f = open(Ring_Name + "_Date_Range_STD_File.txt", "r")
    Date_Range_STD = f.readline()
    f.close()
    
    # Date_Rangeの定義
    Date_Range = Date_Range_STD
    Date_Para = [Date_Range] # リストにする
    
    # 測定データがローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
        for Date_Range_Test in Date_Para:
            path = LC_Room + '_' + Mode_Para + '_Data_' + Date_Range_Test +'.txt'
            is_file = os.path.isfile(path)
            if(is_file == False):
                New_File = 1
                    
    # もし一つでも新しいファイルならデータを取ってローカルPCに保存する
    if(New_File == 1):
        # kekb-co-userに入ってデータを取ってファイルを保存する。タブ区切りのデータファイルが各側室毎にできる。
        Error_f = Get_kekbcouser(Ring_Name, Date_Para, DT_Para, Mode_Para, List_Para)
        if(Error_f != 'none'):
            tk.messagebox.showinfo('SSH Error', '    Restart again     ')
            return Date_Range_STD, Error_f
        
        # データをLocal PCに持ってくる。
        Get_SFTP(Ring_Name, Date_Para, Mode_Para, List_Para)
    
    # 解析結果がローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
            for Date_Range_Test in Date_Para:
                path = LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Result_Data_' + Date_Range_Test + '.npy'
                is_file = os.path.isfile(path)
                if(is_file == False):
                    New_File = 1

    path =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Record_Data_' + Date_Range + '.npz'
    is_file = os.path.isfile(path)
    if(is_file == False):
        New_File = 1
            
    path =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Record_Data_Sel_' + Date_Range + '.npz'
    is_file = os.path.isfile(path)
    if(is_file == False):
        New_File = 1
    
    # 解析結果が既にあればDate_Rangeを返して関数を抜ける    
    if(New_File == 0):     
        return Date_Range_STD, Error_f
    
    # Abnormal, Normalのファイルを読み込む(Method = FNN)(ビーム有り)
    Abnormal_Result_Strg_File_Name = Ring_Name + '_' + Method + '_Abnormal_Class2_Result_Strg_WB.npy'
    Abnormal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method + '_Abnormal_Class2_Result_Strg_WB.txt'
    Abnormal_Result_Strg_List = Check_Strg_Abnormal_Result_File(Abnormal_Result_Strg_File_Name, 
                                                                Abnormal_Result_Strg_Text_File_Name)
    Normal_Result_Strg_File_Name = Ring_Name + '_' + Method + '_Normal_Class2_Result_Strg_WB.npy'
    Normal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method + '_Normal_Class2_Result_Strg_WB.txt'
    Normal_Result_Strg_List = Check_Strg_Normal_Result_File(Normal_Result_Strg_File_Name, 
                                                                Normal_Result_Strg_Text_File_Name)
        
    # 20240118変更
    # ファイルの行数を計算する。
    nstdrow = int(Cal_File_Row(Date_Range, DT_Para))
    
    # 各側室についてループ
    LC_Room_n = 0
    for LC_Room in List_Para:
        # LC_Roomの番号
        LC_Room_n +=1
        
        # 読み込む込むデータのファイル名：タブ区切りのテキストファイル
        filename = LC_Room + '_' + Mode_Para + '_Data_' + Date_Range +'.txt'
        
        # テキストから読み込むデータのリストの定義
        content =[]
        with open (filename, encoding = 'utf8', newline = '') as f:
            csvreader = csv.reader(f, delimiter = '\t') # csvreaderにリストとして読み込む。タブ区切り。
            n_row = 0 #もとのデータの行数を求める変数：初期値
            for row in csvreader:
                if(n_row == 0):
                    nitem = len(row)
                list_row_filtered = filter(None, row)
                row_list = list(list_row_filtered)
                
                while(len(row_list) < nitem):
                    row_list = np.append(row_list, ['0']).tolist()
                    
                content = content + row #contentは1行のリストになる
                n_row += 1 #もとのcsvファイル(リスト)の行数となる
                # 20240118変更
                if(n_row == nstdrow + 2):
                    break
        
        #元となるデータの列数
        n_column = int(len(content)/n_row)
        
        # もとのデータのリスト'content'をベクトルのデータ'data'にする
        data = np.array(content)
        #ベクトルのデータ'data'をn_row行、n_column列の行列"d"にする
        # d = data.reshape(n_row, n_column)
        
        # 1分間隔でRef_Pd日間なら
        n_row = int(60 * 24 * Ref_Pd + 2)
        try:
            d = data.reshape(n_row, -1)
        except Exception as e:
            print(traceback.format_exc())
            print(str(e))
            Error_f = str(e)
            return Date_Range_STD, Error_f
                
        # 時刻、ビーム電流、バンチ数、HOM Power、圧力のベクトルを作る
        Title_Name = d[0] # タイトルの名前の1次元ベクトル
        n_row = d.shape[0]

        # タイトルの名前のブーリアンリスト：最後に'PRES'がある列
        Record_Name_b =  [bool(Title_Name[i][-4:] == 'PRES') for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：'time'がある列:時刻
        Time_Name_b =  [bool(Title_Name[i] == 'time') for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：'BEAMCURRENTがある列：ビーム電流
        Beam_Name_b =  [bool(Title_Name[i] == BEAMCURRENT) for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：BUNCHNOがある列：バンチ数
        Nb_Name_b =  [bool(Title_Name[i] == BUNCHNO) for i in range(len(Title_Name))]

        # タイトルの名前の列をブーリアンで抜き出す：最後に'PRES'がある列
        Record_Data = d[:, Record_Name_b]
        # レコードの数 = 列の数
        N_Records = len(Record_Data[0])
        # タイトルの名前の列をブーリアンで抜き出す:'time'がある列
        Time_Data = d[:, Time_Name_b]
        # タイトルの名前の列をブーリアンで抜き出す:'BEAMCURRENTがある列
        Beam_Data = d[:, Beam_Name_b]
        # タイトルの名前の列をブーリアンで抜き出す:BUNCHNOがある列
        Nb_Data = d[:, Nb_Name_b] 
        
        print('Record data generated for ' + filename)
        print('Number of records in Local Room = ', N_Records, LC_Room, CCG_n[LC_Room_n-1] - 1)
        
        # if(N_Records < CCG_n[LC_Room_n-1] - 1):
        if(N_Records < 1 + 1 + 1):
            Error_f = 'Too few record number for ' + LC_Room
            return Date_Range_STD, Error_f
        
        # 各側室の元となるレコードデータの配列を保存する
        Record_Data_File_Name =  LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Record_Data_' + Date_Range + '.npy'
        np.save(Record_Data_File_Name, Record_Data)

        # HOM power(Beam*Beam/Nb)をつくる
        # ビーム電流とバンチ数の各要素をfloatにする:1番目のデータ(2個目のデータ)以降のベクトルにして
        Beam_Data_f =np.array([float(s) for s in np.delete(Beam_Data, 0)])
        Nb_Data_f = np.array([float(s) for s in np.delete(Nb_Data, 0)])

        # データ取得期間のビーム電流の最大値
        Beam_Max = np.max(Beam_Data_f)
        # print('Max of beam current = ', '{:.3f}'.format(Beam_Max))
        # データ取得期間のバンチ数の最大値(参考)
        Nb_Data_max = np. max(Nb_Data_f)

        # HOM power(Beam*Beam/Nb)^2をつくる:値のみでの計算する
        HOM_Data_f = (Beam_Data_f * Beam_Data_f / Nb_Data_f) ** 2

        #その列の先頭に要素'(I*I/Nb)^2'(名前)を追加する:さらに、縦ベクトルにする
        HOM_Data = np.append('(I*I/Nb)^2', HOM_Data_f).reshape(n_row, 1)

        # 解析開始　-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- ---- -- -- -- 
        # 回帰曲線用解析結果をまとめる表の最初の行
        Result_Data = np.array(['Record Name', 'W0(Beam current)', 'W1(HOM^2)', 'W2(Base)', 'RMSE', 'MaxRSE'])

        # 各レコードについて計算する
        for k in range(N_Records):

            # 列の名前を最初の行に入れる
            Time_Data_Sel = Time_Data[0]
            Record_Data_Sel = Record_Data[0, k]
            Beam_Data_Sel = Beam_Data[0]
            HOM_Data_Sel = HOM_Data[0]
            Nb_Data_Sel = Nb_Data[0]

            # 解析のために条件に合うデータを選ぶ
            n_row_s = 0 # 選んだデータの表の行数を数える
            for i in range(1, n_row):
                # 圧力値(読み値)が1.E-8 P以上
                # ビーム電流が最大値の30%以上
                # ビーム電流が最大値の95%以下
                # バンチ数が10以上
                if (float(Record_Data[i, k]) >= 1.E-8) and (float(Beam_Data[i]) >= (Beam_Max * 0.3)) and \
                                                    (float(Beam_Data[i]) <= (Beam_Max * 0.95)) and (float(Nb_Data[i]) >= 10.):
                    Time_Data_Sel = np.vstack((Time_Data_Sel, Time_Data[i]))
                    Record_Data_Sel = np.vstack((Record_Data_Sel, Record_Data[i, k]))
                    Beam_Data_Sel = np.vstack((Beam_Data_Sel, Beam_Data[i]))
                    HOM_Data_Sel = np.vstack((HOM_Data_Sel, HOM_Data[i]))
                    Nb_Data_Sel = np.vstack((Nb_Data_Sel, Nb_Data[i]))
                    n_row_s += 1 # 選んだデータの表の行数
            
            # レコードのデータを結合していく:列を足していく
            if(k == 0):
                Record_Data_Sel_List = Record_Data_Sel
            else:
                Record_Data_Sel_List = np.append(Record_Data_Sel_List, Record_Data_Sel, axis = 1)
                    
            # 1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
            Record_Data_Sel_f = np.array([float(s) for s in np.delete(Record_Data_Sel, 0)])
            Beam_Data_Sel_f = np.array([float(s) for s in np.delete(Beam_Data_Sel, 0)])
            HOM_Data_Sel_f = np.array([float(s) for s in np.delete(HOM_Data_Sel, 0)])
            # 圧力読み値(Record_Data_Sel_f)を3倍する
            Record_Data_Sel_f3 = Record_Data_Sel_f * 3.
    
            # mseを最小にする係数を求める。回帰曲線を求める。
            if n_row_s < 6: # 該当する値の行が10未満だったらベース圧の直線にする。
                W = [0., 0., 3.0e-8]
                mse = 0.
                maxse = 0.
            else:
                # まず、解析解を求める
                W = fit_plane(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3)
        
                # 負の係数があれば制限付きの勾配法を用いる
                if (W[0] < 0.) or (W[1] < 0.) or (W[2] < 1.e-9):
                    w_1 =[W[0], W[1], W[2]] # 初期値
                    W, dMSE, Tau = fit_plane_num(w_1, Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3) # 読み値を3倍している
                    # print('繰り返し回数 {0}'.format(Tau))
    
                # W[2](ベース)の制限
                if (W[2] < 1.e-9):
                    W[2] = 1.e-9   
                
                # 平均二乗誤差と最大二乗誤差の計算
                mse, maxse = mse_plane(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3, W)

            # 各レコードの2次元回帰曲線のパラメータと標準偏差を解析結果をまとめる表に追加する
            Result_Data = np.vstack((Result_Data, np.array([Record_Data[0, k], '{:.3e}'.format(W[0]), '{:.3e}'.format(W[1]), '{:.3e}'.format(W[2]), 
                                                    '{:.3e}'.format(np.sqrt(mse)), '{:.3e}'.format(np.sqrt(maxse))])))
        
        # 20240224追加　ローカルリストの削除
        del Record_Data
        gc.collect()
        
        #各側室でまとめる
        # 各側室の解析に使用したレコードデータの表を保存する
        Record_Data_File_Name_Sel =  LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Record_Data_Sel_' + Date_Range + '.npy'
        np.save(Record_Data_File_Name_Sel, Record_Data_Sel_List)

        # 各側室の解析結果をまとめた表を保存する
        Result_Data_File_Name =  LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Result_Data_' + Date_Range + '.npy'
        np.save(Result_Data_File_Name, Result_Data)
        
        # 各側室の解析結果をまとめてリスト型辞書にする
        Keys = Result_Data[0].tolist()
        Datas = Result_Data[1:, :].tolist()
        # リスト型辞書にする
        Result_Data_Dict = [dict(zip(Keys, item)) for item in Datas]
        
        # 各側室のリスト型辞書をファイルに保存する
        Resut_Data_File_Name = LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Result_Data_' + Date_Range + '_Dict' + '.npy'
        np.save(Resut_Data_File_Name, Result_Data_Dict)
        
        # 20240221追加　ローカルリストの削除
        del content, row_list, data, d, Title_Name
        del Record_Name_b, Time_Name_b, Beam_Name_b, Nb_Name_b
        del N_Records, Nb_Data, Beam_Data_f, Nb_Data_f,HOM_Data_f
        del Result_Data, Time_Data_Sel, Record_Data_Sel, Nb_Data_Sel, Record_Data_Sel_List
        del Record_Data_Sel_f, Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3
        del W, Keys, Datas, Result_Data_Dict
    
    # リングでまとめる
    # リングのビーム電流、時刻、HOMの配列を保存する:すべての側室で共通
    Beam_File_Name =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Beam_' + Date_Range + '.npy'
    np.save(Beam_File_Name, Beam_Data)
        
    Time_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Time_' + Date_Range + '.npy'
    np.save(Time_File_Name, Time_Data)
    
    HOM_File_Name =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_HOM_' + Date_Range + '.npy'
    np.save(HOM_File_Name, HOM_Data)
    
    # リングの解析に使用したビーム電流、HOMの配列を保存する:すべての側室で共通
    Beam_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Beam_Sel_' + Date_Range + '.npy'
    np.save(Beam_File_Name_Sel, Beam_Data_Sel)
    
    HOM_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_HOM_Sel_' + Date_Range + '.npy'
    np.save(HOM_File_Name_Sel, HOM_Data_Sel)
    
    # リングのすべての側室の解析結果をまとめる
    # リングの時刻、ビーム電流、HOMデータをロードする
    Beam_File_Name =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Beam_' + Date_Range + '.npy'
    Beam_Data = np.load(Beam_File_Name)
        
    Time_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Time_' + Date_Range + '.npy'
    Time_Data = np.load(Time_File_Name)
    
    HOM_File_Name =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_HOM_' + Date_Range + '.npy'
    HOM_Data = np.load(HOM_File_Name)
    
    # リングの時刻、ビーム電流、HOMデータをまとめる
    All_Data = np.append(Time_Data, Beam_Data, axis=1)
    All_Data = np.append(All_Data, HOM_Data, axis=1)
    
    # リングの解析に使用したビーム電流、HOMデータロードする
    Beam_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Beam_Sel_' + Date_Range + '.npy'
    Beam_Data_Sel = np.load(Beam_File_Name_Sel)
    
    HOM_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_HOM_Sel_' + Date_Range + '.npy'
    HOM_Data_Sel = np.load(HOM_File_Name_Sel)
    
    # リングの解析に使用したビーム電流、HOMデータをまとめる
    All_Data_Sel = np.append(Beam_Data_Sel, HOM_Data_Sel, axis=1)
    
    # リングのすべてのデータをまとめる
    # 各側室のレコードデータの配列を読み出して、
    for LC_Room in List_Para:
        # 各側室のレコードデータの配列をロードする
        Record_Data_File_Name =  LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Record_Data_' + Date_Range + '.npy'
        Record_Data_List = np.load(Record_Data_File_Name)
        
        # 各側室の解析に使用したレコードデータの配列を読み出す
        Record_Data_File_Name_Sel =  LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Record_Data_Sel_' + Date_Range + '.npy'
        Record_Data_List_Sel = np.load(Record_Data_File_Name_Sel)
        
        # リングの時刻、ビーム電流、HOMデータにレコードデータを追加する
        All_Data = np.append(All_Data, Record_Data_List, axis=1)
        All_Data_Sel = np.append(All_Data_Sel, Record_Data_List_Sel, axis=1)
        
        # 20240224追加
        del Record_Data_List, Record_Data_List_Sel
        gc.collect()
    
    # リングのすべてのデータを行列として保存する。
    All_File_Name =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Record_Data_' + Date_Range + '.npz'
    np.savez_compressed(All_File_Name, All_Data)
    # print('All Record Data was saved to ' + Ring_Name + '_' + Mode_Para + '_' + Method + 
    #       '_Class2_All_Record_Data_' + Date_Range + '.npz' )
    
    # リングの解析に使用したすべてのデータを行列として保存する。
    All_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Record_Data_Sel_' + Date_Range + '.npz'
    np.savez_compressed(All_File_Name_Sel, All_Data_Sel)
    # print('All Selected Record Data was saved to ' + Ring_Name + '_' + Mode_Para + '_' + Method + 
    #       '_Class2_All_Record_Data_Sel_' + Date_Range + '.npz' )
    
    # リングの解析結果を辞書にして保存する
    # 各側室の解析結果の辞書を読み出してまとめる
    All_Result_Dict = []
    for LC_Room in List_Para:
        Resut_Data_File_Name = LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Result_Data_' + Date_Range + '_Dict' + '.npy'
        Result_Ar = np.load(Resut_Data_File_Name, allow_pickle = 'TRUE')
        Result_Dict = Result_Ar.tolist()
        All_Result_Dict = All_Result_Dict + Result_Dict
    
    # まとめたリングの解析結果を辞書として保存する。
    Result_Data_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Result_Data_' + Date_Range + '_Dict'
    # np.save(Result_Data_File_Name, All_Result_Dict)
    np.savez_compressed(Result_Data_File_Name, All_Result_Dict)

    print('All Result Data was saved to ' + Result_Data_File_Name )

    # 20240221追加　ローカルリストの削除
    del Abnormal_Result_Strg_List,  Normal_Result_Strg_List
    del Beam_Data, Time_Data, HOM_Data, Beam_Data_Sel, HOM_Data_Sel
    del All_Data, All_Data_Sel
    del Result_Dict, All_Result_Dict
    
    # メモリーの開放
    gc.collect()
    
    # Date_Rangeを返す
    return Date_Range_STD, Error_f

# ==============================================================================================================

def Get_Fit_STD_Strg_NB(List_Para, Method, Ref_Pd, CCG_n):
# 各側室の各レコードについて、ビーム電流ゼロの時の圧力変化を最小二乗誤差法でフィットする回帰曲線を計算
# ビームはなし

    # エラーフラグのリセット
    Error_f = 'none'
    
    # モード：ビーム蓄積時
    Mode_Para = 'STD_Strg'
    
    # リング名:LERかHER
    Ring_Name = List_Para[0][:3]
    
    # データの時間間隔
    DT_Para = 'd60'
    
    # ビーム電流とバンチ数のレコード
    if(Ring_Name == 'LER'):
        BEAMCURRENT = 'BMLDCCT:CURRENT'
        BUNCHNO = 'CGLINJ:BKSEL:NOB_SET'
    elif(Ring_Name == 'HER'):
        BEAMCURRENT = 'BMHDCCT:CURRENT'
        BUNCHNO = 'CGHINJ:BKSEL:NOB_SET'

    # STDのデータの時間範囲(ファイルから読み込む)
    path = Ring_Name + '_Date_Range_STD_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print('No Date_Range_STD_File')
        sys.exit()   
    f = open(Ring_Name + "_Date_Range_STD_File.txt", "r")
    Date_Range_STD = f.readline()
    f.close()
    
    # Date_Rangeの定義
    Date_Range =Date_Range_STD
    Date_Para = [Date_Range] # リストにする
    
    # 測定データがローカルPCに既にあるかどうか確認する(NBに依らず)
    New_File = 0
    for LC_Room in List_Para:
        for Date_Range_Test in Date_Para:
            path = LC_Room + '_' + Mode_Para + '_Data_' + Date_Range_Test +'.txt'
            is_file = os.path.isfile(path)
            if(is_file == False):
                New_File = 1
                    
    # もし一つでも新しいファイルならデータを取ってローカルPCに保存する
    if(New_File == 1):
        # kekb-co-userに入ってデータを取ってファイルを保存する。タブ区切りのデータファイルが各側室毎にできる。
        Error_f = Get_kekbcouser(Ring_Name, Date_Para, DT_Para, Mode_Para, List_Para)
        if(Error_f != 'none'):
            tk.messagebox.showinfo('SSH Error', '    Restart again     ')
            return Date_Range_STD, Error_f
        
        # データをLocal PCに持ってくる。
        Get_SFTP(Ring_Name, Date_Para, Mode_Para, List_Para)
    
    # 解析結果がローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
            for Date_Range_Test in Date_Para:
                path = LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Result_Data_' + Date_Range_Test + '_NB.npy'
                is_file = os.path.isfile(path)
                if(is_file == False):
                    New_File = 1
                    
    # データがローカルPCにあるかどうか確認する。(NBに依らず）
    path =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Record_Data_' + Date_Range + '.npz'
    is_file = os.path.isfile(path)
    if(is_file == False):
        New_File = 1
    
    # 解析に使用したデータがローカルPCにあるかどうか確認する。
    path =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Record_Data_Sel_' + Date_Range + '_NB.npz'
    is_file = os.path.isfile(path)
    if(is_file == False):
        New_File = 1
    
    # 解析結果が既にあればDate_Rangeを返して関数を抜ける    
    if(New_File == 0):     
        return Date_Range_STD, Error_f
    
    # Abnormal, Normalのファイルを読み込む（Method = FNN)(ビーム無し)
    Abnormal_Result_Strg_File_Name = Ring_Name + '_' + Method + '_Abnormal_Class2_Result_Strg_NB.npy'
    Abnormal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method + '_Abnormal_Class2_Result_Strg_NB.txt'
    Abnormal_Result_Strg_List = Check_Strg_Abnormal_Result_File(Abnormal_Result_Strg_File_Name, 
                                                                Abnormal_Result_Strg_Text_File_Name)
    Normal_Result_Strg_File_Name = Ring_Name + '_' + Method + '_Normal_Class2_Result_Strg_NB.npy'
    Normal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method + '_Normal_Class2_Result_Strg_NB.txt'
    Normal_Result_Strg_List = Check_Strg_Normal_Result_File(Normal_Result_Strg_File_Name, 
                                                                Normal_Result_Strg_Text_File_Name)
        
    # 20240118変更
    # ファイルの行数を計算する。
    nstdrow = int(Cal_File_Row(Date_Range, DT_Para))
    
    # 各側室についてループ
    LC_Room_n = 0
    for LC_Room in List_Para:
        # LC_Roomの番号
        LC_Room_n +=1
        
        # 読み込む込むデータのファイル名：タブ区切りのテキストファイル(NBに依らず)
        filename = LC_Room + '_' + Mode_Para + '_Data_' + Date_Range +'.txt'
        
        # テキストから読み込むデータのリストの定義
        content =[]
        with open (filename, encoding = 'utf8', newline = '') as f:
            csvreader = csv.reader(f, delimiter = '\t') # csvreaderにリストとして読み込む。タブ区切り。
            n_row = 0 #もとのデータの行数を求める変数：初期値
            for row in csvreader:
                if(n_row == 0):
                    nitem = len(row)
                    
                list_row_filtered = filter(None, row)
                row_list = list(list_row_filtered)
                
                while(len(row_list) < nitem):
                    row_list = np.append(row_list, ['0']).tolist()
                
                content = content + row_list #contentは1行のリストになる
                n_row += 1 #もとのcsvファイル(リスト)の行数となる
                # 20240118変更
                if(n_row == nstdrow + 2):
                    break
        
        #元となるデータの列数
        n_column = int(len(content)/n_row)
        
        # もとのデータのリスト'content'をベクトルのデータ'data'にする
        data = np.array(content)
        
        # 1分間隔でRef_Pd日間なら
        # n_row = int(60 * 24 * Ref_Pd + 2)
        
        try:
            d = data.reshape(n_row, -1)
        except Exception as e:
            print(traceback.format_exc())
            print(str(e))
            Error_f = str(e)
            return Date_Range_STD, Error_f
        
        # 時刻、ビーム電流、バンチ数、HOM Power、圧力のベクトルを作る
        Title_Name = d[0] # タイトルの名前の1次元ベクトル
        n_row = d.shape[0]

        # タイトルの名前のブーリアンリスト：最後に'PRES'がある列
        Record_Name_b =  [bool(Title_Name[i][-4:] == 'PRES') for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：'time'がある列:時刻
        Time_Name_b =  [bool(Title_Name[i] == 'time') for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：'BEAMCURRENTがある列：ビーム電流
        Beam_Name_b =  [bool(Title_Name[i] == BEAMCURRENT) for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：BUNCHNOがある列：バンチ数
        Nb_Name_b =  [bool(Title_Name[i] == BUNCHNO) for i in range(len(Title_Name))]

        # タイトルの名前の列をブーリアンで抜き出す：最後に'PRES'がある列
        Record_Data = d[:, Record_Name_b]
        # レコードの数 = 列の数
        N_Records = len(Record_Data[0])
        # タイトルの名前の列をブーリアンで抜き出す:'time'がある列
        Time_Data = d[:, Time_Name_b]
        # タイトルの名前の列をブーリアンで抜き出す:'BEAMCURRENTがある列
        Beam_Data = d[:, Beam_Name_b]
        # タイトルの名前の列をブーリアンで抜き出す:BUNCHNOがある列
        Nb_Data = d[:, Nb_Name_b] 
        
        print('Record data generated for ' + filename)
        print('Number of records in Local Room = ', N_Records, LC_Room, CCG_n[LC_Room_n-1] - 1)
        
        # if(N_Records < CCG_n[LC_Room_n-1] - 1):
        if(N_Records < 1 + 1 + 1):
            Error_f = 'Too few record number for ' + LC_Room
            return Date_Range_STD, Error_f
        
        # 各側室の元となるレコードデータの配列を保存する(NBに依らず)
        Record_Data_File_Name =  LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Record_Data_' + Date_Range + '.npy'
        np.save(Record_Data_File_Name, Record_Data)

        # HOM power(Beam*Beam/Nb)をつくる
        # ビーム電流とバンチ数の各要素をfloatにする:1番目のデータ(2個目のデータ)以降のベクトルにして
        Beam_Data_f =np.array([float(s) for s in np.delete(Beam_Data, 0)])
        Nb_Data_f = np.array([float(s) for s in np.delete(Nb_Data, 0)])

        # データ取得期間のビーム電流の最大値:ビームはなし
        Beam_Max = np.max(Beam_Data_f)

        # データ取得期間のバンチ数の最大値(参考)
        Nb_Data_max = np. max(Nb_Data_f)

        # HOM power(Beam*Beam/Nb)^2をつくる:バンチ数が不定なのでただのBeam*Beamにする
        HOM_Data_f = (Beam_Data_f * Beam_Data_f)**2

        #その列の先頭に要素'(I*I/Nb)^2'(名前)を追加する:さらに、縦ベクトルにする
        HOM_Data = np.append('(I*I/Nb)^2', HOM_Data_f).reshape(n_row, 1)

        # 解析開始　-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- ---- -- -- -- 
        # 回帰曲線用解析結果をまとめる表の最初の行
        Result_Data = np.array(['Record Name', 'W0(Beam current)', 'W1(HOM^2)', 'W2(Base)', 'RMSE', 'MaxRSE'])

        # 各レコードについて計算する
        for k in range(N_Records):

            # 列の名前を最初の行に入れる
            Time_Data_Sel_a = Time_Data[0]
            Record_Data_Sel_a = Record_Data[0, k]
            Beam_Data_Sel_a = Beam_Data[0]
            HOM_Data_Sel_a = HOM_Data[0]
            Nb_Data_Sel_a = Nb_Data[0]

            # 解析のために条件に合うデータを選ぶ
            n_row_s = 0 # 選んだデータの表の行数を数える
            for i in range(1, n_row):
                # ビーム電流が50mA以下 # 20240206変更
                if (float(Beam_Data[i]) <= 50.):
                    Time_Data_Sel_a = np.vstack((Time_Data_Sel_a, Time_Data[i]))
                    Record_Data_Sel_a = np.vstack((Record_Data_Sel_a, Record_Data[i, k]))
                    Beam_Data_Sel_a = np.vstack((Beam_Data_Sel_a, Beam_Data[i]))
                    HOM_Data_Sel_a = np.vstack((HOM_Data_Sel_a, HOM_Data[i]))
                    Nb_Data_Sel_a = np.vstack((Nb_Data_Sel_a, Nb_Data[i]))
                    n_row_s += 1 # 選んだデータの表の行数
            
            # レコードのデータを結合していく:列を足していく
            if(k == 0):
                Record_Data_Sel_List_a = Record_Data_Sel_a
            else:
                Record_Data_Sel_List_a = np.append(Record_Data_Sel_List_a, Record_Data_Sel_a, axis = 1)
                    
            # 1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
            Record_Data_Sel_f_a = np.array([float(s) for s in np.delete(Record_Data_Sel_a, 0)])
            Beam_Data_Sel_f_a = np.array([float(s) for s in np.delete(Beam_Data_Sel_a, 0)])
            HOM_Data_Sel_f_a = np.array([float(s) for s in np.delete(HOM_Data_Sel_a, 0)])
            # 圧力読み値(Record_Data_Sel_f)を3倍する
            Record_Data_Sel_f3_a = Record_Data_Sel_f_a * 3.
            
            # 20240113変更：ビーム電流をデータの番号とする。-----------------------
            for i in range(len(Beam_Data_Sel_f_a)):
                Beam_Data_Sel_f_a[i] = float(i)
                HOM_Data_Sel_f_a[i] = float(i * i)
            for i in range(1, len(Beam_Data_Sel_a)):
                Beam_Data_Sel_a[i] = str(i - 1)
                HOM_Data_Sel_a[i] = str((i - 1) * (i - 1))
            
            # データ数(列数)をDIFとだいたい合わせる。d30で6時間の時。
            # 20240202変更
            mrow = len(Beam_Data_Sel_f_a)
            mrowe = len(Beam_Data_Sel_f_a) + 1
            Record_Data_Sel_f = np.delete(Record_Data_Sel_f_a, slice(mrow, mrowe), 0)
            Record_Data_Sel_f3 = np.delete(Record_Data_Sel_f3_a, slice(mrow, mrowe), 0)
            Record_Data_Sel = np.delete(Record_Data_Sel_a, slice(mrow, mrowe), 0)
            Beam_Data_Sel_f = np.delete(Beam_Data_Sel_f_a, slice(mrow, mrowe), 0)
            Beam_Data_Sel = np.delete(Beam_Data_Sel_a, slice(mrow, mrowe), 0)
            HOM_Data_Sel_f = np.delete(HOM_Data_Sel_f_a, slice(mrow, mrowe), 0)
            HOM_Data_Sel = np.delete(HOM_Data_Sel_a, slice(mrow, mrowe), 0)
            Time_Data_Sel = np.delete(Time_Data_Sel_a, slice(mrow, mrowe), 0)
            Nb_Data_Sel = np.delete(Nb_Data_Sel_a, slice(mrow, mrowe), 0)
            Record_Data_Sel_List = np.delete(Record_Data_Sel_List_a, slice(mrow, mrowe), 0)

            # print(1511, 'Record_Data_Sel (STD)', Record_Data_Sel)
            # print(1512, 'Beam_Data_Sel (STD)', Beam_Data_Sel)
            # sys.exit()
            
            if(n_row_s < 6): # 該当する値の行が10未満だったらベース圧の直線にする。
                W = [0., 0., 3.0e-8] # w2はベース圧力
                mse = 0.
                maxse = 0.
            else:
                # 20240325変更　2次元の解析解にする
                # 解析解を求める
                # 20240330
                # W = fit_plane(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3)
                W = fit_line(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3)
                # W = [0., 0., 3.0e-8]
                # W[2] = np.average(Record_Data_Sel_f3)
                
                # 負の係数があれば制限付きの勾配法を用いる
                # if (W[0] < 0.) or (W[1] < 0.) or (W[2] < 1.e-9):
                #    w_1 =[W[0], W[1], W[2]] # 初期値
                #    W, dMSE, Tau = fit_plane_num(w_1, Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3)
                #    print('繰り返し回数 {0}'.format(Tau))
                
                # if (W[2] < 3.e-9):
                #     W[2] = 3.e-9
    
                # 平均二乗誤差と最大二乗誤差の計算
                # 20240330
                # mse, maxse = mse_plane(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3, W)
                mse, maxse = mse_line(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3, W)
            
            # 圧力の平均を求める
            # record_ave = np.average(Record_Data_Sel_f3)
            # W = [0., 0., record_ave] # w2は平均値(読み値を3倍している)
            
            # RMSを求める
            # record_rms = 0
            # for i in range(len(Record_Data_Sel_f3) - 1):
            #     record_rms = record_rms + (Record_Data_Sel_f3[i] - record_ave)**2
            # record_rms = np.sqrt(record_rms / (len(Record_Data_Sel_f3)))

            # 各レコードの結果をまとめる表に追加する
            Result_Data = np.vstack((Result_Data, np.array([Record_Data[0, k], '{:.3e}'.format(W[0]), '{:.3e}'.format(W[1]), 
                                                            '{:.3e}'.format(W[2]), '{:.3e}'.format(np.sqrt(mse)), 
                                                            '{:.3e}'.format(np.sqrt(maxse))])))
        
        # 20240224追加
        del Record_Data
        gc.collect()
        
        #各側室でまとめる
        # 各側室の解析に使用したレコードデータの表を保存する
        Record_Data_File_Name_Sel =  LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Record_Data_Sel_' + Date_Range + '_NB.npy'
        np.save(Record_Data_File_Name_Sel, Record_Data_Sel_List)

        # 各側室の解析結果をまとめた表を保存する
        Result_Data_File_Name =  LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Result_Data_' + Date_Range + '_NB.npy'
        np.save(Result_Data_File_Name, Result_Data)
        
        # 各側室の解析結果をまとめてリスト型辞書にする
        Keys = Result_Data[0].tolist()
        Datas = Result_Data[1:, :].tolist()
        # リスト型辞書にする
        Result_Data_Dict = [dict(zip(Keys, item)) for item in Datas]
        
        # 各側室のリスト型辞書をファイルに保存する
        Resut_Data_File_Name = LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Result_Data_' + Date_Range + '_Dict' + '_NB.npy'
        np.save(Resut_Data_File_Name, Result_Data_Dict)
        
        # 20240221追加　ローカルリストの削除
        del content, row_list, data, d, Title_Name
        del Record_Name_b, Time_Name_b, Beam_Name_b, Nb_Name_b
        del N_Records, Nb_Data
        del Beam_Data_f, Nb_Data_f,HOM_Data_f
        del Result_Data, Time_Data_Sel_a, Record_Data_Sel_a, Beam_Data_Sel_a
        del HOM_Data_Sel_a, Nb_Data_Sel_a, Record_Data_Sel_List_a
        del Record_Data_Sel_f_a, Beam_Data_Sel_f_a, HOM_Data_Sel_f_a, Record_Data_Sel_f3_a
        del Record_Data_Sel_f, Record_Data_Sel_f3, Record_Data_Sel, Beam_Data_Sel_f
        del HOM_Data_Sel_f, Time_Data_Sel, Nb_Data_Sel, Record_Data_Sel_List
        del W, Keys, Datas, Result_Data_Dict
        
    # リングでまとめる
    # リングのビーム電流、時刻、HOMの配列を保存する:すべての側室で共通
    Beam_File_Name =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Beam_' + Date_Range + '.npy'
    np.save(Beam_File_Name, Beam_Data)
        
    Time_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Time_' + Date_Range + '.npy'
    np.save(Time_File_Name, Time_Data)
    
    HOM_File_Name =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_HOM_' + Date_Range + '.npy'
    np.save(HOM_File_Name, HOM_Data)
    
    # リングの解析に使用したビーム電流、HOMの配列を保存する:すべての側室で共通
    Beam_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Beam_Sel_' + Date_Range + '_NB.npy'
    np.save(Beam_File_Name_Sel, Beam_Data_Sel)
    
    HOM_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_HOM_Sel_' + Date_Range + '_NB.npy'
    np.save(HOM_File_Name_Sel, HOM_Data_Sel)
    
    # リングのすべての側室の解析結果をまとめる
    # リングの時刻、ビーム電流、HOMデータをロードする
    Beam_File_Name =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Beam_' + Date_Range + '.npy'
    Beam_Data = np.load(Beam_File_Name)
        
    Time_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Time_' + Date_Range + '.npy'
    Time_Data = np.load(Time_File_Name)
    
    HOM_File_Name =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_HOM_' + Date_Range + '.npy'
    HOM_Data = np.load(HOM_File_Name)
    
    # リングの時刻、ビーム電流、HOMデータをまとめる
    All_Data = np.append(Time_Data, Beam_Data, axis=1)
    All_Data = np.append(All_Data, HOM_Data, axis=1)
    
    # リングの解析に使用したビーム電流、HOMデータロードする
    Beam_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Beam_Sel_' + Date_Range + '_NB.npy'
    Beam_Data_Sel = np.load(Beam_File_Name_Sel)
    
    HOM_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_HOM_Sel_' + Date_Range + '_NB.npy'
    HOM_Data_Sel = np.load(HOM_File_Name_Sel)
    
    # リングの解析に使用したビーム電流、HOMデータをまとめる
    All_Data_Sel = np.append(Beam_Data_Sel, HOM_Data_Sel, axis=1)
    
    # リングのすべてのデータをまとめる
    # 各側室のレコードデータの配列を読み出して、
    for LC_Room in List_Para:
        # 各側室のレコードデータの配列をロードする(NBに依らず)
        Record_Data_File_Name =  LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Record_Data_' + Date_Range + '.npy'
        Record_Data_List = np.load(Record_Data_File_Name)
        
        # 各側室の解析に使用したレコードデータの配列を読み出す
        Record_Data_File_Name_Sel =  LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Record_Data_Sel_' + Date_Range + '_NB.npy'
        Record_Data_List_Sel = np.load(Record_Data_File_Name_Sel)
        
        # リングの時刻、ビーム電流、HOMデータにレコードデータを追加する
        try:
            All_Data = np.append(All_Data, Record_Data_List, axis=1)
        except Exception as e:
            print(traceback.format_exc())
            print(str(e))
            Error_f = str(e)
            return Date_Range_STD, Error_f
        
        All_Data_Sel = np.append(All_Data_Sel, Record_Data_List_Sel, axis=1)
        
        # 20240224追加
        del Record_Data_List, Record_Data_List_Sel
        gc.collect()
    
    # リングのすべてのデータを行列として保存する。(NBに依らず)
    All_File_Name =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Record_Data_' + Date_Range + '.npz'
    np.savez_compressed(All_File_Name, All_Data)
    
    # リングの解析に使用したすべてのデータを行列として保存する。
    All_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Record_Data_Sel_' + Date_Range + '_NB.npz'
    np.savez_compressed(All_File_Name_Sel, All_Data_Sel)
    
    # リングの解析結果を辞書にして保存する
    # 各側室の解析結果の辞書を読み出してまとめる
    All_Result_Dict = []
    for LC_Room in List_Para:
        Resut_Data_File_Name = LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Result_Data_' + Date_Range + '_Dict' + '_NB.npy'
        Result_Ar = np.load(Resut_Data_File_Name, allow_pickle = 'TRUE')
        Result_Dict = Result_Ar.tolist()
        All_Result_Dict = All_Result_Dict + Result_Dict
    
    # まとめたリングの解析結果を辞書として保存する。
    Result_Data_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Result_Data_' + Date_Range + '_Dict_NB'
    # np.save(Result_Data_File_Name, All_Result_Dict)
    np.savez_compressed(Result_Data_File_Name, All_Result_Dict)
    print('All Result Data was saved to ' + Result_Data_File_Name )

    # 20240221追加　ローカルリストの削除
    del Abnormal_Result_Strg_List,  Normal_Result_Strg_List
    del Beam_Data, Time_Data, HOM_Data, Beam_Data_Sel, HOM_Data_Sel
    del All_Data, All_Data_Sel
    del Result_Ar, Result_Dict, All_Result_Dict
    
    # メモリーの開放
    gc.collect()
    
    # Date_Rangeを返す
    return Date_Range_STD, Error_f

# ==============================================================================================================
# ==============================================================================================================

def Get_Fit_CHK_Strg(Method, Check_Record_Name, Mode_Para, List_Para, Abort_Timing, CCG_n):
# 各側室の各レコードについて、蓄積中の圧力に対して、最小二乗誤差法でフィットする回帰曲線を計算する。

    # エラーフラグリセット
    Error_f = 'none'
    
    Mode_Para_c = 'DIF_Strg'
    
    # リング名:LERかHER
    Ring_Name = List_Para[0][:3]
    if(Ring_Name == 'LER'):
        BEAMCURRENT = 'BMLDCCT:CURRENT'
        BUNCHNO = 'CGLINJ:BKSEL:NOB_SET'
    elif(Ring_Name == 'HER'):
        BEAMCURRENT = 'BMHDCCT:CURRENT'
        BUNCHNO = 'CGHINJ:BKSEL:NOB_SET'
    
    # データの時間間隔
    DT_Para = 'd30'
    
    # CHK_Strgのデータの時間範囲(ファイルから読み込む)
    path = Ring_Name + '_Date_Range_CHK_Strg_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print('No Date_Range_CHK_File')
        sys.exit()   
    f = open(Ring_Name + "_Date_Range_CHK_Strg_File.txt", "r")
    Date_Range_CHK = f.readline()
    f.close()
    
    # Date_Rangeの定義
    Date_Range = Date_Range_CHK
    Date_Para = [Date_Range] # リストにする
    
    Date_Range_CHK = Date_Range
    
    # 測定データがローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
            for Date_Range_Test in Date_Para:
                path = LC_Room + '_' + Mode_Para_c + '_Data_' + Date_Range_Test +'.txt' # 20230816+c
                is_file = os.path.isfile(path)
                if(is_file == False):
                    New_File = 1
                    
    # もし一つでも新しいファイルならデータを取ってローカルPCに保存する
    if(New_File == 1):
        # kekb-co-userに入ってデータを取ってファイルを保存する。タブ区切りのデータファイルが各側室毎にできる。
        Error_f = Get_kekbcouser(Ring_Name, Date_Para, DT_Para, Mode_Para_c, List_Para) # 20230816+c
        if(Error_f != 'none'):
            tk.messagebox.showinfo('SSH Error', '    Restart again     ')
            return Date_Range_CHK, Error_f
        
        # データをLocal PCに持ってくる。
        Get_SFTP(Ring_Name, Date_Para, Mode_Para_c, List_Para) # 20230816+c
    
    # 解析結果がローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
            for Date_Range_Test in Date_Para:
                path = LC_Room + '_' + Mode_Para_c + '_Result_Data_' + Date_Range_Test + '.npy' # 20230816+c
                is_file = os.path.isfile(path)
                if(is_file == False):
                    New_File = 1

    path =  Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_' + Date_Range_CHK + '.npz' # 20230816+c
    is_file = os.path.isfile(path)
    if(is_file == False):
                    New_File = 1
            
    path =  Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_Sel_' + Date_Range_CHK + '.npz' # 20230816+c
    is_file = os.path.isfile(path)
    if(is_file == False):
                    New_File = 1
    
    if(New_File == 0): # 解析結果が既にあればDate_Rangeを返して関数を抜ける
            # プロットして確かめる
            if(Check_Record_Name != "none"):   
                Plot_Record_Name = Check_Record_Name
                Date_Range_CHK_Strg = Date_Range_CHK
                Make_Plot_Strg(Method, Plot_Record_Name, Date_Range_CHK_Strg, Mode_Para_c, List_Para, Abort_Timing)
        
            return Date_Range_CHK, Error_f
  
    # 20240118変更
    # ファイルの行数を計算する。
    nchkrow = int(Cal_File_Row(Date_Range, DT_Para))
    
    # 各側室についてループ
    LC_Room_n = 0
    for LC_Room in List_Para:
        # LC_Roomの番号
        LC_Room_n +=1
        
        # 読み込む込むデータのファイル名：タブ区切りのテキストファイル
        filename = LC_Room + '_' + Mode_Para_c + '_Data_' + Date_Range +'.txt' #20230816+c
        
        # テキストから読み込むデータのリストの定義
        content =[]
        with open (filename, encoding = 'utf8', newline = '') as f:
            csvreader = csv.reader(f, delimiter = '\t') # csvreaderにリストとして読み込む。タブ区切り。
            n_row = 0 #もとのデータの行数を求める変数：初期値
            for row in csvreader:
                if(n_row == 0):
                    nitem = len(row)
        
                list_row_filtered = filter(None, row)
                row_list = list(list_row_filtered)

                while(len(row_list) < nitem):
                    row_list = np.append(row_list, ['0']).tolist()
                                
                content = content + row_list #contentは1行のリストになる
                n_row += 1 #もとのcsvファイル(リスト)の行数となる
                # 20240118変更
                if(n_row == nchkrow + 2):
                    break
        
        #元となるデータの列数
        n_column = int(len(content)/n_row)
        
        # もとのデータのリスト'content'をベクトルのデータ'data'にする
        data = np.array(content)
        #ベクトルのデータ'data'をn_row行、n_column列の行列"d"にする
        d = data.reshape(n_row, n_column)
            
        # 時刻、ビーム電流、バンチ数、HOM Power、圧力のベクトルを作る
        Title_Name = d[0] # タイトルの名前の1次元ベクトル

        # タイトルの名前のブーリアンリスト：最後に'PRES'がある列
        Record_Name_b =  [bool(Title_Name[i][-4:] == 'PRES') for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：'time'がある列:時刻
        Time_Name_b =  [bool(Title_Name[i] == 'time') for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：'BEAMCURRENTがある列：ビーム電流
        Beam_Name_b =  [bool(Title_Name[i] == BEAMCURRENT) for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：BUNCHNOがある列：バンチ数
        Nb_Name_b =  [bool(Title_Name[i] == BUNCHNO) for i in range(len(Title_Name))]

        # タイトルの名前の列をブーリアンで抜き出す：最後に'PRES'がある列
        Record_Data = d[:, Record_Name_b]
        # レコードの数 = 列の数
        N_Records = len(Record_Data[0])
        # タイトルの名前の列をブーリアンで抜き出す:'time'がある列
        Time_Data = d[:, Time_Name_b]
        # タイトルの名前の列をブーリアンで抜き出す:'BEAMCURRENTがある列
        Beam_Data = d[:, Beam_Name_b]
        # タイトルの名前の列をブーリアンで抜き出す:BUNCHNOがある列
        Nb_Data = d[:, Nb_Name_b] 
        
        print('Record data generated for ' + filename)
        print('Number of records in Local Room = ', N_Records, LC_Room, CCG_n[LC_Room_n-1] - 1)
        
        # if(N_Records < CCG_n[LC_Room_n-1] - 1):
        if(N_Records < 1 + 1 + 1):
            Error_f = 'Too few record number for ' + LC_Room
            return Date_Range_STD, Error_f
        
        # 各側室の元となるレコードデータの配列を保存する
        Record_Data_File_Name =  LC_Room + '_' + Mode_Para_c + '_Record_Data_' + Date_Range + '.npy' #20230816+c
        np.save(Record_Data_File_Name, Record_Data)

        # HOM power(Beam*Beam/Nb)をつくる
        # ビーム電流とバンチ数の各要素をfloatにする:1番目のデータ(2個目のデータ)以降のベクトルにして
        Beam_Data_f =np.array([float(s) for s in np.delete(Beam_Data, 0)])
        Nb_Data_f = np.array([float(s) for s in np.delete(Nb_Data, 0)])

        # データ取得期間のビーム電流の最大値
        Beam_Max = np.max(Beam_Data_f)
        # print('Max of beam current = ', '{:.3f}'.format(Beam_Max))
        # データ取得期間のバンチ数の最大値(参考)
        Nb_Data_max = np. max(Nb_Data_f)

        # HOM power(Beam*Beam/Nb)^2をつくる:値のみでの計算する
        HOM_Data_f = (Beam_Data_f * Beam_Data_f / Nb_Data_f) ** 2

        #その列の先頭に要素'(I*I/Nb)^2'(名前)を追加する:さらに、縦ベクトルにする
        HOM_Data = np.append('(I*I/Nb)^2', HOM_Data_f).reshape(n_row, 1)

        # 解析開始(CHK_Strg)　-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- ---- -- -- -- 
        # 回帰曲線用の解析結果をまとめる表の最初の行
        Result_Data = np.array(['Record Name', 'W0(Beam current)', 'W1(HOM^2)', 'W2(Base)', 'RMSE', 'MaxRSE'])

        # 各レコードについて計算する
        for k in range(N_Records):

            # 列の名前を最初の行に入れる
            Time_Data_Sel = Time_Data[0]
            Record_Data_Sel = Record_Data[0, k]
            Beam_Data_Sel = Beam_Data[0]
            HOM_Data_Sel = HOM_Data[0]
            Nb_Data_Sel = Nb_Data[0]

            # 解析のために条件に合うデータを選ぶ
            n_row_s = 0 # 選んだデータの表の行数を数える
            for i in range(1, n_row):
                # 圧力値(読み値)が1.E-8 P以上
                # ビーム電流が50mA以上 ->　# 20240206 削除
                # ビーム電流が最大値の99.9%以下
                # ビーム電流が最大値の5%以上 20240322追加
                # バンチ数が10以上
                if (float(Record_Data[i, k]) >= 1.E-8) and (float(Beam_Data[i]) <= (Beam_Max * 0.999)) and \
                                (float(Nb_Data[i]) >= 10.) and (float(Beam_Data[i]) >= (Beam_Max * 0.05)):
                    Time_Data_Sel = np.vstack((Time_Data_Sel, Time_Data[i]))
                    Record_Data_Sel = np.vstack((Record_Data_Sel, Record_Data[i, k]))
                    Beam_Data_Sel = np.vstack((Beam_Data_Sel, Beam_Data[i]))
                    HOM_Data_Sel = np.vstack((HOM_Data_Sel, HOM_Data[i]))
                    Nb_Data_Sel = np.vstack((Nb_Data_Sel, Nb_Data[i]))
                    n_row_s += 1 # 選んだデータの表の行数
            
            # レコードのデータを結合していく:列を足していく
            if(k == 0):
                Record_Data_Sel_List = Record_Data_Sel
            else:
                try:
                    Record_Data_Sel_List = np.append(Record_Data_Sel_List, Record_Data_Sel, axis = 1)
                except Exception as e:
                    print(traceback.format_exc())
                    print(str(e))
                    Error_f = str(e)
                    return Date_Range_CHK, Error_f
            
            # 1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
            try:
                Record_Data_Sel_f = np.array([float(s) for s in np.delete(Record_Data_Sel, 0)])
            except Exception as e:
                print(traceback.format_exc())
                print(str(e))
                Error_f = str(e)
                return Date_Range_CHK, Error_f
                
            Beam_Data_Sel_f = np.array([float(s) for s in np.delete(Beam_Data_Sel, 0)])
            HOM_Data_Sel_f = np.array([float(s) for s in np.delete(HOM_Data_Sel, 0)])
            # 圧力読み値(Record_Data_Sel_f)を3倍する
            Record_Data_Sel_f3 = Record_Data_Sel_f * 3.
    
            # mseを最小にする係数を求める。回帰曲線を求める。
            if n_row_s < 6: # 該当する値の行が6未満だったらベース圧の直線にする。
                W = [0., 0., 3.0e-8]
                mse = 0.
                maxse = 0.
            else:
                # まず、解析解を求める
                W = fit_plane(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3)
        
                # 20240320変更　DIFの時はWの符号は問わない
                # 負の係数があれば制限付きの勾配法を用いる
                if (W[0] < 0.) or (W[1] < 0.) or (W[2] < 1.e-9):
                    w_1 =[W[0], W[1], W[2]] # 初期値
                    W, dMSE, Tau = fit_plane_num(w_1, Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3) # 読み値を3倍
                    # print('繰り返し回数 {0}'.format(Tau))
                     
                # W[2]ベースの制限 202401変更
                if (W[2] < 1.e-9):
                    W[2] = 1.e-9
                
                # 平均二乗誤差と最大二乗誤差の計算
                mse, maxse = mse_plane(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3, W)

            # 各レコードの2次元回帰曲線のパラメータと標準偏差を解析結果をまとめる表に追加する
            Result_Data = np.vstack((Result_Data, np.array([Record_Data[0, k], '{:.3e}'.format(W[0]), '{:.3e}'.format(W[1]), '{:.3e}'.format(W[2]), 
                                                    '{:.3e}'.format(np.sqrt(mse)),  '{:.3e}'.format(np.sqrt(maxse))])))
        
        #各側室でまとめる
        # 各側室の解析に使用したレコードデータの表を保存する
        Record_Data_File_Name_Sel =  LC_Room + '_' + Mode_Para_c + '_Record_Data_Sel_' + Date_Range + '.npy' #20230816+c
        np.save(Record_Data_File_Name_Sel, Record_Data_Sel_List)
            
        # 各側室の解析結果をまとめたを表示する
        # print(Result_Data)

        # 各側室の解析結果をまとめた表を保存する
        Result_Data_File_Name =  LC_Room + '_' + Mode_Para_c + '_Result_Data_' + Date_Range + '.npy' #20230816+c
        np.save(Result_Data_File_Name, Result_Data)
        
        # 各側室の解析結果をまとめてリスト型辞書にする
        Keys = Result_Data[0].tolist()
        Datas = Result_Data[1:, :].tolist()
        # リスト型辞書にする
        Result_Data_Dict = [dict(zip(Keys, item)) for item in Datas]
        
        # 各側室のリスト型辞書をファイルに保存する
        Resut_Data_File_Name = LC_Room + '_' + Mode_Para_c + '_Result_Data_' + Date_Range + '_Dict' + '.npy' #20230816+c
        np.save(Resut_Data_File_Name, Result_Data_Dict)
        
        # 20240222追加　ローカルリストの削除
        del content, row_list, data, d, Title_Name, Record_Name_b, Time_Name_b, Beam_Name_b, Nb_Name_b
        del Record_Data, N_Records, Nb_Data, Beam_Data_f, Nb_Data_f, HOM_Data_f
        del Result_Data, Time_Data_Sel, Record_Data_Sel, Nb_Data_Sel, Record_Data_Sel_List
        del Record_Data_Sel_f, Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3
        del W, Keys, Datas, Result_Data_Dict
    
    # リングでまとめる
    # リングのビーム電流、時刻、HOMの配列を保存する:すべての側室で共通
    Beam_File_Name =  Ring_Name + '_' + Mode_Para_c + '_Beam_' + Date_Range + '.npy' #20230816+c
    np.save(Beam_File_Name, Beam_Data)
        
    Time_File_Name = Ring_Name + '_' + Mode_Para_c + '_Time_' + Date_Range + '.npy' #20230816+c
    np.save(Time_File_Name, Time_Data)
    
    HOM_File_Name =  Ring_Name + '_' + Mode_Para_c + '_HOM_' + Date_Range + '.npy' #20230816+c
    np.save(HOM_File_Name, HOM_Data)
    
    # リングの解析に使用したビーム電流、HOMの配列を保存する:すべての側室で共通
    Beam_File_Name_Sel =  Ring_Name + '_' + Mode_Para_c + '_Beam_Sel_' + Date_Range + '.npy' #20230816+c
    np.save(Beam_File_Name_Sel, Beam_Data_Sel)
    
    HOM_File_Name_Sel =  Ring_Name + '_' + Mode_Para_c + '_HOM_Sel_' + Date_Range + '.npy' #20230816+c
    np.save(HOM_File_Name_Sel, HOM_Data_Sel)
    
    # リングのすべての側室の解析結果をまとめる
    # リングの時刻、ビーム電流、HOMデータをロードする
    Beam_File_Name =  Ring_Name + '_' + Mode_Para_c + '_Beam_' + Date_Range + '.npy' #20230816+c
    Beam_Data = np.load(Beam_File_Name)
        
    Time_File_Name = Ring_Name + '_' + Mode_Para_c + '_Time_' + Date_Range + '.npy' #20230816+c
    Time_Data = np.load(Time_File_Name)
    
    HOM_File_Name =  Ring_Name + '_' + Mode_Para_c + '_HOM_' + Date_Range + '.npy' #20230816+c
    HOM_Data = np.load(HOM_File_Name)
    
    # リングの時刻、ビーム電流、HOMデータをまとめる
    All_Data = np.append(Time_Data, Beam_Data, axis=1)
    All_Data = np.append(All_Data, HOM_Data, axis=1)
    
    # リングの解析に使用したビーム電流、HOMデータロードする
    Beam_File_Name_Sel =  Ring_Name + '_' + Mode_Para_c + '_Beam_Sel_' + Date_Range + '.npy' #20230816+c
    Beam_Data_Sel = np.load(Beam_File_Name_Sel)
    
    HOM_File_Name_Sel =  Ring_Name + '_' + Mode_Para_c + '_HOM_Sel_' + Date_Range + '.npy' #20230816+c
    HOM_Data_Sel = np.load(HOM_File_Name_Sel)
    
    # リングの解析に使用したビーム電流、HOMデータをまとめる
    All_Data_Sel = np.append(Beam_Data_Sel, HOM_Data_Sel, axis=1)
    
    # リングのすべてのデータをまとめる
    # 各側室のレコードデータの配列を読み出して、
    for LC_Room in List_Para:
        # 各側室のレコードデータの配列をロードする
        Record_Data_File_Name =  LC_Room + '_' + Mode_Para_c + '_Record_Data_' + Date_Range + '.npy' #20230816+c
        Record_Data_List = np.load(Record_Data_File_Name)
        
        # 各側室の解析に使用したレコードデータの配列を読み出す
        Record_Data_File_Name_Sel =  LC_Room + '_' + Mode_Para_c + '_Record_Data_Sel_' + Date_Range + '.npy' #20230816+c
        Record_Data_List_Sel = np.load(Record_Data_File_Name_Sel)
        
        # リングの時刻、ビーム電流、HOMデータにレコードデータを追加する
        All_Data = np.append(All_Data, Record_Data_List, axis=1)
        All_Data_Sel = np.append(All_Data_Sel, Record_Data_List_Sel, axis=1)
    
    # リングのすべてのデータを行列として保存する。
    All_File_Name =  Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_' + Date_Range_CHK + '.npz' #20230816+c
    np.savez_compressed(All_File_Name, All_Data)
    print('All Record Data was saved to ' + Ring_Name + '_' + Mode_Para_c + 
          '_All_Record_Data_' + Date_Range_CHK + '.npz' )  #20230816+c
    
    # リングの解析に使用したすべてのデータを行列として保存する。
    All_File_Name_Sel =  Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_Sel_' + Date_Range_CHK + '.npz' #20230816+c
    np.savez_compressed(All_File_Name_Sel, All_Data_Sel)
    print('All Selected Record Data was saved to ' + Ring_Name + '_' + Mode_Para_c + 
          '_All_Record_Data_Sel_' + Date_Range_CHK + '.npz' )  #20230816+c
    
    # リングの解析結果を辞書にして保存する
    # 各側室の解析結果の辞書を読み出してまとめる
    All_Result_Dict = []
    for LC_Room in List_Para:
        Resut_Data_File_Name = LC_Room + '_' + Mode_Para_c + '_Result_Data_' + Date_Range + '_Dict' + '.npy' #20230816+c
        Result_Ar = np.load(Resut_Data_File_Name, allow_pickle = 'TRUE')
        Result_Dict = Result_Ar.tolist()
        All_Result_Dict = All_Result_Dict + Result_Dict
    
    # まとめたリングの解析結果を辞書として保存する。
    Result_Data_File_Name = Ring_Name + '_' + Mode_Para_c + '_All_Result_Data_' + Date_Range_CHK + '_Dict' #20230816+c
    np.savez_compressed(Result_Data_File_Name, All_Result_Dict)
    print('All Result Data was saved to ' + Result_Data_File_Name )
    
    # Date_Range_CHKをファイルに保存する。
    f = open('Date_Range_CHK_File.txt', "w")
    f.write(Date_Range_CHK)
    f.close()
    print('Date_Range_CHK was saved to Date_Range_CHK_File.txt')

    # プロットして確かめる---------------------------------------------------------------------
    if(Check_Record_Name != "none"):   
        Plot_Record_Name = Check_Record_Name
        Date_Range_CHK_Strg = Date_Range_CHK
        Make_Plot_Strg(Method, Plot_Record_Name, Date_Range_CHK_Strg, Mode_Para_c, List_Para, Abort_Timing)
        
    # 20240222追加　ローカルリストの削除
    del Beam_Data, Time_Data, HOM_Data, Beam_Data_Sel, HOM_Data_Sel, All_Data, All_Data_Sel
    del Record_Data_List, Record_Data_List_Sel, Result_Ar, Result_Dict, All_Result_Dict
    
    #メモリーの開放
    gc.collect()
    
    return Date_Range_CHK, Error_f # Date_Rangeを返す

# ==============================================================================================================

def Get_Fit_CHK_Strg_NB(Method, Check_Record_Name, Mode_Para, List_Para, Abort_Timing, CCG_n):
# 各側室の各レコードについて、蓄積中の圧力に対して、最小二乗誤差法でフィットする回帰曲線を計算する。
# ビームはなし

    # エラーフラグリセット
    Error_f = 'none'
    
    Mode_Para_c = 'DIF_Strg' #20230816+c
    
    # リング名:LERかHER
    Ring_Name = List_Para[0][:3]
    if(Ring_Name == 'LER'):
        BEAMCURRENT = 'BMLDCCT:CURRENT'
        BUNCHNO = 'CGLINJ:BKSEL:NOB_SET'
    elif(Ring_Name == 'HER'):
        BEAMCURRENT = 'BMHDCCT:CURRENT'
        BUNCHNO = 'CGHINJ:BKSEL:NOB_SET'
    
    # データの時間間隔
    DT_Para = 'd30'
    
    # CHK_Strgのデータの時間範囲(ファイルから読み込む)
    path = Ring_Name + '_Date_Range_CHK_Strg_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print('No Date_Range_CHK_File')
        sys.exit()   
    f = open(Ring_Name + "_Date_Range_CHK_Strg_File.txt", "r")
    Date_Range_CHK = f.readline()
    f.close()
    
    # Date_Rangeの定義
    Date_Range = Date_Range_CHK
    Date_Para = [Date_Range] # リストにする
    
    Date_Range_CHK = Date_Range
    
    # 測定データがローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
            for Date_Range_Test in Date_Para:
                path = LC_Room + '_' + Mode_Para_c + '_Data_' + Date_Range_Test +'.txt' #20230816+c
                is_file = os.path.isfile(path)
                if(is_file == False):
                    New_File = 1
                    
    # もし一つでも新しいファイルならデータを取ってローカルPCに保存する
    if(New_File == 1):
        # kekb-co-userに入ってデータを取ってファイルを保存する。タブ区切りのデータファイルが各側室毎にできる。
        Error_f = Get_kekbcouser(Ring_Name, Date_Para, DT_Para, Mode_Para, List_Para)
        if(Error_f != 'none'):
            tk.messagebox.showinfo('SSH Error', '    Restart again     ')
            return Date_Range_CHK, Error_f
        
        # データをLocal PCに持ってくる。
        Get_SFTP(Ring_Name, Date_Para, Mode_Para, List_Para)
    
    # 解析結果がローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
            for Date_Range_Test in Date_Para:
                path = LC_Room + '_' + Mode_Para_c + '_Result_Data_' + Date_Range_Test + '.npy'  #20230816+c
                is_file = os.path.isfile(path)
                if(is_file == False):
                    New_File = 1

    path =  Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_' + Date_Range_CHK + '.npz'  #20230816+c
    is_file = os.path.isfile(path)
    if(is_file == False):
                    New_File = 1
            
    path =  Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_Sel_' + Date_Range_CHK + '.npz' #20230816+c
    is_file = os.path.isfile(path)
    if(is_file == False):
                    New_File = 1
    
    if(New_File == 0): # 解析結果が既にあればDate_Rangeを返して関数を抜ける
            # プロットして確かめる
            if(Check_Record_Name != "none"):   
                Plot_Record_Name = Check_Record_Name
                Date_Range_CHK_Strg = Date_Range_CHK
                Make_Plot_Strg_NB(Method, Plot_Record_Name, Date_Range_CHK_Strg, Mode_Para_c, List_Para, Abort_Timing)
        
            return Date_Range_CHK, Error_f
  
    # 20240118変更
    # ファイルの行数を計算する。
    nchkrow = int(Cal_File_Row(Date_Range, DT_Para))
    
    # 各側室についてループ
    LC_Room_n = 0
    for LC_Room in List_Para:
        # LC_Roomの番号
        LC_Room_n +=1
        
        # 読み込む込むデータのファイル名：タブ区切りのテキストファイル
        filename = LC_Room + '_' + Mode_Para_c + '_Data_' + Date_Range +'.txt'  #20230816+c
        
        # テキストから読み込むデータのリストの定義
        content =[]
        # print('Data reading started: ' + filename)
        with open (filename, encoding = 'utf8', newline = '') as f:
            csvreader = csv.reader(f, delimiter = '\t') # csvreaderにリストとして読み込む。タブ区切り。
            n_row = 0 #もとのデータの行数を求める変数：初期値
            for row in csvreader:
                if(n_row == 0):
                    nitem = len(row)

                list_row_filtered = filter(None, row)
                row_list = list(list_row_filtered)
                
                while(len(row_list) < nitem):
                    row_list = np.append(row_list, ['0']).tolist()
                    
                content = content + row_list #contentは1行のリストになる
                n_row += 1 #もとのcsvファイル(リスト)の行数となる
                # 20240118変更
                if(n_row == nchkrow + 2):
                    break
        
        #元となるデータの列数
        n_column = int(len(content)/n_row)
        
        # もとのデータのリスト'content'をベクトルのデータ'data'にする
        data = np.array(content)
        #ベクトルのデータ'data'をn_row行、n_column列の行列"d"にする
        d = data.reshape(n_row, n_column)
        
        # 時刻、ビーム電流、バンチ数、HOM Power、圧力のベクトルを作る
        Title_Name = d[0] # タイトルの名前の1次元ベクトル

        # タイトルの名前のブーリアンリスト：最後に'PRES'がある列
        Record_Name_b =  [bool(Title_Name[i][-4:] == 'PRES') for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：'time'がある列:時刻
        Time_Name_b =  [bool(Title_Name[i] == 'time') for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：'BEAMCURRENTがある列：ビーム電流
        Beam_Name_b =  [bool(Title_Name[i] == BEAMCURRENT) for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：BUNCHNOがある列：バンチ数
        Nb_Name_b =  [bool(Title_Name[i] == BUNCHNO) for i in range(len(Title_Name))]

        # タイトルの名前の列をブーリアンで抜き出す：最後に'PRES'がある列
        Record_Data = d[:, Record_Name_b]
        # レコードの数 = 列の数
        N_Records = len(Record_Data[0])
        # タイトルの名前の列をブーリアンで抜き出す:'time'がある列
        Time_Data = d[:, Time_Name_b]
        # タイトルの名前の列をブーリアンで抜き出す:'BEAMCURRENTがある列
        Beam_Data = d[:, Beam_Name_b]
        # タイトルの名前の列をブーリアンで抜き出す:BUNCHNOがある列
        Nb_Data = d[:, Nb_Name_b] 
        
        print('Record data generated for ' + filename)
        print('Number of records in Local Room = ', N_Records, LC_Room, CCG_n[LC_Room_n-1] - 1)
        
        if(N_Records < 1 + 1 + 1):
            Error_f = 'Too few record number for ' + LC_Room
            return Date_Range_STD, Error_f
        
        # 各側室の元となるレコードデータの配列を保存する
        Record_Data_File_Name =  LC_Room + '_' + Mode_Para_c + '_Record_Data_' + Date_Range + '.npy' #20230816+c
        np.save(Record_Data_File_Name, Record_Data)

        # HOM power(Beam*Beam/Nb)をつくる:ビームはなし
        # ビーム電流とバンチ数の各要素をfloatにする:1番目のデータ(2個目のデータ)以降のベクトルにして
        Beam_Data_f =np.array([float(s) for s in np.delete(Beam_Data, 0)])
        Nb_Data_f = np.array([float(s) for s in np.delete(Nb_Data, 0)])

        # データ取得期間のビーム電流の最大値:ビームはなし
        Beam_Max = np.max(Beam_Data_f)

        # データ取得期間のバンチ数の最大値(参考)
        Nb_Data_max = np. max(Nb_Data_f)

        # HOM power(Beam*Beam/Nb)^2をつくる:ビームはないのでただのBeam*Beamとする。
        HOM_Data_f = (Beam_Data_f * Beam_Data_f) ** 2

        #その列の先頭に要素'(I*I/Nb)^2'(名前)を追加する:さらに、縦ベクトルにする:ビームはなし
        HOM_Data = np.append('(I*I/Nb)^2', HOM_Data_f).reshape(n_row, 1)

        # 解析開始(CHK_Strg)　-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- ---- -- -- -- 
        # 回帰曲線用の解析結果をまとめる表の最初の行
        Result_Data = np.array(['Record Name', 'W0(Beam current)', 'W1(HOM^2)', 'W2(Base)', 'RMSE', 'MaxRSE'])

        # 各レコードについて計算する
        for k in range(N_Records):

            # 列の名前を最初の行に入れる
            Time_Data_Sel = Time_Data[0]
            Record_Data_Sel = Record_Data[0, k]
            Beam_Data_Sel = Beam_Data[0]
            HOM_Data_Sel = HOM_Data[0]
            Nb_Data_Sel = Nb_Data[0]
            
            # 20240131変更
            # 一時的なダミーファイル
            Time_Data_Sel_d = Time_Data[0]
            Record_Data_Sel_d = Record_Data[0, k]
            Beam_Data_Sel_d = Beam_Data[0]
            HOM_Data_Sel_d = HOM_Data[0]
            Nb_Data_Sel_d = Nb_Data[0]

            # 解析のために条件に合うデータを選ぶ
            n_row_s = 0 # 選んだデータの表の行数を数える
            n_row_s_d = 0
            for i in range(1, n_row):
                # ビーム電流が50mA以下 # 20240206変更
                if (float(Beam_Data[i]) <= 50.):
                    Time_Data_Sel = np.vstack((Time_Data_Sel, Time_Data[i]))
                    Record_Data_Sel = np.vstack((Record_Data_Sel, Record_Data[i, k]))
                    Beam_Data_Sel = np.vstack((Beam_Data_Sel, Beam_Data[i]))
                    HOM_Data_Sel = np.vstack((HOM_Data_Sel, HOM_Data[i]))
                    Nb_Data_Sel = np.vstack((Nb_Data_Sel, Nb_Data[i]))
                    n_row_s += 1 # 選んだデータの表の行数
                # ダミーに入力(0mAのつもり)
                Time_Data_Sel_d = np.vstack((Time_Data_Sel_d, Time_Data[i]))
                Record_Data_Sel_d = np.vstack((Record_Data_Sel_d, '1.e-8'))
                Beam_Data_Sel_d = np.vstack((Beam_Data_Sel_d, '0.'))
                HOM_Data_Sel_d = np.vstack((HOM_Data_Sel_d, '0.'))
                Nb_Data_Sel_d = np.vstack((Nb_Data_Sel_d, '1.'))
                n_row_s_d += 1 # 選んだデータの表の行数
            
            # 20240131変更
            if(n_row_s <= 10): # 50mA以下のデータ10個以下だったら
                Time_Data_Sel = Time_Data_Sel_d
                Record_Data_Sel = Record_Data_Sel_d
                Beam_Data_Sel = Beam_Data_Sel_d
                HOM_Data_Sel = HOM_Data_Sel_d
                Nb_Data_Sel = Nb_Data_Sel_d
                n_row_s = n_row_s_d
            
            # レコードのデータを結合していく:列を足していく
            if(k == 0):
                Record_Data_Sel_List = Record_Data_Sel
            else:
                try:
                    Record_Data_Sel_List = np.append(Record_Data_Sel_List, Record_Data_Sel, axis = 1)
                except Exception as e:
                    print(traceback.format_exc())
                    print(str(e))
                    Error_f = str(e)
                    return Date_Range_CHK, Error_f
                
            # 1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
            Record_Data_Sel_f = np.array([float(s) for s in np.delete(Record_Data_Sel, 0)])
            Beam_Data_Sel_f = np.array([float(s) for s in np.delete(Beam_Data_Sel, 0)])
            HOM_Data_Sel_f = np.array([float(s) for s in np.delete(HOM_Data_Sel, 0)])
            # 圧力読み値(Record_Data_Sel_f)を3倍する
            Record_Data_Sel_f3 = Record_Data_Sel_f * 3.
    
            # 20240113変更：ビーム電流をデータの番号とする。-----------------------
            for i in range(len(Beam_Data_Sel_f)):
                Beam_Data_Sel_f[i] = float(i)
                HOM_Data_Sel_f[i] = float(i * i)
            for i in range(1, len(Beam_Data_Sel)):
                Beam_Data_Sel[i] = str(i - 1)
                HOM_Data_Sel[i] = str((i - 1) * (i - 1))
            
            if(n_row_s < 6): # 該当する値の行が10未満だったらベース圧の直線にする。
                W = [0., 0., 3.0e-8] # w2はベース圧力
                mse = 0.
                maxse = 0.
            else:
                # 解析解を求める
                # W = fit_plane(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3)
                # 20240330
                W = fit_line(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3)
        
                # 負の係数があれば制限付きの勾配法を用いる
                # if (W[0] < 0.) or (W[1] < 0.) or (W[2] < 0.):
                #    w_1 =[W[0], W[1], W[2]] # 初期値
                #    W, dMSE, Tau = fit_plane_num(w_1, Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3)
                #    print('繰り返し回数 {0}'.format(Tau))
    
                # if (W[2] < 1.e-9):
                #     W[2] = 1.e-9
                
                # 平均二乗誤差と最大二乗誤差の計算
                # mse, maxse = mse_plane(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3, W)
                # 20240330
                mse, maxse = mse_line(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3, W)
            
            # 圧力の平均を求める(読み値を3倍している)
            # record_ave = np.average(Record_Data_Sel_f3)
            # W = [0., 0., record_ave] # w2を平均値としている。(読み値の3倍)
            
            # RMSを求める
            # record_rms = 0
            # for i in range(len(Record_Data_Sel_f3) - 1):
            #     record_rms = record_rms + (Record_Data_Sel_f3[i] - record_ave)**2
            # record_rms = np.sqrt(record_rms / (len(Record_Data_Sel_f3)))
            
            # 各レコードの2次元回帰曲線のパラメータと標準偏差を解析結果をまとめる表に追加する
            Result_Data = np.vstack((Result_Data, np.array([Record_Data[0, k], '{:.3e}'.format(W[0]), '{:.3e}'.format(W[1]), 
                                                            '{:.3e}'.format(W[2]), '{:.3e}'.format(np.sqrt(mse)), 
                                                            '{:.3e}'.format(np.sqrt(maxse))])))
        
        #各側室でまとめる
        # 各側室の解析に使用したレコードデータの表を保存する
        Record_Data_File_Name_Sel =  LC_Room + '_' + Mode_Para_c + '_Record_Data_Sel_' + Date_Range + '.npy' #20230816+c
        np.save(Record_Data_File_Name_Sel, Record_Data_Sel_List)
            
        # 各側室の解析結果をまとめたものを表示する
        # print(Result_Data)

        # 各側室の解析結果をまとめた表を保存する
        Result_Data_File_Name =  LC_Room + '_' + Mode_Para_c + '_Result_Data_' + Date_Range + '.npy' #20230816+c
        np.save(Result_Data_File_Name, Result_Data)
        
        # 各側室の解析結果をまとめてリスト型辞書にする
        Keys = Result_Data[0].tolist()
        Datas = Result_Data[1:, :].tolist()
        # リスト型辞書にする
        Result_Data_Dict = [dict(zip(Keys, item)) for item in Datas]
        
        # 各側室のリスト型辞書をファイルに保存する
        Resut_Data_File_Name = LC_Room + '_' + Mode_Para_c + '_Result_Data_' + Date_Range + '_Dict' + '.npy' #20230816+c
        np.save(Resut_Data_File_Name, Result_Data_Dict)
        
        # 20240222追加　ローカルリストの削除
        del content, row_list, data, d, Title_Name, Record_Name_b, Time_Name_b, Beam_Name_b, Nb_Name_b
        del Record_Data, N_Records, Nb_Data, Beam_Data_f, Nb_Data_f, HOM_Data_f
        del Time_Data_Sel, Record_Data_Sel, Nb_Data_Sel
        del Time_Data_Sel_d, Record_Data_Sel_d, Beam_Data_Sel_d, HOM_Data_Sel_d, Nb_Data_Sel_d
        del Record_Data_Sel_List, Record_Data_Sel_f, Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3
        del W, Keys, Datas, Result_Data_Dict
    
    # リングでまとめる
    # リングのビーム電流、時刻、HOMの配列を保存する:すべての側室で共通
    Beam_File_Name =  Ring_Name + '_' + Mode_Para_c + '_Beam_' + Date_Range + '.npy' #20230816+c
    np.save(Beam_File_Name, Beam_Data)
        
    Time_File_Name = Ring_Name + '_' + Mode_Para_c + '_Time_' + Date_Range + '.npy' #20230816+c
    np.save(Time_File_Name, Time_Data)
    
    HOM_File_Name =  Ring_Name + '_' + Mode_Para_c + '_HOM_' + Date_Range + '.npy' #20230816+c
    np.save(HOM_File_Name, HOM_Data)
    
    # リングの解析に使用したビーム電流、HOMの配列を保存する:すべての側室で共通
    Beam_File_Name_Sel =  Ring_Name + '_' + Mode_Para_c + '_Beam_Sel_' + Date_Range + '.npy' #20230816+c
    np.save(Beam_File_Name_Sel, Beam_Data_Sel)
    
    HOM_File_Name_Sel =  Ring_Name + '_' + Mode_Para_c + '_HOM_Sel_' + Date_Range + '.npy' #20230816+c
    np.save(HOM_File_Name_Sel, HOM_Data_Sel)
    
    # リングのすべての側室の解析結果をまとめる
    # リングの時刻、ビーム電流、HOMデータをロードする
    Beam_File_Name =  Ring_Name + '_' + Mode_Para_c + '_Beam_' + Date_Range + '.npy' #20230816+c
    Beam_Data = np.load(Beam_File_Name)
        
    Time_File_Name = Ring_Name + '_' + Mode_Para_c + '_Time_' + Date_Range + '.npy' #20230816+c
    Time_Data = np.load(Time_File_Name)
    
    HOM_File_Name =  Ring_Name + '_' + Mode_Para_c + '_HOM_' + Date_Range + '.npy' #20230816+c
    HOM_Data = np.load(HOM_File_Name)
    
    # リングの時刻、ビーム電流、HOMデータをまとめる
    All_Data = np.append(Time_Data, Beam_Data, axis=1)
    All_Data = np.append(All_Data, HOM_Data, axis=1)
    
    # リングの解析に使用したビーム電流、HOMデータロードする
    Beam_File_Name_Sel =  Ring_Name + '_' + Mode_Para_c + '_Beam_Sel_' + Date_Range + '.npy' #20230816+c
    Beam_Data_Sel = np.load(Beam_File_Name_Sel)
    
    HOM_File_Name_Sel =  Ring_Name + '_' + Mode_Para_c + '_HOM_Sel_' + Date_Range + '.npy' #20230816+c
    HOM_Data_Sel = np.load(HOM_File_Name_Sel)
    
    # リングの解析に使用したビーム電流、HOMデータをまとめる
    All_Data_Sel = np.append(Beam_Data_Sel, HOM_Data_Sel, axis=1)
    
    # リングのすべてのデータをまとめる
    # 各側室のレコードデータの配列を読み出して、
    for LC_Room in List_Para:
        # 各側室のレコードデータの配列をロードする
        Record_Data_File_Name =  LC_Room + '_' + Mode_Para_c + '_Record_Data_' + Date_Range + '.npy' #20230816+c
        Record_Data_List = np.load(Record_Data_File_Name)
        
        # 各側室の解析に使用したレコードデータの配列を読み出す
        Record_Data_File_Name_Sel =  LC_Room + '_' + Mode_Para_c + '_Record_Data_Sel_' + Date_Range + '.npy' #20230816+c
        Record_Data_List_Sel = np.load(Record_Data_File_Name_Sel)
        
        # リングの時刻、ビーム電流、HOMデータにレコードデータを追加する
        All_Data = np.append(All_Data, Record_Data_List, axis=1)
        All_Data_Sel = np.append(All_Data_Sel, Record_Data_List_Sel, axis=1)
    
    # リングのすべてのデータを行列として保存する。
    All_File_Name =  Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_' + Date_Range_CHK + '.npz' #20230816+c
    np.savez_compressed(All_File_Name, All_Data)
    print('All Record Data was saved to ' + Ring_Name + '_' + Mode_Para + 
          '_All_Record_Data_' + Date_Range_CHK + '.npz' )
    
    # リングの解析に使用したすべてのデータを行列として保存する。
    All_File_Name_Sel =  Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_Sel_' + Date_Range_CHK + '.npz' #20230816+c
    np.savez_compressed(All_File_Name_Sel, All_Data_Sel)
    print('All Selected Record Data was saved to ' + Ring_Name + '_' + Mode_Para_c + 
          '_All_Record_Data_Sel_' + Date_Range_CHK + '.npz' )  #20230816+c
    
    # リングの解析結果を辞書にして保存する
    # 各側室の解析結果の辞書を読み出してまとめる
    All_Result_Dict = []
    for LC_Room in List_Para:
        Resut_Data_File_Name = LC_Room + '_' + Mode_Para_c + '_Result_Data_' + Date_Range + '_Dict' + '.npy' #20230816+c
        Result_Ar = np.load(Resut_Data_File_Name, allow_pickle = 'TRUE')
        Result_Dict = Result_Ar.tolist()
        All_Result_Dict = All_Result_Dict + Result_Dict
    
    # まとめたリングの解析結果を辞書として保存する。
    Result_Data_File_Name = Ring_Name + '_' + Mode_Para_c + '_All_Result_Data_' + Date_Range_CHK + '_Dict' #20230816+c
    # np.save(Result_Data_File_Name, All_Result_Dict)
    np.savez_compressed(Result_Data_File_Name, All_Result_Dict)
    print('All Result Data was saved to ' + Result_Data_File_Name )
    
    # Date_Range_CHKをファイルに保存する。
    f = open('Date_Range_CHK_File.txt', "w")
    f.write(Date_Range_CHK)
    f.close()
    print('Date_Range_CHK was saved to Date_Range_CHK_File.txt')

    # プロットして確かめる---------------------------------------------------------------------
    if(Check_Record_Name != "none"):   
        Plot_Record_Name = Check_Record_Name
        Date_Range_CHK_Strg = Date_Range_CHK
        Make_Plot_Strg_NB(Method, Plot_Record_Name, Date_Range_CHK_Strg, Mode_Para_c, List_Para, Abort_Timing)
        
    # ローカルリストの削除 20220222追加
    del Beam_Data, Time_Data, HOM_Data, Beam_Data_Sel, HOM_Data_Sel, All_Data, All_Data_Sel
    del Record_Data_List, Record_Data_List_Sel, Result_Ar, Result_Dict, All_Result_Dict
    
    # メモリーの開放
    gc.collect()
    
    return Date_Range_CHK, Error_f # Date_Rangeを返す

# ==============================================================================================================

def Get_DIF_Strg(List_Para, CCG_n):
# 各側室の各レコードについて、蓄積中の圧力に対してデータを取得する。解析はしない。
# ビームあり

    # エラーフラグリセット
    Error_f = 'none'
    
    # リング名:LERかHER
    Ring_Name = List_Para[0][:3]
    
    if(Ring_Name == 'LER'):
        BEAMCURRENT = 'BMLDCCT:CURRENT'
        BUNCHNO = 'CGLINJ:BKSEL:NOB_SET'
    elif(Ring_Name == 'HER'):
        BEAMCURRENT = 'BMHDCCT:CURRENT'
        BUNCHNO = 'CGHINJ:BKSEL:NOB_SET'
    
    # Mode
    Mode_Para = 'DIF_Strg'
    
    # データの時間間隔
    DT_Para = 'd30'
    
    # DIF_Strgのデータの時間範囲(ファイルから読み込む)
    path = Ring_Name + '_Date_Range_DIF_Strg_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print('No Date_Range_DIF_File')
        sys.exit()   
    f = open(Ring_Name + "_Date_Range_DIF_Strg_File.txt", "r")
    Date_Range_DIF = f.readline()
    f.close()
    
    # Date_Range：テータを取得する期間を決める
    # Date_Rangeの定義
    Date_Range = Date_Range_DIF
    Date_Para = [Date_Range] # リストにする
    
    # 測定データがローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
            for Date_Range_Test in Date_Para:
                path = LC_Room + '_' + Mode_Para + '_Data_' + Date_Range_Test +'.txt'
                is_file = os.path.isfile(path)
                if(is_file == False):
                    New_File = 1
                    
    # もし一つでも新しいファイルならデータを取ってローカルPCに保存する
    if(New_File == 1):
        # kekb-co-userに入ってデータを取ってファイルを保存する。タブ区切りのデータファイルが各側室毎にできる。
        Error_f = Get_kekbcouser(Ring_Name, Date_Para, DT_Para, Mode_Para, List_Para)
        if(Error_f != 'none'):
            tk.messagebox.showinfo('SSH Error', '    Restart again     ')
            return Date_Range_DIF, Error_f
        
        # データをLocal PCに持ってくる。
        Get_SFTP(Ring_Name, Date_Para, Mode_Para, List_Para)
    
    # データがローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
            for Date_Range_Test in Date_Para:
                path = LC_Room + '_' + Mode_Para + '_Record_Data_' + Date_Range_Test + '.npy'
                is_file = os.path.isfile(path)
                if(is_file == False):
                    New_File = 1

    path =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_' + Date_Range_DIF + '.npz'
    is_file = os.path.isfile(path)
    if(is_file == False):
                    New_File = 1
            
    path =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_Sel_' + Date_Range_DIF + '.npz'
    is_file = os.path.isfile(path)
    if(is_file == False):
                    New_File = 1
    
    if(New_File == 0): # データが既にあればDate_Rangeを返して関数を抜ける
        return Date_Range_DIF, Error_f
    
    # 20240118変更
    # ファイルの行数を計算する。
    ndifrow = int(Cal_File_Row(Date_Range, DT_Para))
    
    # 各側室についてループ
    LC_Room_n = 0
    for LC_Room in List_Para:
        # LC_Roomの番号
        LC_Room_n +=1
        
        # 読み込む込むデータのファイル名：タブ区切りのテキストファイル
        filename = LC_Room + '_' + Mode_Para + '_Data_' + Date_Range +'.txt'
        
        # テキストから読み込むデータのリストの定義
        content =[]
        with open (filename, encoding = 'utf8', newline = '') as f:
            csvreader = csv.reader(f, delimiter = '\t') # csvreaderにリストとして読み込む。タブ区切り。
            n_row = 0 #もとのデータの行数を求める変数：初期値
            for row in csvreader:
                if(n_row == 0):
                    nitem = len(row)
        
                list_row_filtered = filter(None, row)

                row_list = list(list_row_filtered)

                while(len(row_list) < nitem):
                    row_list = np.append(row_list, ['0']).tolist()
                                
                content = content + row_list #contentは1行のリストになる
                n_row += 1 #もとのcsvファイル(リスト)の行数となる
                # 20240118変更
                if(n_row == ndifrow + 2):
                    break
        
        #元となるデータの列数
        n_column = int(len(content)/n_row)
        
        # もとのデータのリスト'content'をベクトルのデータ'data'にする
        data = np.array(content)
        #ベクトルのデータ'data'をn_row行、n_column列の行列"d"にする
        d = data.reshape(n_row, n_column)

        # 時刻、ビーム電流、バンチ数、HOM Power、圧力のベクトルを作る
        Title_Name = d[0] # タイトルの名前の1次元ベクトル

        # タイトルの名前のブーリアンリスト：最後に'PRES'がある列
        Record_Name_b =  [bool(Title_Name[i][-4:] == 'PRES') for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：'time'がある列:時刻
        Time_Name_b =  [bool(Title_Name[i] == 'time') for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：BEAMCURRENTがある列：ビーム電流
        Beam_Name_b =  [bool(Title_Name[i] == BEAMCURRENT) for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：BUNCHNOがある列：バンチ数
        Nb_Name_b =  [bool(Title_Name[i] == BUNCHNO) for i in range(len(Title_Name))]

        # タイトルの名前の列をブーリアンで抜き出す：最後に'PRES'がある列
        Record_Data = d[:, Record_Name_b]
        # レコードの数 = 列の数
        N_Records = len(Record_Data[0])
        # タイトルの名前の列をブーリアンで抜き出す:'time'がある列
        Time_Data = d[:, Time_Name_b]
        # タイトルの名前の列をブーリアンで抜き出す:BEAMCURRENTがある列
        Beam_Data = d[:, Beam_Name_b]
        # タイトルの名前の列をブーリアンで抜き出す:BUNCHNOがある列
        Nb_Data = d[:, Nb_Name_b] 
        
        print('Record data generated for ' + filename)
        print('Number of records in Local Room = ', N_Records, LC_Room, CCG_n[LC_Room_n-1] - 1)
        
        if(N_Records < 1 + 1 + 1):
            Error_f = 'Too few record number for ' + LC_Room
            return Date_Range_STD, Error_f
        
        # 各側室の元となるレコードデータの配列を保存する
        Record_Data_File_Name =  LC_Room + '_' + Mode_Para + '_Record_Data_' + Date_Range + '.npy'
        np.save(Record_Data_File_Name, Record_Data)

        # HOM power(Beam*Beam/Nb)をつくる:ビームあり
        # ビーム電流とバンチ数の各要素をfloatにする:1番目のデータ(2個目のデータ)以降のベクトルにして
        Beam_Data_f =np.array([float(s) for s in np.delete(Beam_Data, 0)])
        Nb_Data_f = np.array([float(s) for s in np.delete(Nb_Data, 0)])

        # データ取得期間のビーム電流の最大値：ビームあり
        Beam_Max = np.max(Beam_Data_f)
        # print('Max of beam current = ', '{:.3f}'.format(Beam_Max))
        # データ取得期間のバンチ数の最大値(参考)
        Nb_Data_max = np. max(Nb_Data_f)

        # HOM power(Beam*Beam/Nb)^2をつくる:値のみでの計算する:ビームあり
        HOM_Data_f = (Beam_Data_f * Beam_Data_f / Nb_Data_f) ** 2

        #その列の先頭に要素'(I*I/Nb)^2'(名前)を追加する:さらに、縦ベクトルにする
        HOM_Data = np.append('(I*I/Nb)^2', HOM_Data_f).reshape(n_row, 1)

        # MSE計算に用いるデータを保管する。--------------------------------------------------------
        # 各レコードについて計算する
        for k in range(N_Records):

            # 列の名前を最初の行に入れる
            Time_Data_Sel = Time_Data[0]
            Record_Data_Sel = Record_Data[0, k]
            Beam_Data_Sel = Beam_Data[0]
            HOM_Data_Sel = HOM_Data[0]
            Nb_Data_Sel = Nb_Data[0]

            # 解析のために条件に合うデータを選ぶ
            n_row_s = 0 # 選んだデータの表の行数を数える
            for i in range(1, n_row):
                # 圧力値(読み値)が1.E-8 P以上
                # ビーム電流が50mA以上 -> # 20240206削除
                # ビーム電流が最大値の99.9%以下
                # ビーム電流が最大値の5%以上
                # バンチ数が10以上
                if (float(Record_Data[i, k]) >= 1.E-8) and (float(Beam_Data[i]) <= (Beam_Max * 0.999)) and \
                            (float(Nb_Data[i]) >= 10.) and (float(Beam_Data[i]) >= (Beam_Max * 0.05)):
                    Time_Data_Sel = np.vstack((Time_Data_Sel, Time_Data[i]))
                    Record_Data_Sel = np.vstack((Record_Data_Sel, Record_Data[i, k]))
                    Beam_Data_Sel = np.vstack((Beam_Data_Sel, Beam_Data[i]))
                    HOM_Data_Sel = np.vstack((HOM_Data_Sel, HOM_Data[i]))
                    Nb_Data_Sel = np.vstack((Nb_Data_Sel, Nb_Data[i]))
                    n_row_s += 1 # 選んだデータの表の行数
            
            # レコードのデータを結合していく:列を足していく
            if(k == 0):
                Record_Data_Sel_List = Record_Data_Sel
            else:
                try:
                    Record_Data_Sel_List = np.append(Record_Data_Sel_List, Record_Data_Sel, axis = 1)
                except Exception as e:
                    print(traceback.format_exc())
                    print(str(e))
                    Error_f = str(e)
                    return Date_Range_DIF, Error_f
                
            # 1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
            try:
                Record_Data_Sel_f = np.array([float(s) for s in np.delete(Record_Data_Sel, 0)])
            except Exception as e:
                print(traceback.format_exc())
                print(str(e))
                Error_f = str(e)
                return Date_Range_DIF, Error_f
            
            Beam_Data_Sel_f = np.array([float(s) for s in np.delete(Beam_Data_Sel, 0)])
            HOM_Data_Sel_f = np.array([float(s) for s in np.delete(HOM_Data_Sel, 0)])
            # 圧力読み値(Record_Data_Sel_f)を3倍する
            Record_Data_Sel_f3 = Record_Data_Sel_f * 3.
        
        #各側室でまとめる
        # 各側室のMSE計算に用いるレコードデータの表を保存する
        Record_Data_File_Name_Sel =  LC_Room + '_' + Mode_Para + '_Record_Data_Sel_' + Date_Range + '.npy'
        np.save(Record_Data_File_Name_Sel, Record_Data_Sel_List)
    
        # 20240222追加　ローカルリストの削除
        del content, row_list, data, d, Title_Name, Record_Name_b, Time_Name_b, Beam_Name_b, Nb_Name_b
        del Record_Data, N_Records, Nb_Data, Beam_Data_f, Nb_Data_f, HOM_Data_f
        del Time_Data_Sel, Record_Data_Sel, Beam_Data_Sel, HOM_Data_Sel, Nb_Data_Sel
        del Record_Data_Sel_List, Record_Data_Sel_f, Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3

    # リングでまとめる
    # リングのビーム電流、時刻、HOMの配列を保存する:すべての側室で共通
    Beam_File_Name =  Ring_Name + '_' + Mode_Para + '_Beam_' + Date_Range + '.npy'
    np.save(Beam_File_Name, Beam_Data)
        
    Time_File_Name = Ring_Name + '_' + Mode_Para + '_Time_' + Date_Range + '.npy'
    np.save(Time_File_Name, Time_Data)
    
    HOM_File_Name =  Ring_Name + '_' + Mode_Para + '_HOM_' + Date_Range + '.npy'
    np.save(HOM_File_Name, HOM_Data)
    
    # リングの解析に使用したビーム電流、HOMの配列を保存する:すべての側室で共通
    Beam_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_Beam_Sel_' + Date_Range + '.npy'
    np.save(Beam_File_Name_Sel, Beam_Data_Sel)
    
    HOM_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_HOM_Sel_' + Date_Range + '.npy'
    np.save(HOM_File_Name_Sel, HOM_Data_Sel)
    
    # リングのすべての側室の解析結果をまとめる
    # リングの時刻、ビーム電流、HOMデータをまとめる
    All_Data = np.append(Time_Data, Beam_Data, axis=1)
    All_Data = np.append(All_Data, HOM_Data, axis=1)
    
    # リングの解析に使用したビーム電流、HOMデータをまとめる
    All_Data_Sel = np.append(Beam_Data_Sel, HOM_Data_Sel, axis=1)
    
    # リングのすべてのデータをまとめる
    # 各側室のレコードデータの配列を読み出して、
    for LC_Room in List_Para:
        # 各側室のレコードデータの配列をロードする
        Record_Data_File_Name =  LC_Room + '_' + Mode_Para + '_Record_Data_' + Date_Range + '.npy'
        Record_Data_List = np.load(Record_Data_File_Name)
        
        # 各側室のMSE計算に用いるレコードデータの配列を読み出す
        Record_Data_File_Name_Sel =  LC_Room + '_' + Mode_Para + '_Record_Data_Sel_' + Date_Range + '.npy'
        Record_Data_List_Sel = np.load(Record_Data_File_Name_Sel)
        
        # リングの時刻、ビーム電流、HOMデータにレコードデータを追加する
        All_Data = np.append(All_Data, Record_Data_List, axis=1)
        All_Data_Sel = np.append(All_Data_Sel, Record_Data_List_Sel, axis=1)
    
    # リングのすべてのデータを行列として保存する。
    All_File_Name =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_' + Date_Range_DIF + '.npz'
    np.savez_compressed(All_File_Name, All_Data)
    print('All Record Data was saved to ' + Ring_Name + '_' + Mode_Para + 
          '_All_Record_Data_' + Date_Range_DIF + '.npz' )
    
    # リングのMSE計算に用いるすべてのデータを行列として保存する。
    All_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_Sel_' + Date_Range_DIF + '.npz'
    np.savez_compressed(All_File_Name_Sel, All_Data_Sel)
    print('All Selected Record Data was saved to ' + Ring_Name + '_' + Mode_Para + 
          '_All_Record_Data_Sel_' + Date_Range_DIF + '.npz' )
    
    # 20240222追加 ローカルリストの削除
    del Beam_Data, Time_Data, HOM_Data, Beam_Data_Sel, HOM_Data_Sel
    del All_Data, All_Data_Sel, Record_Data_List, Record_Data_List_Sel
    
    # メモリーの開放
    gc.collect()
    
    return Date_Range_DIF, Error_f # Date_Rangeを返す

# ==============================================================================================================
# ==============================================================================================================

def Get_DIF_Strg_NB(List_Para, CCG_n):
# 各側室の各レコードについて、蓄積中の圧力に対してデータを取得する。解析はしない。
# ビームはない

    # エラーフラグリセット
    Error_f = 'none'
    
    # リング名:LERかHER
    Ring_Name = List_Para[0][:3]
    
    if(Ring_Name == 'LER'):
        BEAMCURRENT = 'BMLDCCT:CURRENT'
        BUNCHNO = 'CGLINJ:BKSEL:NOB_SET'
    elif(Ring_Name == 'HER'):
        BEAMCURRENT = 'BMHDCCT:CURRENT'
        BUNCHNO = 'CGHINJ:BKSEL:NOB_SET'
    
    # Mode
    Mode_Para = 'DIF_Strg'
    
    # データの時間間隔
    DT_Para = 'd30'
    
    # DIF_Strgのデータの時間範囲(ファイルから読み込む)
    path = Ring_Name + '_Date_Range_DIF_Strg_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print('No Date_Range_DIF_File')
        sys.exit()   
    f = open(Ring_Name + "_Date_Range_DIF_Strg_File.txt", "r")
    Date_Range_DIF = f.readline()
    f.close()
    
    # Date_Range：テータを取得する期間を決める
    # Date_Rangeの定義
    Date_Range = Date_Range_DIF
    Date_Para = [Date_Range] # リストにする
    
    # 測定データがローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
            for Date_Range_Test in Date_Para:
                path = LC_Room + '_' + Mode_Para + '_Data_' + Date_Range_Test +'.txt'
                is_file = os.path.isfile(path)
                if(is_file == False):
                    New_File = 1
                    
    # もし一つでも新しいファイルならデータを取ってローカルPCに保存する
    if(New_File == 1):
        # kekb-co-userに入ってデータを取ってファイルを保存する。タブ区切りのデータファイルが各側室毎にできる。
        Error_f = Get_kekbcouser(Ring_Name, Date_Para, DT_Para, Mode_Para, List_Para)
        if(Error_f != 'none'):
            tk.messagebox.showinfo('SSH Error', '    Restart again     ')
            return Date_Range_DIF, Error_f
        
        # データをLocal PCに持ってくる。
        Get_SFTP(Ring_Name, Date_Para, Mode_Para, List_Para)
    
    # データがローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
            for Date_Range_Test in Date_Para:
                path = LC_Room + '_' + Mode_Para + '_Record_Data_' + Date_Range_Test + '.npy'
                is_file = os.path.isfile(path)
                if(is_file == False):
                    New_File = 1

    path =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_' + Date_Range_DIF + '.npz'
    is_file = os.path.isfile(path)
    if(is_file == False):
                    New_File = 1
            
    path =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_Sel_' + Date_Range_DIF + '.npz'
    is_file = os.path.isfile(path)
    if(is_file == False):
                    New_File = 1
    
    if(New_File == 0): # データが既にあればDate_Raneを返して関数を抜ける
        return Date_Range_DIF, Error_f
    
    # 20240118変更
    # ファイルの行数を計算する。
    ndifrow = int(Cal_File_Row(Date_Range, DT_Para))
    
    # 各側室についてループ
    LC_Room_n = 0
    for LC_Room in List_Para:
        # LC_Roomの番号
        LC_Room_n +=1
        
        # 読み込む込むデータのファイル名：タブ区切りのテキストファイル
        filename = LC_Room + '_' + Mode_Para + '_Data_' + Date_Range +'.txt'
        
        # テキストから読み込むデータのリストの定義
        content =[]
        with open (filename, encoding = 'utf8', newline = '') as f:
            csvreader = csv.reader(f, delimiter = '\t') # csvreaderにリストとして読み込む。タブ区切り。
            n_row = 0 #もとのデータの行数を求める変数：初期値
            for row in csvreader:
                if(n_row == 0):
                    nitem = len(row)
                list_row_filtered = filter(None, row)

                row_list = list(list_row_filtered)
                
                while(len(row_list) < nitem):
                    row_list = np.append(row_list, ['0']).tolist()
                    
                content = content + row_list #contentは1行のリストになる
                n_row += 1 #もとのcsvファイル(リスト)の行数となる
                
                # 20240118変更
                if(n_row == ndifrow + 2):
                    break
        
        #元となるデータの列数
        n_column = int(len(content)/n_row)
        
        # もとのデータのリスト'content'をベクトルのデータ'data'にする
        data = np.array(content)
        #ベクトルのデータ'data'をn_row行、n_column列の行列"d"にする
        d = data.reshape(n_row, n_column)

        # 時刻、ビーム電流、バンチ数、HOM Power、圧力のベクトルを作る
        Title_Name = d[0] # タイトルの名前の1次元ベクトル

        # タイトルの名前のブーリアンリスト：最後に'PRES'がある列
        Record_Name_b =  [bool(Title_Name[i][-4:] == 'PRES') for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：'time'がある列:時刻
        Time_Name_b =  [bool(Title_Name[i] == 'time') for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：BEAMCURRENTがある列：ビーム電流
        Beam_Name_b =  [bool(Title_Name[i] == BEAMCURRENT) for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：BUNCHNOがある列：バンチ数
        Nb_Name_b =  [bool(Title_Name[i] == BUNCHNO) for i in range(len(Title_Name))]

        # タイトルの名前の列をブーリアンで抜き出す：最後に'PRES'がある列
        Record_Data = d[:, Record_Name_b]
        # レコードの数 = 列の数
        N_Records = len(Record_Data[0])
        # タイトルの名前の列をブーリアンで抜き出す:'time'がある列
        Time_Data = d[:, Time_Name_b]
        # タイトルの名前の列をブーリアンで抜き出す:BEAMCURRENTがある列：ビームはなし
        Beam_Data = d[:, Beam_Name_b]
        # タイトルの名前の列をブーリアンで抜き出す:BUNCHNOがある列: ビームはなし
        Nb_Data = d[:, Nb_Name_b] 
        
        print('Record data generated for ' + filename)
        print('Number of records in Local Room = ', N_Records, LC_Room, CCG_n[LC_Room_n-1] - 1)
        
        # if(N_Records < CCG_n[LC_Room_n-1] - 1):
        if(N_Records < 1 + 1 + 1):
            Error_f = 'Too few record number for ' + LC_Room
            return Date_Range_STD, Error_f
        
        # 各側室の元となるレコードデータの配列を保存する
        Record_Data_File_Name =  LC_Room + '_' + Mode_Para + '_Record_Data_' + Date_Range + '.npy'
        np.save(Record_Data_File_Name, Record_Data)

        # HOM powerをつくる：ビームはなし
        # ビーム電流とバンチ数の各要素をfloatにする:1番目のデータ(2個目のデータ)以降のベクトルにして
        Beam_Data_f =np.array([float(s) for s in np.delete(Beam_Data, 0)])
        Nb_Data_f = np.array([float(s) for s in np.delete(Nb_Data, 0)])

        # データ取得期間のビーム電流の最大値：ビームはなし
        Beam_Max = np.max(Beam_Data_f)
        # データ取得期間のバンチ数の最大値(参考)：ビームはなし
        Nb_Data_max = np. max(Nb_Data_f)

        # HOM power(Beam*Beam/Nb)^2をつくる:ビームはないのでただのBeam*Beamとする。
        HOM_Data_f = (Beam_Data_f * Beam_Data_f) ** 2

        #その列の先頭に要素'(I*I/Nb)^2'(名前)を追加する:さらに、縦ベクトルにする：ビームはなし
        HOM_Data = np.append('(I*I/Nb)^2', HOM_Data_f).reshape(n_row, 1)

        # MSE計算に用いるデータを保管する。--------------------------------------------------------
        # 各レコードについて計算する
        for k in range(N_Records):

            # 列の名前を最初の行に入れる
            Time_Data_Sel = Time_Data[0]
            Record_Data_Sel = Record_Data[0, k]
            Beam_Data_Sel = Beam_Data[0]
            HOM_Data_Sel = HOM_Data[0]
            Nb_Data_Sel = Nb_Data[0]
            
            # 20240131変更
            # 一時的なダミーファイル
            Time_Data_Sel_d = Time_Data[0]
            Record_Data_Sel_d = Record_Data[0, k]
            Beam_Data_Sel_d = Beam_Data[0]
            HOM_Data_Sel_d = HOM_Data[0]
            Nb_Data_Sel_d = Nb_Data[0]

            # 解析のために条件に合うデータを選ぶ
            n_row_s = 0 # 選んだデータの表の行数を数える
            n_row_s_d = 0
            for i in range(1, n_row):
                # ビーム電流が50mA以下 # 20240206 変更
                if (float(Beam_Data[i]) <= 50.):
                    Time_Data_Sel = np.vstack((Time_Data_Sel, Time_Data[i]))
                    Record_Data_Sel = np.vstack((Record_Data_Sel, Record_Data[i, k]))
                    Beam_Data_Sel = np.vstack((Beam_Data_Sel, Beam_Data[i]))
                    HOM_Data_Sel = np.vstack((HOM_Data_Sel, HOM_Data[i]))
                    Nb_Data_Sel = np.vstack((Nb_Data_Sel, Nb_Data[i]))
                    n_row_s += 1 # 選んだデータの表の行数
                    
                # ダミーに入力(0mAのつもり)
                Time_Data_Sel_d = np.vstack((Time_Data_Sel_d, Time_Data[i]))
                Record_Data_Sel_d = np.vstack((Record_Data_Sel_d, '1.e-8'))
                Beam_Data_Sel_d = np.vstack((Beam_Data_Sel_d, '0.'))
                HOM_Data_Sel_d = np.vstack((HOM_Data_Sel_d, '0.'))
                Nb_Data_Sel_d = np.vstack((Nb_Data_Sel_d, '1.'))
                n_row_s_d += 1 # 選んだデータの表の行数
            
            # 20240131変更
            if(n_row_s <= 10): # 50mA以下のデータ10個以下だったら
                Time_Data_Sel = Time_Data_Sel_d
                Record_Data_Sel = Record_Data_Sel_d
                Beam_Data_Sel = Beam_Data_Sel_d
                HOM_Data_Sel = HOM_Data_Sel_d
                Nb_Data_Sel = Nb_Data_Sel_d
                n_row_s = n_row_s_d
                
            # レコードのデータを結合していく:列を足していく
            if(k == 0):
                Record_Data_Sel_List = Record_Data_Sel
            else:
                try:
                    Record_Data_Sel_List = np.append(Record_Data_Sel_List, Record_Data_Sel, axis = 1)
                except Exception as e:
                    print(traceback.format_exc())
                    print(str(e))
                    Error_f = str(e)
                    return Date_Range_DIF, Error_f
                
            # 1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
            Record_Data_Sel_f = np.array([float(s) for s in np.delete(Record_Data_Sel, 0)])
            Beam_Data_Sel_f = np.array([float(s) for s in np.delete(Beam_Data_Sel, 0)])
            HOM_Data_Sel_f = np.array([float(s) for s in np.delete(HOM_Data_Sel, 0)])
            # 圧力読み値(Record_Data_Sel_f)を3倍する
            Record_Data_Sel_f3 = Record_Data_Sel_f * 3.
            
            # 20240113変更：ビーム電流をデータの番号とする。-----------------------
            for i in range(len(Beam_Data_Sel_f)):
                Beam_Data_Sel_f[i] = float(i)
                HOM_Data_Sel_f[i] = float(i * i)
            for i in range(1, len(Beam_Data_Sel)):
                Beam_Data_Sel[i] = str(i - 1)
                HOM_Data_Sel[i] = str((i - 1) * (i - 1))
        
        #各側室でまとめる
        # 各側室のMSE計算に用いるレコードデータの表を保存する
        Record_Data_File_Name_Sel =  LC_Room + '_' + Mode_Para + '_Record_Data_Sel_' + Date_Range + '.npy'
        np.save(Record_Data_File_Name_Sel, Record_Data_Sel_List)
    
        # 20240222追加 ローカルリストの削除
        del content, row_list, data, d, Title_Name, Record_Name_b, Time_Name_b, Beam_Name_b, Nb_Name_b
        del Record_Data, N_Records, Nb_Data, Beam_Data_f, Nb_Data_f, HOM_Data_f
        del Time_Data_Sel, Record_Data_Sel, Nb_Data_Sel
        del Time_Data_Sel_d, Record_Data_Sel_d, Beam_Data_Sel_d, HOM_Data_Sel_d, Nb_Data_Sel_d
        del Record_Data_Sel_List, Record_Data_Sel_f, Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3
        
    # リングでまとめる
    # リングのビーム電流、時刻、HOMの配列を保存する:すべての側室で共通
    Beam_File_Name =  Ring_Name + '_' + Mode_Para + '_Beam_' + Date_Range + '.npy'
    np.save(Beam_File_Name, Beam_Data)
        
    Time_File_Name = Ring_Name + '_' + Mode_Para + '_Time_' + Date_Range + '.npy'
    np.save(Time_File_Name, Time_Data)
    
    HOM_File_Name =  Ring_Name + '_' + Mode_Para + '_HOM_' + Date_Range + '.npy'
    np.save(HOM_File_Name, HOM_Data)
    
    # リングの解析に使用したビーム電流、HOMの配列を保存する:すべての側室で共通
    Beam_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_Beam_Sel_' + Date_Range + '.npy'
    np.save(Beam_File_Name_Sel, Beam_Data_Sel)
    
    HOM_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_HOM_Sel_' + Date_Range + '.npy'
    np.save(HOM_File_Name_Sel, HOM_Data_Sel)
    
    # リングのすべての側室の解析結果をまとめる
    # リングの時刻、ビーム電流、HOMデータをまとめる
    All_Data = np.append(Time_Data, Beam_Data, axis=1)
    All_Data = np.append(All_Data, HOM_Data, axis=1)
    
    # リングの解析に使用したビーム電流、HOMデータをまとめる
    All_Data_Sel = np.append(Beam_Data_Sel, HOM_Data_Sel, axis=1)
    
    # リングのすべてのデータをまとめる
    # 各側室のレコードデータの配列を読み出して、
    for LC_Room in List_Para:
        # 各側室のレコードデータの配列をロードする
        Record_Data_File_Name =  LC_Room + '_' + Mode_Para + '_Record_Data_' + Date_Range + '.npy'
        Record_Data_List = np.load(Record_Data_File_Name)
        
        # 各側室のMSE計算に用いるレコードデータの配列を読み出す
        Record_Data_File_Name_Sel =  LC_Room + '_' + Mode_Para + '_Record_Data_Sel_' + Date_Range + '.npy'
        Record_Data_List_Sel = np.load(Record_Data_File_Name_Sel)
        
        # リングの時刻、ビーム電流、HOMデータにレコードデータを追加する
        All_Data = np.append(All_Data, Record_Data_List, axis=1)
        All_Data_Sel = np.append(All_Data_Sel, Record_Data_List_Sel, axis=1)
    
    # リングのすべてのデータを行列として保存する。
    All_File_Name =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_' + Date_Range_DIF + '.npz'
    np.savez_compressed(All_File_Name, All_Data)
    print('All Record Data was saved to ' + Ring_Name + '_' + Mode_Para + 
          '_All_Record_Data_' + Date_Range_DIF + '.npz' )
    
    # リングのMSE計算に用いるすべてのデータを行列として保存する。
    All_File_Name_Sel =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_Sel_' + Date_Range_DIF + '.npz'
    np.savez_compressed(All_File_Name_Sel, All_Data_Sel)
    print('All Selected Record Data was saved to ' + Ring_Name + '_' + Mode_Para + 
          '_All_Record_Data_Sel_' + Date_Range_DIF + '.npz' )
    
    # 20240222追加　ローカルリストの削除
    del Beam_Data, Time_Data, HOM_Data, Beam_Data_Sel, HOM_Data_Sel
    del All_Data, All_Data_Sel, Record_Data_List, Record_Data_List_Sel
    
    # メモリーの開放
    gc.collect()
    
    return Date_Range_DIF, Error_f # Date_Rangeを返す

# ==============================================================================================================

def Get_Fit_STD_Tail(List_Para, Method):
# 各側室の各レコードについて、アボート後(Tail)の圧力変化を、
# 最小二乗誤差法でフィットする回帰曲線を計算する。
    
    # エラーフラグのリセット
    Error_f = 'none'

    # モード
    Mode_Para = 'STD_Tail'
    # 遡る日数
    Dday = 3
    
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    
    if(Ring_Name == 'LER'):
        BEAMCURRENT = 'BMLDCCT:CURRENT'
        BUNCHNO = 'CGLINJ:BKSEL:NOB_SET'
    elif(Ring_Name == 'HER'):
        BEAMCURRENT = 'BMHDCCT:CURRENT'
        BUNCHNO = 'CGHINJ:BKSEL:NOB_SET'
        
    # STDのデータの時間範囲(ファイルから読み込む)
    path = Ring_Name + '_Date_Range_STD_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print(3047, 'No Date_Range_STD_File')
        sys.exit()   
    f = open(Ring_Name + "_Date_Range_STD_File.txt", "r")
    Date_Range_STD = f.readline()
    f.close()
    
    # Date_Rangeの定義
    Date_Range = Date_Range_STD
    Date_Para = [Date_Range] # リストにする
    
    # 1分単位でアボート時刻を見つける
    # Time_Data_Abort_Listを読み込む
    path = Ring_Name + 'Time_Data_Abort_STD_File.npy'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print('No Time_Data_Abort_STD_File')
        Error_f = 'no abort'
        return Date_Range, Error_f
    Time_Data_Abort_File_Name = Ring_Name + 'Time_Data_Abort_STD_File.npy'
    Time_Data_Abort = np.load(Time_Data_Abort_File_Name)

    # アボートした時刻でDate_Paraのリストをつくる
    Time_Data_Abort_List = ['Abort date para']
    for i in range (1, len(Time_Data_Abort)):
        datedata = Time_Data_Abort[i, 0]
        dt1 = Convert_Excel_to_Dtime(datedata)
        dt0 = dt1 + datetime.timedelta(minutes = -2) # ビーム電流がゼロになった時刻の2分前
        dt2 = dt1 + datetime.timedelta(minutes = 2) # ビーム電流がゼロになった時刻の2分後
            
        df0 = '{:%Y%m%d%H%M%S}'.format(dt0) # Excel形式の日付に書式変更
        df2 = '{:%Y%m%d%H%M%S}'.format(dt2) # Excel形式の日付に書式変更
            
        Time_Data_Abort_List.append(df0 + '-' + df2)
    
    # Tail用のパラメータ：2秒(DT_Para)毎にデータをつくる
    DT_Para ='d2'
    Date_Para_Abort = Time_Data_Abort_List[1:]
    
    # 測定データがローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
            for Date_Range_Abort in Date_Para_Abort:
                path = LC_Room + '_' + Mode_Para + '_Data_' + Date_Range_Abort +'.txt'
                is_file = os.path.isfile(path)
                if(is_file == False):
                    New_File = 1
    
    # もし一つでも新しいファイルならデータを取ってローカルPCに保存する
    if(New_File == 1):
        # kekb-co-userに入ってデータを取ってファイルを保存する。タブ区切りのデータファイルが各側室毎にできる。
        Error_f = Get_kekbcouser(Ring_Name, Date_Para_Abort, DT_Para, Mode_Para, List_Para)
        if(Error_f != 'none'):
            tk.messagebox.showinfo('SSH Error', '    Restart again     ')
            return Date_Range, Error_f
        
        # ファイルをLocal PCに持ってくる。
        Get_SFTP(Ring_Name, Date_Para_Abort, Mode_Para, List_Para)
    
    # 結果のファイルがローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
            for Date_Range_Test in Date_Para:
                path = LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Result_Data_' + Date_Range_Test + '.npy'
                is_file = os.path.isfile(path)
                if(is_file == False):
                    New_File = 1
    
    path =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Record_Data_' + Date_Range + '.npz'
    is_file = os.path.isfile(path)
    if(is_file == False):
                    New_File = 1
 
    path =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Record_Data_B_' + Date_Range + '.npz'
    is_file = os.path.isfile(path)
    if(is_file == False):
                    New_File = 1

    # 結果が既にあればDate_Rangeを返して関数を抜ける
    if(New_File == 0):
        return Date_Range, Error_f
    
    # 20240118変更
    # ファイルの行数を計算する。
    # nstdrow = int(Cal_File_Row(Date_Range, DT_Para))
    
    # 各側室についてループ
    for LC_Room in List_Para:
        #各時間範囲でループ
        n_rows_s = 0
        for k in range(len(Date_Para_Abort)):
            Date_Range_Abort = Date_Para_Abort[k]
            # 20240204変更
            # ファイルの行数を計算する。
            nstdrow = int(Cal_File_Row(Date_Range_Abort, DT_Para))
            # fileからテキストデータを読み込む
            filename = LC_Room + '_' + Mode_Para + '_Data_' + Date_Range_Abort +'.txt'
            
            # 読み込むテキストのリストの定義
            content =[]
            with open (filename, encoding = 'utf8', newline = '') as f:
                csvreader = csv.reader(f, delimiter = '\t') # csvreaderにリストとして読み込む。タブ区切り。
                n_row = 0 #もとのデータの行数を求める変数：初期値
                for row in csvreader:
                    if(n_row == 0):
                        nitem = len(row)
        
                    list_row_filtered = filter(None, row)

                    row_list = list(list_row_filtered)

                    while(len(row_list) < nitem):
                        row_list = np.append(row_list, ['0']).tolist()
                                
                    content = content + row_list #contentは1行のリストになる
                    n_row += 1 #もとのcsvファイル(リスト)の行数となる
                    # 20240118変更
                    if(n_row == nstdrow + 2):
                        break
            
            #もとのデータの列数
            n_column = int(len(content)/n_row) 
            
            data = np.array(content) # もとのデータのリスト'content'をベクトルのデータ'data'にする
            d = data.reshape(n_row, n_column) #ベクトルのデータ'data'をn_row行、n_column列の行列"d"にする
            
            # アボート後の時刻、ビーム電流、圧力のベクトルを作る  
            Title_Name = d[0] # タイトルの名前の1次元ベクトル

            # タイトルの名前のブーリアンリスト：'time'がある列:時刻
            Time_Name_b =  [bool(Title_Name[i] == 'time') for i in range(len(Title_Name))]
            # タイトルの名前のブーリアンリスト：BEAMCURRENTがある列：ビーム電流
            Beam_Name_b =  [bool(Title_Name[i] == BEAMCURRENT) for i in range(len(Title_Name))]
            # タイトルの名前のブーリアンリスト：最後に'PRES'がある列
            Record_Name_b =  [bool(Title_Name[i][-4:] == 'PRES') for i in range(len(Title_Name))]

            # タイトルの名前の列をブーリアンで抜き出す：最後に'PRES'がある列
            Record_Data = d[:, Record_Name_b]
            # レコードの数 = 列の数
            N_Records = len(Record_Data[0])
            # タイトルの名前の列をブーリアンで抜き出す:'time'がある列
            Time_Data = d[:, Time_Name_b]
            # タイトルの名前の列をブーリアンで抜き出す:BEAMCURRENTがある列
            Beam_Data = d[:, Beam_Name_b]
            
            # 時刻、ビーム電流、レコードデータのベクトルをつくる
            if(k == 0):
                Beam_Data_Tail = Beam_Data[0]
                Time_Data_Tail = Time_Data[0]
                Record_Data_Tail = Record_Data[0]
                Record_Data_Tail_Norm = Record_Data[0]
                Step_Tail = ['Time Step']
            
            #アボートしたタイミングを見つける：2秒(DT_Para)の単位で
            # 条件に合うデータを抜き出す
            j = 0
            for i in range(2, n_row - 1):
                # iのビーム電流が1mA未満(ビームアボート後)
                if (float(Beam_Data[i]) <= 1.):
                    Time_Data_Tail = np.vstack((Time_Data_Tail, Time_Data[i + 1]))
                    Beam_Data_Tail = np.vstack((Beam_Data_Tail, Beam_Data[i + 1]))
                    Record_Data_Tail = np.vstack((Record_Data_Tail, Record_Data[i + 1]))
                    if(j == 0):
                        rmax = Record_Data[i - 1] # ビームがある時
                    Record_Data_Tail_Norm = np.vstack((Record_Data_Tail_Norm, Record_Data[i + 1]))
                    Step_Tail = np.vstack((Step_Tail, str(j * int(DT_Para[-1:])))) # 秒単位にするため
                    j += 1
                    n_rows_s += 1
            
            # 各時刻のレコードデータを、ベース(rmin)を差し引いて規格化する
            rmin = 1.E-8
            for i in range(n_rows_s - j + 1, n_rows_s + 1):
                for m in range(len(Record_Data_Tail[i])):
                    if(float(rmax[m]) - rmin != 0) and (float(rmax[m]) != float(Record_Data_Tail_Norm[n_rows_s, m])):
                        Record_Data_Tail_Norm[i, m] = "{:.5f}".format((float(Record_Data_Tail_Norm[i, m]) - rmin) 
                                                                      / (float(rmax[m]) -rmin))
                    else:
                        Record_Data_Tail_Norm[i, m] = "{:.5f}".format(0.0)

        # 各側室の解析に使用するレコードデータ(もとの圧力と規格化した圧力)の配列を保存する
        Record_Data_Tail_File_Name =  LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Data_' + Date_Range + '.npy'
        np.save(Record_Data_Tail_File_Name, Record_Data_Tail)
        
        Record_Data_Tail_Norm_File_Name =  LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Data_Norm_' + Date_Range + '.npy'
        np.save(Record_Data_Tail_Norm_File_Name, Record_Data_Tail_Norm)
        
        # 各側室の回帰曲線用解析結果の表をつくる (STD_Tail)
        # 表の最初の行
        # 202401変更
        Result_Data_Tail = np.array(['Record Name', 'W0', 'W1', 'W2', 'W3', 'W4', 'RMSE', 'MaxRSE'])

        # 各レコードについて計算する
        for k in range(N_Records):
            Record_Data_Tail_Norm_Sel = Record_Data_Tail_Norm[:, k]

            # 1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
            Record_Data_Tail_Norm_Sel_f = np.array([float(s) 
                                                    for s in np.delete(Record_Data_Tail_Norm_Sel, 0)])
            Step_Tail_f = np.array([float(s) for s in np.delete(Step_Tail, 0)])
            
            # モデル_C 20240116、17、20変更
            # モデルmodel_Cを使って、mseを最小にする係数を求める。回帰曲線を求める。
            # 初期値
            # W_1 = [0.1, 700., 0.7, 300.] model_B
            W_1 = [1., 0.1, 1., 0.1] # model_C 
            # 解析
            W = fit_model_C(W_1, Step_Tail_f, Record_Data_Tail_Norm_Sel_f)
            
            # MSEの計算
            mse = mse_model_C(W, Step_Tail_f, Record_Data_Tail_Norm_Sel_f)
            
            # 最大SEの計算
            maxse = maxse_model_C(W, Step_Tail_f, Record_Data_Tail_Norm_Sel_f)
            
            # 結果の表示
            # print("w0={0:.3e}, w1={1:.3e}, w2 = {2:.3e}, w3 = {3:.3e}".format(W[0], W[1], W[2], W[3]))

            # 20240320変更
            # 各レコードの2次元回帰曲線のパラメータと標準偏差を行列にまとめる
            Result_Data_Tail = np.vstack((Result_Data_Tail, np.array([Record_Data[0, k], '{:.3e}'.format(abs(W[0])), 
                                            '{:.3e}'.format(abs(W[1])), '{:.3e}'.format(abs(W[2])), '{:.3e}'.format(abs(W[3])), 
                                            '{:.3e}'.format(0),'{:.3e}'.format(np.sqrt(mse)), 
                                            '{:.3e}'.format(np.sqrt(maxse))])))

        # 各側室のまとめた結果を保存する
        Result_Data_File_Name =  LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Result_Data_' + Date_Range + '.npy'
        np.save(Result_Data_File_Name, Result_Data_Tail)
        
        # 各側室のまとめた結果をリスト型辞書にする
        Keys = Result_Data_Tail[0].tolist()
        Datas = Result_Data_Tail[1:, :].tolist()

        # リスト型辞書
        Result_Data_Tail_Dict = [dict(zip(Keys, item)) for item in Datas]

        # 各側室のリスト型辞書をファイルに保存する
        Resut_Data_File_Name_Dict = LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Result_Data_' + Date_Range \
                                        + '_Dict' + '.npy'
        np.save(Resut_Data_File_Name_Dict, Result_Data_Tail_Dict)
        
        # 20240222追加　ローカルリストの削除
        del content, row_list, data, d
        del Title_Name, Time_Name_b, Beam_Name_b, Record_Name_b, Record_Data, N_Records, Time_Data, Beam_Data
        del Record_Data_Tail, Record_Data_Tail_Norm
        del Result_Data_Tail, Record_Data_Tail_Norm_Sel, Record_Data_Tail_Norm_Sel_f, Step_Tail_f
        del W, Keys, Datas, Result_Data_Tail_Dict

    # リングでまとめる
    # リングのビーム電流、時刻、アボート後の時間ステップの配列を保存する:すべての側室で共通
    Time_Step_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Time_Step_' + Date_Range + '.npy'
    np.save(Time_Step_File_Name, Step_Tail)
    
    Beam_File_Name =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Beam_' + Date_Range + '.npy'
    np.save(Beam_File_Name, Beam_Data_Tail)
        
    Time_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Time_' + Date_Range + '.npy'
    np.save(Time_File_Name, Time_Data_Tail)
                
    # リングのすべての側室の解析結果をまとめる
    # まず、アボート後(Tail)の時間ステップを読み込む
    Time_Step_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_Time_Step_' + Date_Range + '.npy'
    All_Data_Tail = np.load(Time_Step_File_Name)
    All_Data_Tail_B = np.load(Time_Step_File_Name)
    
    # リングの各側室のレコードデータの配列を読み出してまとめる
    for LC_Room in List_Para:
        # 各側室のレコードデータの配列を読み出す
        Record_Data_Norm_File_Name =  LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Data_Norm_' + Date_Range + '.npy'
        Record_Data_Norm_List = np.load(Record_Data_Norm_File_Name)
        Record_Data_File_Name =  LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Data_' + Date_Range + '.npy'
        Record_Data_List = np.load(Record_Data_File_Name)
        
        # データの列を追加する
        All_Data_Tail = np.append(All_Data_Tail, Record_Data_Norm_List, axis = 1)
        All_Data_Tail_B = np.append(All_Data_Tail_B, Record_Data_List, axis = 1)
    
    # 時間ステップ、すべての側室のレコードのデータを行列として保存する。
    All_File_Name =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Record_Data_' + Date_Range + '.npz'
    np.savez_compressed(All_File_Name, All_Data_Tail)
    All_File_Name_B =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Record_Data_B_' + Date_Range + '.npz'
    np.savez_compressed(All_File_Name_B, All_Data_Tail_B)
    
    print(Ring_Name + ' All Record Data was saved to ' + All_File_Name)
    
    # 各側室の解析結果のリスト型辞書を読み出してまとめる
    All_Result_Dict = []
    for LC_Room in List_Para:
        Resut_Data_File_Name = LC_Room + '_' + Mode_Para + '_' + Method + '_Class2_Result_Data_' + Date_Range + '_Dict' + '.npy'
        Result_Ar = np.load(Resut_Data_File_Name, allow_pickle = 'TRUE')
        Result_Dict = Result_Ar.tolist()
        All_Result_Dict = All_Result_Dict + Result_Dict
    
    # すべての側室のレコードの解析結果をリスト型辞書として保存する。
    Result_Data_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Result_Data_' + Date_Range + '_Dict'
    # np.save(Result_Data_File_Name, All_Result_Dict)
    np.savez_compressed(Result_Data_File_Name, All_Result_Dict)
    print('All Result Data was saved to ' + Result_Data_File_Name)
    
    # 20240222追加　ローカルリストの削除
    del Step_Tail, Beam_Data_Tail, Time_Data_Tail, All_Data_Tail, All_Data_Tail_B
    del Record_Data_Norm_List, Record_Data_List
    del Result_Ar, Result_Dict, All_Result_Dict
    
    # メモリーの開放
    gc.collect()
    
    # Date_Rangeを返す
    return Date_Range, Error_f

# ==============================================================================================================
# ==============================================================================================================

def Get_Fit_CHK_Tail(Method, Check_Record_Name, Mode_Para, List_Para, Abort_Timing):
# 各側室の各レコードについて、アボート後(Tail)の圧力変化を、
# 最小二乗誤差法でフィットする回帰曲線を計算する

    # エラーフラグリセット:'none'なら正常
    Error_f = 'none'
    
    Mode_Para_c = 'DIF_Tail'
    
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    if(Ring_Name == 'LER'):
        BEAMCURRENT = 'BMLDCCT:CURRENT'
        BUNCHNO = 'CGLINJ:BKSEL:NOB_SET'
    elif(Ring_Name == 'HER'):
        BEAMCURRENT = 'BMHDCCT:CURRENT'
        BUNCHNO = 'CGHINJ:BKSEL:NOB_SET'
    
    # Date_Range：CHK_Tailのテータを取得する期間を決める。
    # Date_Range_CHK_Tailの定義
    path = Ring_Name + '_Date_Range_CHK_Tail_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print('No Date_Range_CHK_Tail_File')
        Error_f = 'No date file'
        Date_Range_CHK = 'none'
        return Date_Range_CHK, Error_f
    
    f = open(Ring_Name + "_Date_Range_CHK_Tail_File.txt", "r")
    Date_Range_CHK_Tail = f.readline()
    f.close()
    
    Date_Range = Date_Range_CHK_Tail
    Date_Para = [Date_Range]
    
    # Date_Range_CHKをロードする
    # データのファイルがあるか？
    path = Ring_Name + '_Date_Range_CHK_Strg_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print('No Date_Range_CHK_Strg_File')
        Error_f = 'No date file'
        Date_Range_CHK = "none"
        return Date_Range_CHK, Error_f
    
    # データのファイルからDate_Range_CHKを読み込む    
    f = open(Ring_Name + "_Date_Range_CHK_Strg_File.txt", "r")
    Date_Range_CHK_Strg = f.readline()
    f.close()
    
    Date_Range_CHK= Date_Range_CHK_Strg

    Date_Para_S = Date_Range[:14]

    # アボート時刻を見つける
    Time_Data_Abort_List = ['Abort date para']
    datedata = Date_Para_S
    dt1 = Convert_Kblogrd_to_Dtime(datedata)
    dt0 = dt1 + datetime.timedelta(minutes = -2) # ビーム電流がゼロになった時刻の2分前
    dt2 = dt1 + datetime.timedelta(minutes = 2) # ビーム電流がゼロになった時刻の2分後
            
    Date_Para_S = '{:%Y%m%d%H%M%S}'.format(dt0) # 書式変更
    Date_Para_E = '{:%Y%m%d%H%M%S}'.format(dt2) # 書式変更
        
    Time_Data_Abort_List.append(Date_Para_S + '-' + Date_Para_E)
    
    # Tail用のパラメータ：2秒(DT_Para)毎にデータをつくる
    DT_Para ='d2'
    Date_Para_Abort = Time_Data_Abort_List[1:]
    
    print('Date_Range_CHK_Tail =', Date_Range_CHK_Tail)
    print('Date_Para_Abort=', Date_Para_Abort)
    
    # 測定データがローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
            for Date_Range_Abort in Date_Para_Abort:
                path = LC_Room + '_' + Mode_Para_c + '_Data_' + Date_Range_Abort +'.txt' #20230816+c
                is_file = os.path.isfile(path)
                if(is_file == False):
                    New_File = 1
    
    # もし一つでも新しいファイルならデータを取ってローカルPCに保存する
    if(New_File == 1):
        # kekb-co-userに入ってデータを取ってファイルを保存する。タブ区切りのデータファイルが各側室毎にできる。
        Error_f = Get_kekbcouser(Ring_Name, Date_Para_Abort, DT_Para, Mode_Para_c, List_Para)
        if(Error_f != 'none'):
            tk.messagebox.showinfo('SSH Error', '    Restart again     ')
            return Date_Range_CHK, Error_f
        
        # ファイルをLocal PCに持ってくる。
        Get_SFTP(Ring_Name, Date_Para_Abort, Mode_Para_c, List_Para)
    
    # 回帰曲線用の結果のファイルがローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
            for Date_Range_Test in Date_Para:
                path = LC_Room + '_' + Mode_Para_c + '_Result_Data_' + Date_Range_Test + '.npy' #20230816+c
                is_file = os.path.isfile(path)
                if(is_file == False):
                    New_File = 1
    
    path =  Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_' + Date_Range_CHK + '.npz' #20230816+c
    is_file = os.path.isfile(path)
    if(is_file == False):
                    New_File = 1
 
    path =  Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_B_' + Date_Range_CHK + '.npz' #20230816+c
    is_file = os.path.isfile(path)
    if(is_file == False):
                    New_File = 1
    
    # 結果が既にあればDate_Rangeを返して関数を抜ける
    if(New_File == 0): 
        # プロットして確かめる
        if(Check_Record_Name != "none"):
            Plot_Record_Name = Check_Record_Name
            Date_Range_CHK_Tail = Date_Range_CHK
                
            Make_Plot_Tail(Method, Plot_Record_Name, Date_Range_CHK_Tail, Mode_Para_c, List_Para, Abort_Timing)
        
        return Date_Range_CHK, Error_f
    
    # 20240207変更
    # ファイルの行数を計算する。
    # ndifrow = int(Cal_File_Row(Date_Para_Abort[0], DT_Para))
    # print(3504, 'Date_Range =', Date_Range, ', DT_Para -', DT_Para, ndifrow)
    # print(3608, 'Date_Para_Abort', Date_Para_Abort)
    
    # 各側室についてループ
    for LC_Room in List_Para:
        #各時間範囲でループ
        n_rows_s = 0
        for k in range(len(Date_Para_Abort)):
            Date_Range_Abort = Date_Para_Abort[k]
            # 20240207変更
            # ファイルの行数を計算する。
            ndifrow = int(Cal_File_Row(Date_Range_Abort, DT_Para))
            # fileからテキストデータを読み込む
            filename = LC_Room + '_' + Mode_Para_c + '_Data_' + Date_Range_Abort +'.txt' #20230816+c
            
            # 読み込むテキストのリストの定義
            content =[]
            
            with open (filename, encoding = 'utf8', newline = '') as f:
                csvreader = csv.reader(f, delimiter = '\t') # csvreaderにリストとして読み込む。タブ区切り。
                n_row = 0 #もとのデータの行数を求める変数：初期値
                for row in csvreader:
                    if(n_row == 0):
                        nitem = len(row)
        
                    list_row_filtered = filter(None, row)
                    row_list = list(list_row_filtered)

                    while(len(row_list) < nitem):
                        row_list = np.append(row_list, ['0']).tolist()
                                
                    content = content + row_list #contentは1行のリストになる
                    n_row += 1 #もとのcsvファイル(リスト)の行数となる
                    
                    # 20240118変更
                    if(n_row == ndifrow + 2):
                        break
            
            #もとのデータの列数
            n_column = int(len(content)/n_row) 
            
            # もとのデータのリスト'content'をベクトルのデータ'data'にする
            data = np.array(content) 
            d = data.reshape(n_row, n_column) #ベクトルのデータ'data'をn_row行、n_column列の行列"d"にする
            
            # アボート後の時刻、ビーム電流、圧力のベクトルを作る  
            Title_Name = d[0] # タイトルの名前の1次元ベクトル

            # タイトルの名前のブーリアンリスト：最後に'PRES'がある列
            Record_Name_b =  [bool(Title_Name[i][-4:] == 'PRES') for i in range(len(Title_Name))]
            # タイトルの名前のブーリアンリスト：'time'がある列:時刻
            Time_Name_b =  [bool(Title_Name[i] == 'time') for i in range(len(Title_Name))]
            # タイトルの名前のブーリアンリスト：BEAMCURRENTがある列：ビーム電流
            Beam_Name_b =  [bool(Title_Name[i] == BEAMCURRENT) for i in range(len(Title_Name))]

            # タイトルの名前の列をブーリアンで抜き出す：最後に'PRES'がある列
            Record_Data = d[:, Record_Name_b]
            # レコードの数 = 列の数
            N_Records = len(Record_Data[0])
            # タイトルの名前の列をブーリアンで抜き出す:'time'がある列
            Time_Data = d[:, Time_Name_b]
            # タイトルの名前の列をブーリアンで抜き出す:BEAMCURRENTがある列
            Beam_Data = d[:, Beam_Name_b]
            
            if(k == 0):
                Beam_Data_Tail = Beam_Data[0]
                Time_Data_Tail = Time_Data[0]
                Record_Data_Tail = Record_Data[0]
                Record_Data_Tail_Norm = Record_Data[0]
                Step_Tail = ['Time Step']
            
            #アボートしたタイミングを見つける：2秒(DT_Para)の単位で
            # 条件に合うデータを抜き出す
            j = 0
            for i in range(2, n_row - 1):
                # iのビーム電流が1mA未満
                if (float(Beam_Data[i]) <= 1.):
                    Time_Data_Tail = np.vstack((Time_Data_Tail, Time_Data[i + 1]))
                    Beam_Data_Tail = np.vstack((Beam_Data_Tail, Beam_Data[i + 1]))
                    Record_Data_Tail = np.vstack((Record_Data_Tail, Record_Data[i + 1]))
                    if(j == 0):
                        rmax = Record_Data[i - 1] # ビームがある時
                    Record_Data_Tail_Norm = np.vstack((Record_Data_Tail_Norm, Record_Data[i + 1]))
                    Step_Tail = np.vstack((Step_Tail, str(j * int(DT_Para[-1:])))) # 秒単位にするため
                    j += 1
                    n_rows_s += 1
            
            # 各時刻のデータを、ベース(rmin)を差し引いて規格化する
            rmin = 1.E-8
            for i in range(n_rows_s - j + 1, n_rows_s + 1):
                for m in range(len(Record_Data_Tail[i])):
                    if(float(rmax[m]) - rmin != 0) and (float(rmax[m]) != float(Record_Data_Tail_Norm[n_rows_s, m])):
                        Record_Data_Tail_Norm[i, m] = "{:.5f}".format((float(Record_Data_Tail_Norm[i, m]) - rmin) 
                                                                      / (float(rmax[m]) -rmin))
                    else:
                        Record_Data_Tail_Norm[i, m] = "{:.5f}".format(0.0)
        
        # print(3596, Beam_Data)
        # print(3597, Beam_Data_Tail)
        if(Record_Data_Tail_Norm.ndim == 1):
            print('\nNo Tail Data. Check HER or LER.')
            Error_f = 'No Tail Data'
            return Date_Range_CHK, Error_f
            
        # 各側室の解析に使用したレコードデータ(もとの圧力と規格化した圧力)の配列を保存する
        Record_Data_Tail_File_Name =  LC_Room + '_' + Mode_Para_c + '_Data_' + Date_Range + '.npy' #20230816+c
        np.save(Record_Data_Tail_File_Name, Record_Data_Tail)
        Record_Data_Tail_Norm_File_Name =  LC_Room + '_' + Mode_Para_c + '_Data_Norm_' + Date_Range + '.npy'#20230816+c
        np.save(Record_Data_Tail_Norm_File_Name, Record_Data_Tail_Norm)
        
        # 202401変更
        # 各側室の解析結果の表をつくる
        # 表の最初の行
        Result_Data_Tail = np.array(['Record Name', 'W0', 'W1', 'W2', 'W3', 'W4', 'RMSE', 'MaxRSE'])

        # 各レコードについて計算する
        for k in range(N_Records):
            Record_Data_Tail_Norm_Sel = Record_Data_Tail_Norm[:, k]

            # 1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
            Record_Data_Tail_Norm_Sel_f = np.array([float(s) 
                                                    for s in np.delete(Record_Data_Tail_Norm_Sel, 0)])
            Step_Tail_f = np.array([float(s) for s in np.delete(Step_Tail, 0)])
            
            # model_C 202401変更
            # モデルmodel_Cを使って、mseを最小にする係数を求める。回帰曲線を求める。
            # 初期値
            W_1 = [1., 0.1, 1., 0.1] # model_C
            
            # 解析
            W = fit_model_C(W_1, Step_Tail_f, Record_Data_Tail_Norm_Sel_f)
            
            # MSEの計算
            mse = mse_model_C(W, Step_Tail_f, Record_Data_Tail_Norm_Sel_f)
            
            # 最大SEの計算
            maxse = maxse_model_C(W, Step_Tail_f, Record_Data_Tail_Norm_Sel_f)
            
            # 結果の表示
            # print("w0={0:.3e}, w1={1:.3e}, w2 = {2:.3e}, w3 = {3:.3e}".format(W[0], W[1], W[2], W[3]))

            # 20240320変更
            # 各レコードの2次元回帰曲線のパラメータと標準偏差を行列にまとめる
            Result_Data_Tail = np.vstack((Result_Data_Tail, np.array([Record_Data[0, k], '{:.3e}'.format(abs(W[0])), 
                                                        '{:.3e}'.format(abs(W[1])), '{:.3e}'.format(abs(W[2])), 
                                                        '{:.3e}'.format(abs(W[3])), '{:.3e}'.format(0), 
                                                        '{:.3e}'.format(np.sqrt(mse)), '{:.3e}'.format(np.sqrt(maxse))])))

        # 各側室のまとめた結果を保存する
        Result_Data_File_Name =  LC_Room + '_' + Mode_Para_c + '_Result_Data_' + Date_Range + '.npy'#20230816+c
        np.save(Result_Data_File_Name, Result_Data_Tail)
        
        # 各側室のまとめた結果をリスト型辞書にする
        Keys = Result_Data_Tail[0].tolist()
        Datas = Result_Data_Tail[1:, :].tolist()

        # リスト型辞書
        Result_Data_Tail_Dict = [dict(zip(Keys, item)) for item in Datas]

        # 各側室のリスト型辞書をファイルに保存する
        Resut_Data_File_Name_Dict = LC_Room + '_' + Mode_Para_c + '_Result_Data_' + Date_Range \
                                        + '_Dict' + '.npy' #20230816+c
        np.save(Resut_Data_File_Name_Dict, Result_Data_Tail_Dict)

        # 20240222追加　ローカルリストの削除
        del content, row_list, data, d
        del Title_Name, Time_Name_b, Beam_Name_b, Record_Name_b, Record_Data, N_Records, Time_Data, Beam_Data
        del Beam_Data_Tail, Time_Data_Tail, Record_Data_Tail, Record_Data_Tail_Norm
        del Result_Data_Tail, Record_Data_Tail_Norm_Sel, Record_Data_Tail_Norm_Sel_f, Step_Tail_f
        del W, Keys, Datas, Result_Data_Tail_Dict

    # リングのアボート後の時間ステップの配列を保存する
    Time_Step_File_Name = Ring_Name + '_' + Mode_Para_c + '_Time_Step_' + Date_Range_CHK + '.npy'#20230816+c
    np.save(Time_Step_File_Name, Step_Tail)
    
    # リングのすべての側室の解析結果をまとめる
    # まず、アボート後(Tail)の時間ステップを読み込む
    Time_Step_File_Name = Ring_Name + '_' + Mode_Para_c + '_Time_Step_' + Date_Range_CHK + '.npy'#20230816+c
    All_Data_Tail = np.load(Time_Step_File_Name)
    All_Data_Tail_B = np.load(Time_Step_File_Name)
    
    # リングの各側室のレコードデータの配列を読み出してまとめる
    for LC_Room in List_Para:
        # 各側室のレコードデータの配列を読み出す
        Record_Data_Norm_File_Name =  LC_Room + '_' + Mode_Para_c + '_Data_Norm_' + Date_Range + '.npy'#20230816+c
        Record_Data_Norm_List = np.load(Record_Data_Norm_File_Name)
        Record_Data_File_Name =  LC_Room + '_' + Mode_Para_c + '_Data_' + Date_Range + '.npy' #20230816+c
        Record_Data_List = np.load(Record_Data_File_Name)
        
        # データの列を追加する
        try:
            All_Data_Tail = np.append(All_Data_Tail, Record_Data_Norm_List, axis = 1)
        except Exception as e:
            print(traceback.format_exc())
            print(str(e))
            Error_f = str(e)
            return Date_Range_CHK, Error_f
        
        All_Data_Tail_B = np.append(All_Data_Tail_B, Record_Data_List, axis = 1)
    
    All_File_Name =  Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_' + Date_Range_CHK + '.npz' #20230816+c
    np.savez_compressed(All_File_Name, All_Data_Tail)
    All_File_Name_B =  Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_B_' + Date_Range_CHK + '.npz' #20230816+c
    np.savez_compressed(All_File_Name_B, All_Data_Tail_B)
    
    print(3841, Ring_Name + ' All Record Data was saved to ' + All_File_Name)
    
    # 各側室の解析結果のリスト型辞書を読み出してまとめる
    All_Result_Dict = []
    for LC_Room in List_Para:
        Resut_Data_File_Name = LC_Room + '_' + Mode_Para_c + '_Result_Data_' + Date_Range + '_Dict' + '.npy' #20230816+c
        Result_Ar = np.load(Resut_Data_File_Name, allow_pickle = 'TRUE')
        Result_Dict = Result_Ar.tolist()
        All_Result_Dict = All_Result_Dict + Result_Dict
    
    # すべての側室のレコードの解析結果をリスト型辞書として保存する。
    Result_Data_File_Name = Ring_Name + '_' + Mode_Para_c + '_All_Result_Data_' + Date_Range_CHK + '_Dict' #20230816+c
    np.savez_compressed(Result_Data_File_Name, All_Result_Dict)
    print('All Result Data was saved to ' + Result_Data_File_Name)

    # プロットして確かめる---------------------------------------------------------------------
    if(Check_Record_Name != "none"):
        Plot_Record_Name = Check_Record_Name
        Date_Range_CHK_Tail = Date_Range_CHK
        Make_Plot_Tail(Method, Plot_Record_Name, Date_Range_CHK_Tail, Mode_Para_c, List_Para, Abort_Timing)
            
    # 20240222追加 ローカルリストの削除
    del Step_Tail, All_Data_Tail, All_Data_Tail_B
    del Record_Data_Norm_List, Record_Data_List
    del Result_Ar, Result_Dict, All_Result_Dict
    
    # メモリーの開放
    gc.collect()
    
    return Date_Range_CHK, Error_f # Date_Rangeとエラーフラグを返す

# ==============================================================================================================

def Get_DIF_Tail(List_Para):
# 各側室の各レコードについて、アボート後(Tail)の圧力をMSE計算のため読み出す

    # エラーフラグリセット:'none'なら正常
    Error_f = 'none'

    # リングの名前 LERかHER
    Ring_Name = List_Para[0][:3]
    
    if(Ring_Name == 'LER'):
        BEAMCURRENT = 'BMLDCCT:CURRENT'
        BUNCHNO = 'CGLINJ:BKSEL:NOB_SET'
    elif(Ring_Name == 'HER'):
        BEAMCURRENT = 'BMHDCCT:CURRENT'
        BUNCHNO = 'CGHINJ:BKSEL:NOB_SET'
    
    # Mode
    Mode_Para = 'DIF_Tail'
    
    # Date_Range：DIF_Tailのテータを取得する期間を決める。
    # Date_Range_DIF_Tailの定義
    path = Ring_Name + '_Date_Range_DIF_Tail_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print('No Date_Range_DIF_Tail_File')
        Error_f = 'No date file'
        Date_Range_CHK = 'none'
        return Date_Range_DIF, Error_f
        
    f = open(Ring_Name + "_Date_Range_DIF_Tail_File.txt", "r")
    Date_Range_DIF_Tail = f.readline()
    f.close()  
    
    Date_Range = Date_Range_DIF_Tail
    
    # Date_Range_DIFをロードする
    # データのファイルがあるか？
    path = Ring_Name + '_Date_Range_DIF_Strg_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print('No Date_Range_DIF_Strg_File')
        Error_f = 'No date file'
        Date_Range_CHK = 'none'
        return Date_Range_DIF, Error_f
    
    # データのファイルからDate_Range_DIFを読み込む    
    f = open(Ring_Name + "_Date_Range_DIF_Strg_File.txt", "r")
    Date_Range_DIF_Strg = f.readline()
    f.close()
    
    Date_Range_DIF = Date_Range_DIF_Strg
    
    Date_Para_S = Date_Range[:14]
    
    # アボートした時刻でDate_Paraのリストをつくる
    Time_Data_Abort_List = ['Abort date para']
    datedata = Date_Para_S
    dt1 = Convert_Kblogrd_to_Dtime(datedata)
    dt0 = dt1 + datetime.timedelta(minutes = -2) # ビーム電流がゼロになった時刻の2分前
    dt2 = dt1 + datetime.timedelta(minutes = 2) # ビーム電流がゼロになった時刻の2分後
            
    Date_Para_S = '{:%Y%m%d%H%M%S}'.format(dt0) # 書式変更
    Date_Para_E = '{:%Y%m%d%H%M%S}'.format(dt2) # 書式変更
        
    Date_Range = Date_Para_S + '-' + Date_Para_E
    Date_Para = [Date_Range]
    Time_Data_Abort_List.append(Date_Range)
    
    # Tail用のパラメータ：2秒(DT_Para)毎にデータをつくる
    DT_Para ='d2'
    Date_Para_Abort = Time_Data_Abort_List[1:]
    
    # 測定データがローカルPCに既にあるかどうか確認する
    New_File = 0
    for LC_Room in List_Para:
            for Date_Range_Abort in Date_Para_Abort:
                path = LC_Room + '_' + Mode_Para + '_Data_' + Date_Range_Abort +'.txt'
                is_file = os.path.isfile(path)
                if(is_file == False):
                    New_File = 1

    # もし一つでも新しいファイルならデータを取ってローカルPCに保存する
    if(New_File == 1):
        # kekb-co-userに入ってデータを取ってファイルを保存する。タブ区切りのデータファイルが各側室毎にできる。
        Error_f = Get_kekbcouser(Ring_Name, Date_Para_Abort, DT_Para, Mode_Para, List_Para)
        if(Error_f != 'none'):
            tk.messagebox.showinfo('SSH Error', '    Restart again     ')
            return Date_Range_DIF, Error_f
        
        # ファイルをLocal PCに持ってくる。
        Get_SFTP(Ring_Name, Date_Para_Abort, Mode_Para, List_Para)
    
    # 回帰曲線用の結果があるかどうか
    New_File = 0
    for LC_Room in List_Para:
        for Date_Range in Date_Para:
            path = LC_Room + '_' + Mode_Para + '_Data_' + Date_Range + '.npy'
            is_file = os.path.isfile(path)
            if(is_file == False):
                New_File = 1
    
    path =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_' + Date_Range_DIF + '.npz'
    is_file = os.path.isfile(path)
    if(is_file == False):
        New_File = 1
 
    path =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_B_' + Date_Range_DIF + '.npz'
    is_file = os.path.isfile(path)
    if(is_file == False):
        New_File = 1
    
    if(New_File == 0): # データが既にあればDate_Rangeを返して関数を抜ける
        return Date_Range_DIF, Error_f
    
    # 20240118変更
    # ファイルの行数を計算する。
    # ndifrow = int(Cal_File_Row(Date_Para_Abort[0], DT_Para))
    
    # 各側室についてループ
    for LC_Room in List_Para:
        #各時間範囲でループ
        n_rows_s = 0
        for k in range(len(Date_Para_Abort)):
            Date_Range_Abort = Date_Para_Abort[k]
            # 20240207変更
            # ファイルの行数を計算する。
            ndifrow = int(Cal_File_Row(Date_Range_Abort, DT_Para))
            # fileからテキストデータを読み込む
            filename = LC_Room + '_' + Mode_Para + '_Data_' + Date_Range_Abort +'.txt'
            
            # 読み込むテキストのリストの定義
            content =[]
            
            with open (filename, encoding = 'utf8', newline = '') as f:
                csvreader = csv.reader(f, delimiter = '\t') # csvreaderにリストとして読み込む。タブ区切り。
                n_row = 0 #もとのデータの行数を求める変数：初期値
                for row in csvreader:
                    if(n_row == 0):
                        nitem = len(row)
        
                    list_row_filtered = filter(None, row)

                    row_list = list(list_row_filtered)

                    while(len(row_list) < nitem):
                        row_list = np.append(row_list, ['0']).tolist()
                                
                    content = content + row_list #contentは1行のリストになる
                    n_row += 1 #もとのcsvファイル(リスト)の行数となる
                    # 20240118変更
                    if(n_row == ndifrow + 2):
                        break
            
            #もとのデータの列数
            n_column = int(len(content)/n_row) 
            
            data = np.array(content) # もとのデータのリスト'content'をベクトルのデータ'data'にする
            d = data.reshape(n_row, n_column) #ベクトルのデータ'data'をn_row行、n_column列の行列"d"にする
            
            # アボート後の時刻、ビーム電流、圧力のベクトルを作る  
            Title_Name = d[0] # タイトルの名前の1次元ベクトル

            # タイトルの名前のブーリアンリスト：最後に'PRES'がある列
            Record_Name_b =  [bool(Title_Name[i][-4:] == 'PRES') for i in range(len(Title_Name))]
            # タイトルの名前のブーリアンリスト：'time'がある列:時刻
            Time_Name_b =  [bool(Title_Name[i] == 'time') for i in range(len(Title_Name))]
            # タイトルの名前のブーリアンリスト：BEAMCURRENTがある列：ビーム電流
            Beam_Name_b =  [bool(Title_Name[i] == BEAMCURRENT) for i in range(len(Title_Name))]

            # タイトルの名前の列をブーリアンで抜き出す：最後に'PRES'がある列
            Record_Data = d[:, Record_Name_b]
            # レコードの数 = 列の数
            N_Records = len(Record_Data[0])
            # タイトルの名前の列をブーリアンで抜き出す:'time'がある列
            Time_Data = d[:, Time_Name_b]
            # タイトルの名前の列をブーリアンで抜き出す:BEAMCURRENTがある列
            Beam_Data = d[:, Beam_Name_b]
            
            if(k == 0):
                Beam_Data_Tail = Beam_Data[0]
                Time_Data_Tail = Time_Data[0]
                Record_Data_Tail = Record_Data[0]
                Record_Data_Tail_Norm = Record_Data[0]
                Step_Tail = ['Time Step']
            
            #アボートしたタイミングを見つける：2秒(DT_Para)の単位で
            # 条件に合うデータを抜き出す
            j = 0
            for i in range(2, n_row - 1):
                # iのビーム電流が1mA未満
                if (float(Beam_Data[i]) <= 1.):
                    Time_Data_Tail = np.vstack((Time_Data_Tail, Time_Data[i + 1]))
                    Beam_Data_Tail = np.vstack((Beam_Data_Tail, Beam_Data[i + 1]))
                    Record_Data_Tail = np.vstack((Record_Data_Tail, Record_Data[i + 1]))
                    if(j == 0):
                        rmax = Record_Data[i - 1] # ビームがある時

                    Record_Data_Tail_Norm = np.vstack((Record_Data_Tail_Norm, Record_Data[i + 1]))
                    Step_Tail = np.vstack((Step_Tail, str(j * int(DT_Para[-1:])))) # 秒単位にするため
                    j += 1
                    n_rows_s += 1
            
            # 各時刻のデータを、ベース(rmin)を差し引いて規格化する
            rmin = 1.e-8
            for i in range(n_rows_s - j + 1, n_rows_s + 1):
                for m in range(len(Record_Data_Tail[i])):
                    if(float(rmax[m]) - rmin != 0) and (float(rmax[m]) != float(Record_Data_Tail_Norm[n_rows_s, m])):
                        Record_Data_Tail_Norm[i, m] = "{:.5f}".format((float(Record_Data_Tail_Norm[i, m]) - rmin) 
                                                                      / (float(rmax[m]) -rmin))
                    else:
                        Record_Data_Tail_Norm[i, m] = "{:.5f}".format(0.0)
            
        # 各側室のMSE計算に使用するレコードデータ(規格化した圧力)の配列を保存する
        Record_Data_Tail_Norm_File_Name =  LC_Room + '_' + Mode_Para + '_Data_Norm_' + Date_Range + '.npy'
        np.save(Record_Data_Tail_Norm_File_Name, Record_Data_Tail_Norm)
        Record_Data_Tail_File_Name =  LC_Room + '_' + Mode_Para + '_Data_' + Date_Range + '.npy'
        np.save(Record_Data_Tail_File_Name, Record_Data_Tail)

        # 20240222追加　ローカルリストの削除
        del content, row_list, data, d
        del Title_Name, Time_Name_b, Beam_Name_b, Record_Name_b, Record_Data, N_Records, Time_Data, Beam_Data
        del Beam_Data_Tail, Time_Data_Tail, Record_Data_Tail, Record_Data_Tail_Norm

    # リングのアボート後の時間ステップの配列を保存する
    Time_Step_File_Name = Ring_Name + '_' + Mode_Para + '_Time_Step_' + Date_Range_DIF + '.npy'
    np.save(Time_Step_File_Name, Step_Tail)
    
    # リングのすべてのデータをまとめる
    # まず、アボート後(Tail)の時間ステップを読み込む
    Time_Step_File_Name = Ring_Name + '_' + Mode_Para + '_Time_Step_' + Date_Range_DIF + '.npy'
    All_Data_Tail = np.load(Time_Step_File_Name)
    All_Data_Tail_B = np.load(Time_Step_File_Name)
    
    # リングの各側室のレコードデータの配列を読み出してまとめる
    for LC_Room in List_Para:
        # 各側室のレコードデータの配列を読み出す
        Record_Data_Norm_File_Name =  LC_Room + '_' + Mode_Para + '_Data_Norm_' + Date_Range + '.npy'
        Record_Data_List = np.load(Record_Data_Norm_File_Name)
        Record_Data_File_Name_B =  LC_Room + '_' + Mode_Para + '_Data_' + Date_Range + '.npy'
        Record_Data_List_B = np.load(Record_Data_File_Name_B)
        
        # データの列を追加する
        try:
            All_Data_Tail = np.append(All_Data_Tail, Record_Data_List, axis = 1)
        except Exception as e:
            print(traceback.format_exc())
            print(str(e))
            Error_f = str(e)
            return Date_Range_DIF, Error_f
        
        All_Data_Tail_B = np.append(All_Data_Tail_B, Record_Data_List_B, axis = 1)
    
    # 時間ステップ、すべての側室のレコードのデータを行列として保存する。
    All_File_Name =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_' + Date_Range_DIF + '.npz'
    np.savez_compressed(All_File_Name, All_Data_Tail)
    All_File_Name_B =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_B_' + Date_Range_DIF + '.npz'
    np.savez_compressed(All_File_Name_B, All_Data_Tail_B)
    
    print(4132, Ring_Name + ' All Record Data was saved to ' + All_File_Name)
    
    # 20240222追加　ローカルリストの削除
    del Step_Tail, All_Data_Tail, All_Data_Tail_B, Record_Data_List_B, Record_Data_List
    
    return Date_Range_DIF, Error_f # Date_Rangeを返す

# ==============================================================================================================

def Define_Date_Range(Abort_Timing, List_Para, Run_Mode, Auto_Mode, Cns_Mode, Arun, Hadv, Last_Refd, Ref_Pd):
    # アボートタイミングから、各モードのDate_Rangeを決める
    # 入力されるアボートタイミングはdatetime型の日付
    
    Error_f = 'none'
    No_Beam = 0
    
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    
    # Abort Timingの書式変換
    date_dt = datetime.datetime.strptime(Abort_Timing, '%Y-%m-%d %H:%M:%S')
    
    # STDデータ終わりはアボート時のLast_Refd日前
    day_advance = -Last_Refd
    dtime = date_dt + datetime.timedelta(days = day_advance)
    Date_Para_E_STD = '{:%Y%m%d}'.format(dtime) + '000000' # 書式変更
    # STDデータ始まりはさらにRef_Pd日前
    day_advance = -Last_Refd - Ref_Pd
    dtime = date_dt + datetime.timedelta(days = day_advance)
    Date_Para_S_STD = '{:%Y%m%d}'.format(dtime) + '000000' # 書式変更
    Date_Range_STD = Date_Para_S_STD + '-' + Date_Para_E_STD
    
    # Date_Range_STDをファイルに保存する
    f = open(Ring_Name + '_Date_Range_STD_File.txt', "w")
    f.write(Date_Range_STD)
    f.close()
    print(Date_Range_STD, ' was saved to Ring_Name + Date_Range_STD_File.txt')
    
    # DIF, CHKデータの最後の時刻
    # アボートタイミングの3分後-->2分後 #20240207変更
    minit_advance = 2
    dtime = date_dt + datetime.timedelta(minutes = minit_advance)
    Date_Para_E_DIF = '{:%Y%m%d%H%M%S}'.format(dtime) # 書式変更
    Date_Para_E_CHK = Date_Para_E_DIF
    
    # 各リングのビーム電流のレコード名
    if(Ring_Name == 'LER'):
        BEAMCURRENT = 'BMLDCCT:CURRENT'
    elif(Ring_Name == 'HER'):
        BEAMCURRENT = 'BMHDCCT:CURRENT'

    # STDのためのアボート時刻 -------------------------------------------------------------------------------
    Date_Range = Date_Range_STD
    
    Mode_Para = 'STD_Strg'
    # 1分毎
    DT_Para = 'd60'

    Date_Para = [Date_Range] # リストにする
    # ビーム電流を取ってくるshファイル名
    Beam_Para = ['BEAM']
        
    # 測定データがローカルPCに既にあるかどうか確認する
    New_File = 0
    for Date_Range in Date_Para:
        path = Ring_Name + '_' + Mode_Para + '_Beam_Data_' + Date_Range +'.txt'
        is_file = os.path.isfile(path)
        if(is_file == False):
            New_File = 1
    
    # もし一つでも新しいファイルならデータを取ってきてローカルPCに保存する
    if(New_File == 1):
        # kekb-co-userに入ってビームデータを取ってファイルを保存する。タブ区切りのデータファイルが各側室毎にできる。
        Error_f = Get_kekbcouser(Ring_Name, Date_Para, DT_Para, Mode_Para, Beam_Para)
        if(Error_f != 'none'):
            tk.messagebox.showinfo('SSH Error', '    Restart again     ')
            return No_Beam, Error_f
        
        # ファイルをLocal PCに持ってくる。
        Get_SFTP(Ring_Name, Date_Para, Mode_Para, Beam_Para)
        
    # テキストファイルの名前
    filename = Ring_Name + '_' + Mode_Para + '_Beam_Data_' + Date_Range +'.txt'
    
    # 20240118変更
    # ファイルの行数を計算する。
    nstdrow = int(Cal_File_Row(Date_Range, DT_Para))
    
    # データのリストの定義
    content =[]
    with open (filename, encoding = 'utf8', newline = '') as f:
        csvreader = csv.reader(f, delimiter = '\t') # csvreaderにリストとして読み込む。タブ区切り。
        n_row = 0 #もとのデータの行数を求める変数：初期値
        for row in csvreader:
            if(n_row == 0):
                nitem = len(row)
        
            list_row_filtered = filter(None, row)

            row_list = list(list_row_filtered)

            while(len(row_list) < nitem):
                row_list = np.append(row_list, ['0']).tolist()
                                
            content = content + row_list #contentは1行のリストになる
            n_row += 1 #もとのcsvファイル(リスト)の行数となる
            # 20240118変更
            if(n_row == nstdrow + 2):
                break
    
    if (n_row == 0): # 'BEAM'のパラメータ
        print('3992 ',content)
        content = ['time', 'BMLDCCT:CURRENT', 'BMLDCCT:LIFE', 'BMHDCCT:CURRENT', 'BMHDCCT:LIFE', 'COpLER:BEAM:LIFE',
                'COeHER:BEAM:LIFE', 'CGLINJ:BKSEL:NOB_SET', 'CGHINJ:BKSEL:NOB_SET']
        n_row = 1
        print('3996 ',content)
            
    #データ表の列数
    n_column = int(len(content)/n_row) 

    data = np.array(content) # もとのデータのリスト'content'をベクトルのデータ'data'にする
    d = data.reshape(n_row, n_column) #ベクトルのデータ'data'をn_row行、n_column列の行列"d"にする
    
    # データが無かったら(kblogrdが止まっていたら）
    if(n_row == 1):
        # 最初の時刻
        time_ex = Convert_Kblogrd_to_Excel(Date_Para_S_STD)
        d = np.vstack((d, [time_ex, 0, 0, 0, 0, 0, 0, 0, 0]))
    
    # Data_Rangeの最後の時刻じゃなかったら、それまで0を入れる
    # 最後のデータの時刻
    time_end = Convert_Excel_to_Kblogrd(d[-1][0])

    while(float(time_end) < float(Date_Para_E_STD)):
        time_end_dt = Convert_Kblogrd_to_Dtime(time_end)
        time_end_dt = time_end_dt + datetime.timedelta(minutes = 1)
        time_end_kb = '{:%Y%m%d%H%M%S}'.format(time_end_dt)
        time_end_ex = Convert_Kblogrd_to_Excel(time_end_kb) 
        d = np.vstack((d, [time_end_ex, 0, 0, 0, 0, 0, 0, 0, 0]))
        # 最後のデータの時刻
        time_end = Convert_Excel_to_Kblogrd(d[-1][0])
    
    # 時刻、ビーム電流のベクトルを作る(Date_Rangeに範囲)
    Title_Name = d[0] # タイトルの名前の1次元ベクトル

    # タイトルの名前のブーリアンリスト：'time'がある列:時刻
    Time_Name_b =  [bool(Title_Name[i] == 'time') for i in range(len(Title_Name))]
    # タイトルの名前のブーリアンリスト：BEAMCURRENTがある列：ビーム電流
    Beam_Name_b =  [bool(Title_Name[i] == BEAMCURRENT) for i in range(len(Title_Name))]

    # タイトルの名前の列をブーリアンで抜き出す:'time'がある列
    Time_Data = d[:, Time_Name_b]
    # タイトルの名前の列をブーリアンで抜き出す:BEAMCURRENTがある列
    Beam_Data = d[:, Beam_Name_b]
    
    # アボートした時刻のリストをつくる。 1分の精度で。 
    # ビーム電流の要素をfloatにする:1番目のデータ(2個目のデータ)以降のベクトルにして
    Beam_Data_f =np.array([float(s) for s in np.delete(Beam_Data, 0)])

    # データ取得期間のビーム電流の最大値
    Beam_Max = np.max(Beam_Data_f)
    
    # STD用のアボートした時刻のリストをつくる。
    # 列の名前を最初の行に入れる
    Beam_Data_Abort_STD = Beam_Data[0]
    Time_Data_Abort_STD = Time_Data[0]

    # アボートした時のデータを抜き出す
    # アボート時のビーム電流の、範囲内最大電流に対する割合
    br = 0.3
        
    # 抜き出した表の行数
    n_row_s = 0 
    for i in range(2, n_row):
        # i-1のビーム電流が最大電流のbr％以上で、iのビーム電流が1mA未満で、最大ビーム電流が50mA以上
        if (float(Beam_Data[i - 1]) >= (Beam_Max * br)) and (float(Beam_Data[i]) <= 1.) and (Beam_Max >= 50.):
            Time_Data_Abort_STD = np.vstack((Time_Data_Abort_STD, Time_Data[i]))
            Beam_Data_Abort_STD = np.vstack((Beam_Data_Abort_STD, Beam_Data[i]))
            n_row_s += 1 # 抜き出した表の行数
    
    # STD用のアボートした時刻の行列を保存する
    # Arrayのまま保存する
    # データがすべてゼロ(ビームがない）なら、タイトル行だけ
    np.save(Ring_Name + 'Time_Data_Abort_STD_File.npy', Time_Data_Abort_STD)
    
    # 20240131変更
    No_Beam_std = 0
    Error_fstd = 'none'
    # 最大ビーム電流が50mA未満でタイトル行だけだったら(STDでアボートが無い、と見なされたら)
    if(Time_Data_Abort_STD.shape[0] == 1) and (Beam_Max < 50.):
        No_Beam_std = 1
    if(Time_Data_Abort_STD.shape[0] == 1) and (Beam_Max >= 50.):
        Error_f = 'No_STD_Abort'
        Error_fstd = Error_f
    
    # print(4132, 'No_Beam_std=', No_Beam_std, 'Error_f =', Error_f, 'Cns_Mode =', Cns_Mode)
    
    # CHK, DIFのためのアボート時刻 -----------------------------------------------------------------------------
    
    Mode_Para = 'DIF_Strg'
    DT_Para = 'd30'
        
    # 最初の遡る時間
    hour_advance = -Hadv
    constanth = 0
    # 単なるManual(Autoのスレッドは走っていない)
    if(Run_Mode == 'Manual') and (Arun == 0):
        hour_advance = -Hadv # Hadv 時間遡る
        constanth = 0 # Abortが無ければさらに遡る
    # 単なるAuto-Trigger(定期の時ではない）
    elif(Run_Mode == 'Auto') and (Auto_Mode == 'Abt Trg') and (Cns_Mode == 0):
        hour_advance = -Hadv # Hadv 時間遡る
        constanth = 0 # Abortが無ければさらに遡る
    # Manualだが、Autoのスレッドは走っている
    elif(Run_Mode == 'Manual') and (Arun == 1):
        hour_advance = -Hadv
        # 20240212 0に変更
        constanth = 0 # Abortが無ければさらに遡る
    # 単なるAuto-Constの時
    elif(Run_Mode == 'Auto') and (Auto_Mode == 'Cns Intr'):
        hour_advance = -Hadv
        constanth = 1
    # Auto-Triggerで定期の時
    elif(Run_Mode == 'Auto') and (Auto_Mode == 'Abt Trg') and (Cns_Mode == 1):
        hour_advance = -Hadv
        constanth = 1
    
    while True:
        # 最初の日時を時刻変数にする
        # Date_Para_E_CHK、DIFはアボートタイミングの3分後-->2分後 #20240207
        if(Mode_Para[0:3] == 'CHK'):
            datestart = Date_Para_E_CHK
            d1 = Convert_Kblogrd_to_Dtime(datestart)
            d0 = d1 + datetime.timedelta(hours = hour_advance) # ビーム電流がゼロになった時刻のhour_advance時間前
            Date_Para_S_CHK = '{:%Y%m%d%H%M%S}'.format(d0) # 書式変更
            Date_Range_CHK = Date_Para_S_CHK + '-' + Date_Para_E_CHK
            Date_Range = Date_Range_CHK
        elif(Mode_Para[0:3] == 'DIF'):
            datestart = Date_Para_E_DIF
            d1 = Convert_Kblogrd_to_Dtime(datestart)
            d0 = d1 + datetime.timedelta(hours = hour_advance) # ビーム電流がゼロになった時刻のhour_advance時間前
            Date_Para_S_DIF = '{:%Y%m%d%H%M%S}'.format(d0) # 書式変更
            Date_Range_DIF = Date_Para_S_DIF + '-' + Date_Para_E_DIF
            Date_Range = Date_Range_DIF
            
        Date_Para = [Date_Range] # リストにする
        Beam_Para = ['BEAM']
        
        # 測定データがローカルPCに既にあるかどうか確認する
        New_File = 0
        for Date_Range in Date_Para:
            path = Ring_Name + '_' + Mode_Para + '_Beam_Data_' + Date_Range +'.txt'
            is_file = os.path.isfile(path)
            if(is_file == False):
                New_File = 1
    
        # もし一つでも新しいファイルならデータを取ってきてローカルPCに保存する
        if(New_File == 1):
            # kekb-co-userに入ってビームデータを取ってファイルを保存する。タブ区切りのデータファイルが各側室毎にできる。
            Error_f = Get_kekbcouser(Ring_Name, Date_Para, DT_Para, Mode_Para, Beam_Para)
            if(Error_f != 'none'):
                tk.messagebox.showinfo('SSH Error', '    Restart again     ')
                return No_Beam, Error_f
        
            # ファイルをLocal PCに持ってくる。
            Get_SFTP(Ring_Name, Date_Para, Mode_Para, Beam_Para)
        
        # テキストファイルの名前
        filename = Ring_Name + '_' + Mode_Para + '_Beam_Data_' + Date_Range +'.txt'
        
        # 20240118変更
        # ファイルの行数を計算する。
        ndifrow = int(Cal_File_Row(Date_Range, DT_Para))
        
        # データのリストの定義
        content =[]
        with open (filename, encoding = 'utf8', newline = '') as f:
            csvreader = csv.reader(f, delimiter = '\t') # csvreaderにリストとして読み込む。タブ区切り。
            n_row = 0 #もとのデータの行数を求める変数：初期値
            for row in csvreader:
                if(n_row == 0):
                    nitem = len(row)
        
                list_row_filtered = filter(None, row)
                row_list = list(list_row_filtered)

                while(len(row_list) < nitem):
                    row_list = np.append(row_list, ['0']).tolist()
                                
                content = content + row_list #contentは1行のリストになる
                n_row += 1 #もとのcsvファイル(リスト)の行数となる
                # 20240118変更
                if(n_row == ndifrow + 2):
                    break
        
        if (n_row == 0): # 'BEAM'のパラメータ
            print('4093 ',content)
            content = ['time', 'BMLDCCT:CURRENT', 'BMLDCCT:LIFE', 'BMHDCCT:CURRENT', 'BMHDCCT:LIFE', 'COpLER:BEAM:LIFE',
                       'COeHER:BEAM:LIFE', 'CGLINJ:BKSEL:NOB_SET', 'CGHINJ:BKSEL:NOB_SET']
            n_row = 1
        
        #データ表の列数
        try:
            n_column = int(len(content)/n_row)
        except Exception as e:
            print(traceback.format_exc())
            print(str(e))
            Error_f = str(e)
            tk.messagebox.showinfo('Check SSH Error', Error_f)
            return No_Beam, Error_f

        data = np.array(content) # もとのデータのリスト'content'をベクトルのデータ'data'にする
        d = data.reshape(n_row, n_column) #ベクトルのデータ'data'をn_row行、n_column列の行列"d"にする
        
        # データが無かったら(kblogrdが止まっていたら）
        if(n_row == 1):
            # 最初の時刻
            time_ex = Convert_Kblogrd_to_Excel(Date_Range[:14])
            d = np.vstack((d, [time_ex, 0, 0, 0, 0, 0, 0, 0, 0]))
    
            # Data_Rangeの最後の時刻じゃなかったら、それまで0を入れる
            # 最後のデータの時刻
            time_end = Convert_Excel_to_Kblogrd(d[-1][0])
        
            while(float(time_end) < float(Date_Range[15:])):
                time_end_dt = Convert_Kblogrd_to_Dtime(time_end)
                time_end_dt = time_end_dt + datetime.timedelta(seconds = 30)
                time_end_kb = '{:%Y%m%d%H%M%S}'.format(time_end_dt)
                time_end_ex = Convert_Kblogrd_to_Excel(time_end_kb) 
                d = np.vstack((d, [time_end_ex, 0, 0, 0, 0, 0, 0, 0, 0]))
                # 最後のデータの時刻
                time_end = Convert_Excel_to_Kblogrd(d[-1][0])
    
        # 時刻、ビーム電流のベクトルを作る
        Title_Name = d[0] # タイトルの名前の1次元ベクトル

        # タイトルの名前のブーリアンリスト：'time'がある列:時刻
        Time_Name_b =  [bool(Title_Name[i] == 'time') for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：BEAMCURRENTがある列：ビーム電流
        Beam_Name_b =  [bool(Title_Name[i] == BEAMCURRENT) for i in range(len(Title_Name))]

        # タイトルの名前の列をブーリアンで抜き出す:'time'がある列
        Time_Data = d[:, Time_Name_b]
        # タイトルの名前の列をブーリアンで抜き出す:BEAMCURRENTがある列
        Beam_Data = d[:, Beam_Name_b]
    
        # アボートした時刻のリストをつくる。 30秒の精度で。 
        # ビーム電流の要素をfloatにする:1番目のデータ(2個目のデータ)以降のベクトルにして
        Beam_Data_f =np.array([float(s) for s in np.delete(Beam_Data, 0)])

        # データ取得期間のビーム電流の最大値
        Beam_Max = np.max(Beam_Data_f)
        
        # アボートした時刻のリストをつくる。
        # 列の名前を最初の行に入れる
        Beam_Data_Abort_DIF = Beam_Data[:2]
        Time_Data_Abort_DIF = Time_Data[:2]

        # アボートした時のデータを抜き出す
        # アボート時のビーム電流の割合
        br = 0.3

        # 抜き出した表の行数
        n_row_s = 0 
        for i in range(2, n_row):
            # i-1のビーム電流が最大電流のbr以上で、iのビーム電流が1mA未満で、最大ビーム電流が50mA以上
            if (float(Beam_Data[i - 1]) >= (Beam_Max * br)) and (float(Beam_Data[i]) <= 1.) and (Beam_Max >= 50):
                Time_Data_Abort_DIF = np.vstack((Time_Data_Abort_DIF, Time_Data[i]))
                Beam_Data_Abort_DIF = np.vstack((Beam_Data_Abort_DIF, Beam_Data[i]))
                n_row_s += 1 # 抜き出した表の行数
        
        No_Beam = 0
        if(constanth == 1):
            # 指定時間ビームがなければ(50mA未満)抜ける
            # 20240206変更
            if(Beam_Max < 50.) or (No_Beam_std == 1):
                No_Beam = 1
            break
        elif(constanth == 0):
            # 指定時間ビームがなければ(50mA未満)抜ける
            # 20240206変更
            if(Beam_Max < 50.) or (No_Beam_std == 1):
                No_Beam = 1
                break
            # アボートが2個以上あれば抜ける
            elif(n_row_s > 1):
                break
            # 指定時間x4 時間アボートが無ければ抜ける
            elif(hour_advance < -4 * Hadv):
                break
            # 指定時間からHadv時間さらに遡る
            else:
                hour_advance = hour_advance - Hadv
        
    # print(4330, No_Beam, n_row_s)
    # print(Time_Data_Abort_DIF)
    
    # 最後のフィルの確認で、No_Beam = 0(ビーム有)で、
    # もしDete_Para_Eから10分前までにアボート時刻がなければ、3分前にアボートしたことにする。
    # ビーム電流が小さかった時の対処
    Shift = 0
    # 現在の最後のアボート時刻
    lastdate = Time_Data_Abort_DIF[-1][0]

    d1 = Convert_Excel_to_Dtime(lastdate)
    d2 = Convert_Kblogrd_to_Dtime(Date_Para_E_DIF)
    d3 = d2 - d1
        
    d3_sec = d3.total_seconds()
        
    # もし10分以上の差があったら３分前のデータを最後に加える(d30として)
    Shift = 0
    if(d3_sec > 10. * 60.):
        Time_Data_Abort_DIF = np.vstack((Time_Data_Abort_DIF, Time_Data[-7:-6]))
        Beam_Data_Abort_DIF = np.vstack((Beam_Data_Abort_DIF, Beam_Data[-7:-6]))
        lastdate = Time_Data_Abort_DIF[-1][0]
        Shift = 1
    
    # print(4383, Time_Data_Abort_DIF)
        
    # ビームがあって、constanth =1で、DIFでアボートがなかった時
    Shift2 = 0
    # print(4356, Time_Data_Abort_DIF[-2-Shift:-1][0, 0])
    if(Time_Data_Abort_DIF[-2 - Shift:-1][0, 0] == 'time'):
        Shift2 = 1
        # Error_f = 'No_DIF_Abort'
    
    # DIF, CHK用のアボートした時刻の行列を保存する
    # Arrayのまま保存する
    np.save(Ring_Name + 'Time_Data_Abort_CHK_File.npy', Time_Data_Abort_DIF)
    
    np.save(Ring_Name + 'Time_Data_Abort_DIF_File.npy', Time_Data_Abort_DIF)
    
    # CHK_StrgのためのDate_Para_S
    # 20240202変更
    datedata = Time_Data_Abort_DIF[-2 - Shift + Shift2:-1][0, 0] #リスト最後の一つ前の時刻
    
    # print(4372, Shift, Shift2, datedata)
    
    dtt0 = Convert_Excel_to_Dtime(datedata)
    dtt1 = dtt0 + datetime.timedelta(minutes = 2) # ビーム電流がゼロになった時刻の2分後
    dtt2 = '{:%Y%m%d%H%M%S}'.format(dtt1) # Excel形式の日付に書式変更
    Date_Para_S_CHK_Strg = dtt2

    Date_Range_CHK_Strg = Date_Para_S_CHK_Strg + '-' + Date_Para_E_CHK
    
    # CHK_TailのためのDate_Para_S
    datedata = Time_Data_Abort_DIF[-1 - Shift][0] #リストの最後
    # datedata = Time_Data_Abort_DIF[-1][0] #リストの最後 shift無視
    Date_Para_S_CHK_Tail = Convert_Excel_to_Kblogrd(datedata)
    Date_Range_CHK_Tail = Date_Para_S_CHK_Tail + '-' + Date_Para_E_CHK
    
    # DIF_StrgのためのDate_Para_S
    # 20240202変更
    datedata = Time_Data_Abort_DIF[-2 - Shift + Shift2:-1][0, 0] #リスト最後の一つ前の時刻

    dtt0 = Convert_Excel_to_Dtime(datedata)
    dtt1 = dtt0 + datetime.timedelta(minutes = 2) # ビーム電流がゼロになった時刻の2分後
    dtt2 = '{:%Y%m%d%H%M%S}'.format(dtt1) # Excel形式の日付に書式変更
    Date_Para_S_DIF_Strg = dtt2

    Date_Range_DIF_Strg = Date_Para_S_DIF_Strg + '-' + Date_Para_E_DIF

    # DIF_TailのためのDate_Para_S
    datedata = Time_Data_Abort_DIF[-1 - Shift][0] #リストの最後
    # datedata = Time_Data_Abort_DIF[-1][0] #リストの最後   shift無視
    Date_Para_S_DIF_Tail = Convert_Excel_to_Kblogrd(datedata)
    Date_Range_DIF_Tail = Date_Para_S_DIF_Tail + '-' + Date_Para_E_DIF
    
    # Date_Range_DIF_Strg, Tailをファイルに保存する
    f = open(Ring_Name + '_Date_Range_DIF_Strg_File.txt', "w")
    f.write(Date_Range_DIF_Strg)
    f.close()
    print(Date_Range_DIF_Strg, ' was saved to Ring_Name + Date_Range_DIF_Strg_File.txt')
    
    f = open(Ring_Name + '_Date_Range_DIF_Tail_File.txt', "w")
    f.write(Date_Range_DIF_Tail)
    f.close()
    print(Date_Range_DIF_Tail, ' was saved to Ring_Name + Date_Range_DIF_Tail_File.txt')
    
    # Date_Range_CHK_Strg, Tailをファイルに保存する
    f = open(Ring_Name + '_Date_Range_CHK_Strg_File.txt', "w")
    f.write(Date_Range_CHK_Strg)
    f.close()
    print(Date_Range_CHK_Strg, ' was saved to Ring_Name + Date_Range_CHK_Strg_File.txt')
    
    f = open(Ring_Name + '_Date_Range_CHK_Tail_File.txt', "w")
    f.write(Date_Range_CHK_Tail)
    f.close()
    print(Date_Range_CHK_Tail, ' was saved to Ring_Name + Date_Range_CHK_Tail_File.txt') 
    
    # NB = 0 の時CHK_Strg期間のビーム電流確認: 最大電流が50mA未満ならビーム無しとるす
    if(No_Beam == 0):
        Date_Para = [Date_Range_CHK_Strg]
        Beam_Para = ['BEAM']
        Mode_Para = 'CHK_Strg'
        Mode_Para_c = 'DIF_Strg' #20230816+c
        DT_Para = 'd30'
        
        # 測定データがローカルPCに既にあるかどうか確認する
        New_File = 0
        for Date_Range in Date_Para:
            path = Ring_Name + '_' + Mode_Para_c + '_Beam_Data_' + Date_Range +'.txt' #20230816+c
            is_file = os.path.isfile(path)
            if(is_file == False):
                New_File = 1
    
        # もし一つでも新しいファイルならデータを取ってきてローカルPCに保存する
        if(New_File == 1):
            # kekb-co-userに入ってビームデータを取ってファイルを保存する。タブ区切りのデータファイルが各側室毎にできる。
            Error_f = Get_kekbcouser(Ring_Name, Date_Para, DT_Para, Mode_Para_c, Beam_Para)
            if(Error_f != 'none'):
                tk.messagebox.showinfo('SSH Error', '    Restart again     ')
                return No_Beam, Error_f
            
            # ファイルをLocal PCに持ってくる。
            Get_SFTP(Ring_Name, Date_Para, Mode_Para_c, Beam_Para)
        
        # 20240118変更
        # ファイルの行数を計算する。
        nchkrow = int(Cal_File_Row(Date_Range, DT_Para))
        
        # テキストファイルの名前
        filename = Ring_Name + '_' + Mode_Para_c + '_Beam_Data_' + Date_Range +'.txt' #20230816+c
        # データのリストの定義
        content =[]
        with open (filename, encoding = 'utf8', newline = '') as f:
            csvreader = csv.reader(f, delimiter = '\t') # csvreaderにリストとして読み込む。タブ区切り。
            n_row = 0 #もとのデータの行数を求める変数：初期値
            for row in csvreader:
                if(n_row == 0):
                    nitem = len(row)
        
                list_row_filtered = filter(None, row)
                row_list = list(list_row_filtered)

                while(len(row_list) < nitem):
                    row_list = np.append(row_list, ['0']).tolist()
                                
                content = content + row_list #contentは1行のリストになる
                n_row += 1 #もとのcsvファイル(リスト)の行数となる
                # 20240118変更
                if(n_row == nchkrow + 2):
                    break
                
        #データ表の列数
        n_column = int(len(content)/n_row) 

        data = np.array(content) # もとのデータのリスト'content'をベクトルのデータ'data'にする
        d = data.reshape(n_row, n_column) #ベクトルのデータ'data'をn_row行、n_column列の行列"d"にする
        
        # データが無かったら(kblogrdが止まっていたら）
        if(n_row == 1):
            # 最初の時刻
            time_ex = Convert_Kblogrd_to_Excel(Date_Range[:14])
            d = np.vstack((d, [time_ex, 0, 0, 0, 0, 0, 0, 0, 0]))
    
            # Data_Rangeの最後の時刻じゃなかったら、それまで0を入れる
            # 最後のデータの時刻
            time_end = Convert_Excel_to_Kblogrd(d[-1][0])
        
            while(float(time_end) < float(Date_Range[15:])):
                time_end_dt = Convert_Kblogrd_to_Dtime(time_end)
                time_end_dt = time_end_dt + datetime.timedelta(seconds = 30)
                time_end_kb = '{:%Y%m%d%H%M%S}'.format(time_end_dt)
                time_end_ex = Convert_Kblogrd_to_Excel(time_end_kb) 
                d = np.vstack((d, [time_end_ex, 0, 0, 0, 0, 0, 0, 0, 0]))
                # 最後のデータの時刻
                time_end = Convert_Excel_to_Kblogrd(d[-1][0])
    
        # 時刻、ビーム電流のベクトルを作る
        Title_Name = d[0] # タイトルの名前の1次元ベクトル

        # タイトルの名前のブーリアンリスト：'time'がある列:時刻
        Time_Name_b =  [bool(Title_Name[i] == 'time') for i in range(len(Title_Name))]
        # タイトルの名前のブーリアンリスト：BEAMCURRENTがある列：ビーム電流
        Beam_Name_b =  [bool(Title_Name[i] == BEAMCURRENT) for i in range(len(Title_Name))]

        # タイトルの名前の列をブーリアンで抜き出す:'time'がある列
        Time_Data = d[:, Time_Name_b]
        # タイトルの名前の列をブーリアンで抜き出す:BEAMCURRENTがある列
        Beam_Data = d[:, Beam_Name_b]

        # ビーム電流の要素をfloatにする:1番目のデータ(2個目のデータ)以降のベクトルにして
        Beam_Data_f =np.array([float(s) for s in np.delete(Beam_Data, 0)])
        # データ取得期間のビーム電流の最大値
        Beam_Max = np.max(Beam_Data_f)
    
        #ビームデータの最大値が50mA未満ならビーム無しとみなす。
        if(Beam_Max < 50.):
            No_Beam = 1
        
        #20240204追加
        if (Error_fstd == 'No_STD_Abort'):
            Error_f = 'No_STD_Abort'
            
    return No_Beam, Error_f

# ==============================================================================================================
# ==============================================================================================================

def Make_Plot_Strg(Method, Record_Name, Date_Range, Mode_Para, List_Para, Abort_Timing):
# ビームあり
    
    if(Record_Name == 'none'):
        return
    
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    if(Ring_Name == 'LER'):
        BEAMCURRENT = 'BMLDCCT:CURRENT'
        BUNCHNO = 'CGLINJ:BKSEL:NOB_SET'
    elif(Ring_Name == 'HER'):
        BEAMCURRENT = 'BMHDCCT:CURRENT'
        BUNCHNO = 'CGHINJ:BKSEL:NOB_SET'
    
    # リングのすべてのレコードのデータをロードする。
    if(Mode_Para == 'STD_Strg'):
        All_Data_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method +'_Class2_All_Record_Data_' + Date_Range + '.npz'
    else:
        Mode_Para_c = 'DIF_Strg'  #20230816+c
        All_Data_File_Name = Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_' + Date_Range + '.npz' #20230816+c
    All_Data = np.load(All_Data_File_Name)['arr_0']
    # print(4472, All_Data)
    
    if(Mode_Para == 'STD_Strg'):
        ad = All_Data[0]
        for i in range(1, All_Data.shape[0]):
            # ビーム電流が5mA以上
            if (float(All_Data[i, 1:2]) >= 5):
                ad = np.vstack((ad, All_Data[i]))
        All_Data = ad
            
    if(Mode_Para == 'STD_Strg'):
        All_Data_File_Name_Sel = Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Record_Data_Sel_' + Date_Range + '.npz'
    else:
        Mode_Para_c = 'DIF_Strg'  #20230816+c
        All_Data_File_Name_Sel = Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_Sel_' + Date_Range + '.npz' #20230816+c
    All_Data_Sel = np.load(All_Data_File_Name_Sel)['arr_0']
    
    # リングのすべてのレコードの解析結果をロードする。リスト型辞書(ビームあり)
    if(Mode_Para == 'STD_Strg'):
        All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Result_Data_' + Date_Range + '_Dict.npz'
    else:
        Mode_Para_c = 'DIF_Strg'  #20230816+c
        All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para_c + '_All_Result_Data_' + Date_Range + '_Dict.npz' #20230816+c
    All_Result = np.load(All_Result_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_Dict = All_Result.tolist()
    
    # CHKの時、STDのデータもプロットする
    # STDのデータの時間範囲(ファイルから読み込む)
    path = Ring_Name + '_Date_Range_STD_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print('No Date_Range_STD_File')
        sys.exit()   
    f = open(Ring_Name + "_Date_Range_STD_File.txt", "r")
    Date_Range_STD = f.readline()
    f.close()
    
    All_Result_STD_Data_File_Name = Ring_Name + '_STD_Strg_' + Method + '_Class2_All_Result_Data_' + Date_Range_STD + '_Dict.npz'
    All_Result_STD = np.load(All_Result_STD_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_Dict_STD = All_Result_STD.tolist()
    
    # ビーム電流、HOM、該当するレコード名の列を抽出する。
    # ビーム電流: BEAMCURRENT 列番号 = 1
    Beam_Current = All_Data[:, 1:2]
    # Beam_Current_Sel = All_Data_Sel[:, 1:2]
    # ビーム電流: BEAMCURRENT 列番号 = 0
    Beam_Current_Sel = All_Data_Sel[:, 0:1]
    
    # HOM: (I*I/Nb)^2: 列番号 = 2
    HOM_2 = All_Data[:, 2:3]
    # HOM: (I*I/Nb)^2: 列番号 = 1
    HOM_2_Sel = All_Data_Sel[:, 1:2]
    
    # タイトルの名前のブーリアンリスト：Record_Nameの列
    Record_Name_b = [bool(All_Data[0, i] == Record_Name) for i in range(len(All_Data[0]))]
    # タイトルの名前の列をブーリアンで抜き出す
    Record_Data = All_Data[:, Record_Name_b]
    # もし該当するレコード名がなかったら
    if(Record_Data.size == 0):
        print('Error: No such Record Name')
        if(Mode_Para == 'STD_Strg'):
            tk.messagebox.showinfo('No such Record Name', '    No such Record Name     ' + Record_Name)
        return

    Record_Name_b_Sel = [bool(All_Data_Sel[0, i] == Record_Name) for i in range(len(All_Data_Sel[0]))]
    # タイトルの名前の列をブーリアンで抜き出す
    Record_Data_Sel = All_Data_Sel[:, Record_Name_b_Sel] 
    
    # 該当するレコード名の解析結果を辞書から抽出する。
    list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_Dict))
    dict_search = list_search[0]
    w0 = dict_search['W0(Beam current)']
    w1 = dict_search['W1(HOM^2)']
    w2 = dict_search['W2(Base)']
    rmse = dict_search['RMSE']
    maxrse = dict_search['MaxRSE']
    
    # CHKの時、該当するレコード名のSTD解析結果を辞書から抽出する。
    list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_Dict_STD))
    dict_search = list_search[0]
    w0_std = dict_search['W0(Beam current)']
    w1_std = dict_search['W1(HOM^2)']
    w2_std = dict_search['W2(Base)']
    rmse_std = dict_search['RMSE']
    maxrse_std = dict_search['MaxRSE']
    
    # 20240120 変更
    if(float(rmse) <= 1.e-8):
        rmse = '1.e-8'
        
    # プロットの設定
    if(Mode_Para == 'STD_Strg'):
        Mode_Para_t = 'REF_Strg'
    elif(Mode_Para == 'STD_Tail'):
        Mode_Para_t = 'REF_Tail'
    else:
        Mode_Para_t = Mode_Para
    
    title = Record_Name + '\n' + Mode_Para_t + ' Abort: ' + Abort_Timing
            
    # 軸ラベルの設定
    x_label = "Beam current [mA]"
    y_label = "Pressure [Pa] (3 x reading)"

    # 軸の目盛設定
    Record_Data_f = np.array([float(s) for s in np.delete(Record_Data, 0)])
    Record_Data_f3 = Record_Data_f * 3.
    Beam_Current_f = np.array([float(s) for s in np.delete(Beam_Current, 0)])
    HOM_2_f = np.array([float(s) for s in np.delete(HOM_2, 0)])
    
    Record_Data_f_Sel = np.array([float(s) for s in np.delete(Record_Data_Sel, 0)])
    Record_Data_f3_Sel = Record_Data_f_Sel * 3.
    Beam_Current_f_Sel = np.array([float(s) for s in np.delete(Beam_Current_Sel, 0)])
    HOM_2_f_Sel = np.array([float(s) for s in np.delete(HOM_2_Sel, 0)])
    
    x_max = np.max(np.array([float(s) for s in np.delete(Beam_Current, 0)]))
    x_max = int(x_max / 100 + 1) * 100.
    # 20240202変更
    y_max1 = np.max(Beam_Current_f_Sel * float(w0_std) + HOM_2_f_Sel * float(w1_std) + float(w2_std))
    # y_max2 = np.max(np.array([float(s) for s in np.delete(Record_Data_Sel, 0)]))
    y_max2 = np.max(Record_Data_f3_Sel)
    y_max = np.maximum(y_max1, y_max2)
    y_max = np.maximum(int(y_max * 1.e8 * 1.3) / 1.e8, 5.e-8)

    x_limit = [0, x_max]
    y_limit = [0, y_max]
        
    # 凡例の設定
    y1_legend = 'Measured'
    if(Mode_Para == 'DIF_Strg'):
        y1_legend = 'Reg. REF'
    y2_legend = 'Measured'
    y3_legend = 'Reg. CHK'
    
    # テキスト
    # text1 = Mode_Para
    text1 = ' '
    
    # plotするデータ
    x1 = Beam_Current_f
    y1 = Record_Data_f3
    # 20240202変更
    if(Mode_Para == 'DIF_Strg'):
        x1 = Beam_Current_f_Sel
        y1_r = Beam_Current_f_Sel * float(w0_std) + HOM_2_f_Sel * float(w1_std) + float(w2_std)
        y1 = np.maximum(y1_r, 3.e-8)
    x2 = Beam_Current_f
    y2 = Record_Data_f3
    x3 = Beam_Current_f
    y3_r = Beam_Current_f * float(w0) + HOM_2_f * float(w1) + float(w2)
    y3 = np.maximum(y3_r, 3.e-8)
        
    # 引数
    x = [x1, x2, x3]
    y = [y1, y2, y3]
    y_legend = [y1_legend, y2_legend, y3_legend]
    cl = ["pink", "tomato", "blue"]
    if(Mode_Para == 'DIF_Strg'):
        cl = ["cyan", "red", "blue"]
    si = [24, 24, 16]
    tx = text1
    
    if(Ring_Name == 'HER'):
        fc =(0.96, 0.96, 1.0)
        al = 0.8
    elif(Ring_Name == 'LER'):
        fc = (1.0, 0.96, 0.96)
        al = 0.8
                
    # データをプロット
    show_data1(x, y, title, x_label, y_label, x_limit, y_limit, y_legend, si, cl, tx, fc, al)
    
    # 20240223追加　ローカルリストの削除
    del All_Data, All_Data_Sel, All_Result, All_Result_Dict, All_Result_STD, All_Result_Dict_STD
    del Beam_Current, Beam_Current_Sel, HOM_2, HOM_2_Sel
    del Record_Data, Record_Name_b_Sel, Record_Data_Sel
    del Record_Data_f, Record_Data_f3, Beam_Current_f, HOM_2_f
    del Record_Data_f_Sel, Record_Data_f3_Sel, Beam_Current_f_Sel, HOM_2_f_Sel 
    del x1, y1, x2, y2, x3, y3_r, y3, x, y
    
    #　メモリーの開放
    gc.collect()

# ==============================================================================================================

def Make_Plot_Strg_Time(Method, Record_Name, Date_Range, Mode_Para, List_Para, Abort_Timing):
    
    if(Record_Name == 'none'):
        return
    
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    if(Ring_Name == 'LER'):
        BEAMCURRENT = 'BMLDCCT:CURRENT'
        BUNCHNO = 'CGLINJ:BKSEL:NOB_SET'
    elif(Ring_Name == 'HER'):
        BEAMCURRENT = 'BMHDCCT:CURRENT'
        BUNCHNO = 'CGHINJ:BKSEL:NOB_SET'

    # リングのすべてのレコードのデータをロードする。
    if(Mode_Para == 'STD_Strg'):
        All_Data_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method +'_Class2_All_Record_Data_' + Date_Range + '.npz'
    else:
        Mode_Para_c = 'DIF_Strg' #20230816+c
        # All_Data_File_Name = Ring_Name + '_' + Mode_Para + '_All_Record_Data_' + Date_Range + '.npz' #20230816+c
        All_Data_File_Name = Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_' + Date_Range + '.npz' #20230816+c
    All_Data = np.load(All_Data_File_Name)['arr_0']
    
    # ビーム電流を抽出する。
    # ビーム電流: BEAMCURRENT 列番号 = 1
    Beam_Current = All_Data[:, 1:2]
    
    # 時刻を抽出する。列番号 = 0
    Time_Excel = All_Data[:, 0:1]
    
    # タイトルの名前のブーリアンリスト：Record_Nameの列
    Record_Name_b = [bool(All_Data[0, i] == Record_Name) for i in range(len(All_Data[0]))]
    # タイトルの名前の列をブーリアンで抜き出す
    Record_Data = All_Data[:, Record_Name_b]
    # もし該当するレコード名がなかったら
    if(Record_Data.size == 0):
        print('Error: No such Record Name')
        tk.messagebox.showinfo('No such Record Name', '    No such Record Name     ' + Record_Name)
        return
        
    # プロットの設定
    if(Mode_Para == 'STD_Strg'):
        Mode_Para_t = 'REF_Strg'
    elif(Mode_Para == 'STD_Tail'):
        Mode_Para_t = 'REF_Tail'
    else:
        Mode_Para_t = Mode_Para
        
    title = Record_Name + '\n' + Mode_Para_t + ' Abort: ' + Abort_Timing + '(Raw)'
            
    # 軸ラベルの設定
    x_label = "Date"
    y1_label = "Pressure [Pa] (3 x reading)"
    y2_label = "Bema current [mA]"

    # プロットするデータ
    # 圧力読み値
    Record_Data_f = np.array([float(s) for s in np.delete(Record_Data, 0)])
    # 圧力を3倍
    Record_Data_f3 = Record_Data_f * 3.
    Beam_Current_f = np.array([float(s) for s in np.delete(Beam_Current, 0)])
    # 時刻をdatetime形式にする
    Time_Excel_f = Time_Excel[1:]
    Time_Datetime_f = []
    for n in Time_Excel_f:
        s = Convert_Excel_to_Dtime(n[0])
        Time_Datetime_f.append(s)
    
    x = Time_Datetime_f
    y = Record_Data_f3
    y2 = Beam_Current_f
    
    if(Mode_Para == 'STD_Strg'):
        cl = ['tomato', 'black']
    else:
        cl = ['red', 'black']
        
    rname = ['P', 'I']
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.5), tight_layout=True)
    # fig.autofmt_xdate()
    fig.autofmt_xdate(rotation=0, ha="center")
    ax.set_title(title, fontsize = 8)
    ax.set_xlabel(x_label, size = 10, weight = 'light')
    ax.set_ylabel(y1_label, size = 10, weight = 'light')
    ax.set_xlim(Time_Datetime_f[0], Time_Datetime_f[-1])
    ax.set_ylim(0, np.max(y) * 1.2)
    ax.tick_params(direction = 'inout', length = 5, colors = 'black')
    
    ax.plot(x, y, color = cl[0], label = rname[0])
    ax.legend(loc = 'upper left', fontsize = 8)
    if(Mode_Para == 'STD_Strg'):
        # x軸のラベル
        formatter = mdates.DateFormatter("%y-%m-%d")
        ax.xaxis.set_major_formatter(formatter)
        # 1日毎にラベル
        locator = mdates.DayLocator()
        locator = mdates.AutoDateLocator(minticks=2, maxticks=4)
        ax.xaxis.set_major_locator(locator)
    elif(Mode_Para == 'CHK_Strg'):
        # x軸のラベル
        formatter = mdates.DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(formatter)
        # 1時間毎にラベル
        locator = mdates.HourLocator()
        locator = mdates.AutoDateLocator(minticks=2, maxticks=4)
        ax.xaxis.set_major_locator(locator)
    
    if(Ring_Name == 'HER'):
        ax.set_facecolor((0.96, 0.96, 1.0))
        ax.set_alpha(0.8)
    elif(Ring_Name == 'LER'):
        ax.set_facecolor((1.0, 0.96, 0.96))
        ax.set_alpha(0.8)
        
    # ビーム電流を2軸にプロットする。
    ax2 = ax.twinx()
    ax2.set_ylim(0, np.maximum(np.max(y2) * 1.2 + 1, 50.))
    ax2.plot(x, y2, color = cl[1], linestyle = "dotted", label = rname[1])
    ax2.legend(loc = 'upper right', fontsize = 8)
    ax2.set_ylabel(y2_label, size = 10, weight = 'light')
    
    # 20240223追加　ローカルリストの削除
    del All_Data, Beam_Current, Time_Excel, Record_Name_b, Record_Data
    del Record_Data_f, Record_Data_f3, Beam_Current_f, Time_Excel_f, Time_Datetime_f
    del x, y, y2
    
    # メモリーの開放
    gc.collect()
    
    return
    
# ==============================================================================================================

def Make_Plot_Strg_NB(Method, Record_Name, Date_Range, Mode_Para, List_Para, Abort_Timing):
# ビームはなし
    
    if(Record_Name == 'none'):
        return
    
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    if(Ring_Name == 'LER'):
        BEAMCURRENT = 'BMLDCCT:CURRENT'
        BUNCHNO = 'CGLINJ:BKSEL:NOB_SET'
    elif(Ring_Name == 'HER'):
        BEAMCURRENT = 'BMHDCCT:CURRENT'
        BUNCHNO = 'CGHINJ:BKSEL:NOB_SET'
    
    # リングのすべてのレコードのデータをロードする。(NBに依らず)
    if(Mode_Para == 'STD_Strg'):
        All_Data_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method +'_Class2_All_Record_Data_' + Date_Range + '.npz'
    else:
        Mode_Para_c = 'DIF_Strg' #20230816+c
        All_Data_File_Name = Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_' + Date_Range + '.npz' #20230816+c
    All_Data = np.load(All_Data_File_Name)['arr_0']
    
    # リングの解析に使用したすべてのレコードのデータをロードする。
    if(Mode_Para == 'STD_Strg'):
        All_Data_File_Name_Sel = Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Record_Data_Sel_' + \
        Date_Range + '_NB.npz'
    else:
        Mode_Para_c = 'DIF_Strg' #20230816+c
        All_Data_File_Name_Sel = Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_Sel_' + Date_Range + '.npz' #20230816+c
    All_Data_Sel = np.load(All_Data_File_Name_Sel)['arr_0']
    
    # リングのすべてのレコードの解析結果をロードする。リスト型辞書。(NB)
    if(Mode_Para == 'STD_Strg'):
        All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Result_Data_' + \
        Date_Range + '_Dict_NB.npz'
    else:
        Mode_Para_c = 'DIF_Strg' #20230816+c
        All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para_c + '_All_Result_Data_' + Date_Range + '_Dict.npz' #20230816+c
    All_Result = np.load(All_Result_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_Dict = All_Result.tolist()
    
    # CHKの時、STDのデータもプロットする
    # STDのデータの時間範囲(ファイルから読み込む)
    path = Ring_Name + '_Date_Range_STD_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print('No Date_Range_STD_File')
        sys.exit()   
    f = open(Ring_Name + "_Date_Range_STD_File.txt", "r")
    Date_Range_STD = f.readline()
    f.close()
    
    # STDの結果(NB)
    All_Result_STD_Data_File_Name = Ring_Name + '_STD_Strg_' + Method + '_Class2_All_Result_Data_' + Date_Range_STD + \
    '_Dict_NB.npz'
    All_Result_STD = np.load(All_Result_STD_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_Dict_STD = All_Result_STD.tolist()
    
    # ビーム電流、HOM、該当するレコード名の列を抽出する。
    # ビーム電流: BEAMCURRENT 列番号 = 1
    Beam_Current = All_Data[:, 1:2]
    # ビーム電流: BEAMCURRENT 列番号 = 0
    Beam_Current_Sel = All_Data_Sel[:, 0:1]
    
    # HOM: (I*I/Nb)^2: 列番号 = 2：ビームはなし
    HOM_2 = All_Data[:, 2:3]
    # HOM: (I*I/Nb)^2: 列番号 = 1: ビームはなし
    HOM_2_Sel = All_Data_Sel[:, 1:2]
    
    # タイトルの名前のブーリアンリスト：Record_Nameの列
    Record_Name_b = [bool(All_Data[0, i] == Record_Name) for i in range(len(All_Data[0]))]
    # タイトルの名前の列をブーリアンで抜き出す
    Record_Data = All_Data[:, Record_Name_b]
    # もし該当するレコード名がなかったら
    if(Record_Data.size == 0):
        print('Error: No such Record Name')
        tk.messagebox.showinfo('No such Record Name', '    No such Record Name     ' + Record_Name)
        return
    
    Record_Name_b_Sel = [bool(All_Data_Sel[0, i] == Record_Name) for i in range(len(All_Data_Sel[0]))]
    # タイトルの名前の列をブーリアンで抜き出す
    Record_Data_Sel = All_Data_Sel[:, Record_Name_b_Sel] 
    
    # 該当するレコード名の解析結果を辞書から抽出する。
    list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_Dict))
    dict_search = list_search[0]
    w0 = dict_search['W0(Beam current)']
    w1 = dict_search['W1(HOM^2)']
    w2 = dict_search['W2(Base)']
    rmse = dict_search['RMSE']
    maxrse = dict_search['MaxRSE']
    
    # CHKの時、該当するレコード名のSTD解析結果を辞書から抽出する。
    list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_Dict_STD))
    dict_search = list_search[0]
    w0_std = dict_search['W0(Beam current)']
    w1_std = dict_search['W1(HOM^2)']
    w2_std = dict_search['W2(Base)']
    rmse_std = dict_search['RMSE']
    maxrse_std = dict_search['MaxRSE']
    
    # 20240120 変更
    if(float(rmse) <= 1.e-8):
        rmse = '1.e-8'
        
    # プロットの設定
    if(Mode_Para == 'STD_Strg'):
        Mode_Para_t = 'REF_Strg'
    elif(Mode_Para == 'STD_Tail'):
        Mode_Para_t = 'REF_Tail'
    elif(Mode_Para == 'DIF_Strg'):
        Mode_Para_t = 'CHK_Strg'
    elif(Mode_Para == 'DIF_Tail'):
        Mode_Para_t = 'CHK_Tail'
    else:
        Mode_Para_t = Mode_Para
    
    title = Record_Name + '\n' + Mode_Para_t + ' Abort: ' + Abort_Timing
            
    # 軸ラベルの設定
    x_label = "Sampling step"
    y_label = "Pressure [Pa] (3 x reading)"

    # 軸の目盛設定
    Record_Data_f = np.array([float(s) for s in np.delete(Record_Data, 0)])
    Record_Data_f3 = Record_Data_f * 3.
    Beam_Current_f = np.array([float(s) for s in np.delete(Beam_Current, 0)])
    HOM_2_f = np.array([float(s) for s in np.delete(HOM_2, 0)])
    
    Record_Data_f_Sel = np.array([float(s) for s in np.delete(Record_Data_Sel, 0)])
    Record_Data_f3_Sel = Record_Data_f_Sel * 3.
    Beam_Current_f_Sel = np.array([float(s) for s in np.delete(Beam_Current_Sel, 0)])
    HOM_2_f_Sel = np.array([float(s) for s in np.delete(HOM_2_Sel, 0)])
    
    # HOM_2_f_STD = np.copy(HOM_2_f_Sel)
    # HOM_2_fを平均値とする。
    # for i in range(len(Beam_Current_f_Sel)):
    #     Beam_Current_f_Sel[i] = float(i) # データの数
    #     HOM_2_f_Sel[i] = 1. * float(w2) # DIFの平均値
    #     HOM_2_f_STD[i] = 1. * float(w2_std) # STDの平均値
    
    # x_max = np.max(np.array([float(s) for s in np.delete(Beam_Current, 0)]))
    x_max = np.max(Beam_Current_f_Sel)
    x_max = int(x_max / 100 + 1) * 100.
    # 20240202変更
    y_max1 = np.max(Beam_Current_f_Sel * float(w0_std) + HOM_2_f_Sel * float(w1_std) + float(w2_std))
    y_max2 = np.max(Record_Data_f3_Sel)
    y_max = np.maximum(y_max1, y_max2)
    # y_max = int(y_max * 1.e8 * 1.2) / 1.e8
    y_max = y_max * 1.2
    # print(4983, Mode_Para, y_max)

    x_limit = [0, x_max]
    y_limit = [0, y_max]
        
    # 凡例の設定
    y1_legend = 'Measured'
    if(Mode_Para == 'DIF_Strg'):
        y1_legend = 'Reg. REF'
    y2_legend = 'Measured'
    y3_legend = 'Reg. CHK'
    
    # テキスト
    text1 = ' '
    
    # 202401変更
    # plotするデータ
    x1 = Beam_Current_f_Sel
    y1 = Record_Data_f3_Sel
    if(Mode_Para == 'DIF_Strg'):
        y1_r = Beam_Current_f_Sel * float(w0_std) + HOM_2_f_Sel * float(w1_std) + float(w2_std)
        y1 = np.maximum(y1_r, 3.e-8)
    x2 = Beam_Current_f_Sel
    y2 = Record_Data_f3_Sel
    x3 = Beam_Current_f_Sel
    y3_r = Beam_Current_f_Sel * float(w0) + HOM_2_f_Sel * float(w1) + float(w2)
    # y3_r = HOM_2_f_Sel # 平均値
    y3 = np.maximum(y3_r, 3.e-8)
        
    # 引数
    x = [x1, x2, x3]
    y = [y1, y2, y3]
    y_legend = [y1_legend, y2_legend, y3_legend]
    cl = ["pink", "tomato", "blue"]
    if(Mode_Para == 'DIF_Strg'):
        cl = ["cyan", "red", "blue"]
    si = [24, 24, 16]
    tx = text1
    
    if(Ring_Name == 'HER'):
        fc =(0.96, 0.96, 1.0)
        al = 0.8
    elif(Ring_Name == 'LER'):
        fc = (1.0, 0.96, 0.96)
        al = 0.8
        
    # データをプロット
    show_data1(x, y, title, x_label, y_label, x_limit, y_limit, y_legend, si, cl, tx, fc, al)

    # 20240223追加　ローカルリストの削除
    del All_Data, All_Data_Sel, All_Result, All_Result_Dict, All_Result_STD, All_Result_Dict_STD
    del Beam_Current, Beam_Current_Sel, HOM_2, HOM_2_Sel
    del Record_Data, Record_Name_b_Sel, Record_Data_Sel
    del Record_Data_f, Record_Data_f3, Beam_Current_f, HOM_2_f
    del Record_Data_f_Sel, Record_Data_f3_Sel, Beam_Current_f_Sel, HOM_2_f_Sel 
    del x1, y1, x2, y2, x3, y3_r, y3, x, y
    
    #　メモリーの開放
    gc.collect()
    
# ==============================================================================================================

def Make_Plot_Tail(Method, Record_Name, Date_Range, Mode_Para, List_Para, Abort_Timing):
    
    if(Record_Name == 'none'):
        return
    
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    if(Ring_Name == 'LER'):
        BEAMCURRENT = 'BMLDCCT:CURRENT'
        BUNCHNO = 'CGLINJ:BKSEL:NOB_SET'
    elif(Ring_Name == 'HER'):
        BEAMCURRENT = 'BMHDCCT:CURRENT'
        BUNCHNO = 'CGHINJ:BKSEL:NOB_SET'
    
    # 各側室のすべてのレコードのデータをロードする。
    if(Mode_Para == "STD_Tail"):
        All_Data_File_Name =  Ring_Name + '_' + Mode_Para + '_' + Method +'_Class2_All_Record_Data_' + Date_Range + '.npz'
    else:
        Mode_Para_c = 'DIF_Tail' #20230816+c
        All_Data_File_Name =  Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_' + Date_Range + '.npz' #20230816+c
    All_Data = np.load(All_Data_File_Name)['arr_0']
    
    if(Mode_Para == "STD_Tail"):
        All_Data_File_Name_B =  Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Record_Data_B_' + Date_Range + '.npz'
    else:
        Mode_Para_c = 'DIF_Tail' #20230816+c
        All_Data_File_Name_B =  Ring_Name + '_' + Mode_Para_c + '_All_Record_Data_B_' + Date_Range + '.npz' #20230816+c
    All_Data_B = np.load(All_Data_File_Name_B)['arr_0']
    
    # 各側室のすべてのレコードの解析結果をロードする。リスト型辞書。
    if(Mode_Para == 'STD_Tail'):
        All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para + '_' + Method + '_Class2_All_Result_Data_' + Date_Range + '_Dict.npz'
    else:
        Mode_Para_c = 'DIF_Tail' #20230816+c
        All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para_c + '_All_Result_Data_' + Date_Range + '_Dict.npz' #20230816+c
    All_Result = np.load(All_Result_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_Dict = All_Result.tolist()
    
    # CHKの時、STDのデータもプロットする
    # STDのデータの時間範囲(ファイルから読み込む)
    path = Ring_Name + '_Date_Range_STD_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print('No Date_Range_STD_File')
        sys.exit()   
    f = open(Ring_Name + "_Date_Range_STD_File.txt", "r")
    Date_Range_STD = f.readline()
    f.close()
    
    All_Result_STD_Data_File_Name = Ring_Name + '_STD_Tail_' + Method + '_Class2_All_Result_Data_' + Date_Range_STD + '_Dict.npz'
    All_Result_STD = np.load(All_Result_STD_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_Dict_STD = All_Result_STD.tolist()
    
    # アボート後の時間ステップ、該当するレコード名の列を抽出する。
    # ステップ: 列番号 = 0
    Time_Step_Data = All_Data[:, 0:1]
    
    # タイトルの名前のブーリアンリスト：Record_Nameの列
    Record_Name_b = [bool(All_Data[0, i] == Record_Name) for i in range(len(All_Data[0]))]
    # タイトルの名前の列をブーリアンで抜き出す：Record_Nameの列
    Record_Data = All_Data[:, Record_Name_b] 
    # もし該当するレコード名がなかったら
    if(Record_Data.size == 0):
        print('Error: No such Record Name')
        tk.messagebox.showinfo('No such Record Name', '    No such Record Name     ' + Record_Name)
        return
    
    Record_Name_B_b = [bool(All_Data_B[0, i] == Record_Name) for i in range(len(All_Data_B[0]))]
    # タイトルの名前の列をブーリアンで抜き出す：Record_Nameの列
    Record_Data_B = All_Data_B[:, Record_Name_B_b] 
    
    # 該当するレコード名の解析結果を辞書から抽出する。
    list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_Dict))
    dict_search = list_search[0]
    
    # model_C
    # 20240120変更
    w0 = dict_search['W0']
    w1 = dict_search['W1']
    w2 = dict_search['W2']
    w3 = dict_search['W3']
    w4 = dict_search['W4']
    w4 = '0'
    rmse = dict_search['RMSE']
    maxrse = dict_search['MaxRSE']
    
    # CHKの時、該当するレコード名のSTD解析結果を辞書から抽出する。
    list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_Dict_STD))
    dict_search = list_search[0]
    
    # model_C
    # 20240120変更
    w0_std = dict_search['W0']
    w1_std = dict_search['W1']
    w2_std = dict_search['W2']
    w3_std = dict_search['W3']
    w4_std = dict_search['W4']
    w4_std = '0'
    rmse_std = dict_search['RMSE']
    maxrse_std = dict_search['MaxRSE']
 
    # 202401変更
    if(float(rmse) <= 1.e-2):
        rmse = '1.e-2'
        
    # プロットの設定
    if(Mode_Para == 'STD_Strg'):
        Mode_Para_t = 'REF_Strg'
    elif(Mode_Para == 'STD_Tail'):
        Mode_Para_t = 'REF_Tail'
    elif(Mode_Para == 'DIF_Strg'):
        Mode_Para_t = 'CHK_Strg'
    elif(Mode_Para == 'DIF_Tail'):
        Mode_Para_t = 'CHK_Tail'
    else:
        Mode_Para_t = Mode_Para
        
    title1 = Record_Name + '\n' + Mode_Para_t + ' Abort: ' + Abort_Timing
    title2 = Record_Name + '\n' + Mode_Para_t + ' Abort: ' + Abort_Timing
            
    # 軸ラベルの設定
    x_label1 = 'Time step after beam abort [s]'
    y_label1 = 'Normalized pressure'
    y_label2 = '3 x Measured pressure (Reading -1e-8) [Pa]'

    # 軸の目盛設定
    x_max1 = np.max(np.array([float(s) for s in np.delete(Time_Step_Data, 0)])) + 2.
    y_max1 = (int(np.max(np.array([float(s) for s in np.delete(Record_Data, 0)])) * 10) + 2) / 10
    y_max2 = (int(np.max(np.array([float(s) for s in np.delete(Record_Data_B, 0)])) * 1.e8 * 3) * 1.4) / 1.e8

    x_limit1 = [0, x_max1]
    y_limit1 = [0, y_max1]
    y_limit2 = [0, y_max2]
    
    # テキスト
    text1 = ' '

    # plotするデータ
    Record_Data_f = np.array([float(s) for s in np.delete(Record_Data, 0)])
    Record_Data_B_f3 = np.array([float(s) for s in np.delete(Record_Data_B, 0)]) * 3.
    Time_Step_Data_f = np.array([float(s) for s in np.delete(Time_Step_Data, 0)])
    
    if(Mode_Para == 'STD_Tail'):
        y1_legend = 'Measured (Normalized)'
        y2_legend = 'Reg. REF'
        y3_legend = 'Measured (Raw)'
    
        x1 = Time_Step_Data_f
        y1 = Record_Data_f
        x2 = Time_Step_Data_f
        # model_B
        # w = [float(w0), float(w1), float(w2), float(w3)]
        # model_C
        # 20240120変更
        w = [float(w0), float(w1), float(w2), float(w3)]
        y2 = model_C(x2, w)
        x3 = Time_Step_Data_f
        y3 = Record_Data_B_f3
        
        # 引数
        x = [x1, x2, x3]
        y = [y1, y2, y3]
        title = [title1, title1, title2]
        x_label = [x_label1, x_label1, x_label1]
        y_label = [y_label1, y_label1, y_label2]
        x_limit = [x_limit1, x_limit1, x_limit1]
        y_limit = [y_limit1, y_limit1, y_limit2]
        y_legend = [y1_legend, y2_legend, y3_legend]
        cl = ["tomato", "blue", "tomato"]
        si = [20, 20, 20]
        tx = text1

    if(Mode_Para == 'DIF_Tail'):
        y1_legend = 'Reg. REF'
        y2_legend = 'Measured (Normalized)'
        y3_legend = 'Reg. CHK'
        y4_legend = 'Measured (Raw)'
    
        x1 = Time_Step_Data_f
        w_std = [float(w0_std), float(w1_std), float(w2_std), float(w3_std)]
        # model_C
        # 20240120変更
        y1 = model_C(x1, w_std)
        x2 = Time_Step_Data_f
        y2 = Record_Data_f
        x3 = Time_Step_Data_f
        w = [float(w0), float(w1), float(w2), float(w3)]
        # model_C
        # 20240120変更
        y3 = model_C(x2, w)
        x4 = Time_Step_Data_f
        y4 = Record_Data_B_f3
        
        # 引数
        x = [x1, x2, x3, x4]
        y = [y1, y2, y3, y4]
        title = [title1, title1, title1, title2]
        x_label = [x_label1, x_label1, x_label1, x_label1]
        y_label = [y_label1, y_label1, y_label1, y_label2]
        x_limit = [x_limit1, x_limit1, x_limit1, x_limit1]
        y_limit = [y_limit1, y_limit1, y_limit1, y_limit2]
        y_legend = [y1_legend, y2_legend, y3_legend, y4_legend]
        cl = ["cyan", "red", "blue", "red"]
        si = [20, 20, 20, 20]
        tx = text1

    if(Ring_Name == 'HER'):
        fc = (0.96, 1.0, 0.96)
        al = 0.8
    elif(Ring_Name == 'LER'):
        fc = (1.0, 1.0, 0.96)
        al = 0.8
        
    # データをプロット
    show_data2(x, y, title, x_label, y_label, x_limit, y_limit, y_legend, si, cl, tx, fc, al)

    # 20240223追加　ローカルリストの削除
    del All_Data, All_Data_B, All_Result, All_Result_Dict, All_Result_STD, All_Result_Dict_STD
    del Record_Name_b, Record_Data, Record_Name_B_b, Record_Data_B
    del x1, y1, x2, y2, x3, y3, x, y
    
    # メモリーの開放
    gc.collect()
    
# ==============================================================================================================
   
def Find_Abnormal_Strg(Method, Mode_Para, List_Para, File_Save_Para, Plot_Para, Abort_Timing, Class_Method, Check_Record_Name):
    # ビームあり, Method = FNN
    
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    
    # STDのモード
    Mode_Para_S = 'STD_Strg'
    
    # Kerasの結果をロードする
    if(Class_Method == 'Keras'):
        # path = 'model_strg_wb.keras'
        path = 'model_strg_wb.h5'
        is_file = os.path.isfile(path)
        if(is_file == False):
            tk.messagebox.showinfo('File Error', path + 'does not exist')
            return
        # model_strg_wb = keras.models.load_model('model_strg_wb.keras')
        model_strg_wb = keras.models.load_model('model_strg_wb.h5')
        
        # 20240227追加
        # meanとstdをロードする
        # データのファイルがあるか？
        path = 'sms_strg_wb.txt'
        is_file = os.path.isfile(path)
        if(is_file == False):
            tk.messagebox.showinfo('File Error', path + 'does not exist')
            return
        # データのファイルからmeanとstdを読み込む 
        content =[]
        with open (path, encoding = 'utf8', newline = '') as f:
            csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
            for row in csvreader:
                content = content + row #contentは1行のリストになる
        sms_strg_wb = content
    
    # Abnormal, Normalのファイルがあるか？無ければ作り、あればロードする
    Abnormal_Result_Strg_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_WB.npy'
    Abnormal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_WB.txt'
    Abnormal_Result_Strg_List = Check_Strg_Abnormal_Result_File(Abnormal_Result_Strg_File_Name, 
                                                               Abnormal_Result_Strg_Text_File_Name)
    Normal_Result_Strg_File_Name = Ring_Name + '_' + Method +'_Normal_Class2_Result_Strg_WB.npy'
    Normal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method +'_Normal_Class2_Result_Strg_WB.txt'
    Normal_Result_Strg_List = Check_Strg_Normal_Result_File(Normal_Result_Strg_File_Name, 
                                                               Normal_Result_Strg_Text_File_Name)
    
    # Date_Range_STDをロードする
    # データのファイルがあるか？
    path = Ring_Name + '_Date_Range_STD_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        tk.messagebox.showinfo('File Error', path + 'does not exist')
        return
    # データのファイルからDate_Range_STDを読み込む    
    f = open(Ring_Name + "_Date_Range_STD_File.txt", "r")
    Date_Range_STD = f.readline()
    f.close()
    
    # Date_Range_DIFをロードする
    # データのファイルがあるか？
    path = Ring_Name + '_Date_Range_DIF_Strg_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        tk.messagebox.showinfo('File Error', path + 'does not exist')
        return
    # データのファイルからDate_Range_DIFを読み込む    
    f = open(Ring_Name + "_Date_Range_DIF_Strg_File.txt", "r")
    Date_Range_DIF_Strg = f.readline()
    Date_Range_DIF = Date_Range_DIF_Strg
    f.close()
    
    # 違いを調べる　
    # 測定値と、STDの回帰曲線パラメータで計算した値でのRMSEを計算し、比べる(DIF_Strg)
    
    # DIFモードの、リングのすべてのレコードのデータをロードする(DIF_Strg)
    All_Data_File_Name = Ring_Name + '_' + Mode_Para + '_All_Record_Data_' + Date_Range_DIF + '.npz'
    All_Data = np.load(All_Data_File_Name)['arr_0']
    
    # DIFモードの、計算に用いたリングのすべてのレコードのデータをロードする。
    All_Data_File_Name_Sel = Ring_Name + '_' + Mode_Para + '_All_Record_Data_Sel_' + Date_Range_DIF + '.npz'
    All_Data_Sel = np.load(All_Data_File_Name_Sel)['arr_0']
    
    # DIFモードの、リングのすべてのレコードの解析結果をロードする。リスト型辞書(DIF_Strg)
    All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para + '_All_Result_Data_' + Date_Range_DIF + '_Dict.npz'
    All_Result = np.load(All_Result_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_DIF_Dict = All_Result.tolist()
    
    # STDモードの、リングのすべてのレコードの解析結果をロードする。リスト型辞書(STD_Strg)
    All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para_S + '_' + Method + '_Class2_All_Result_Data_' + Date_Range_STD + '_Dict.npz'
    All_Result = np.load(All_Result_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_STD_Dict = All_Result.tolist()
    
    # STDモードの、リングのすべてのレコードのデータをロードする。
    All_Data_File_Name_STD = Ring_Name + '_' + Mode_Para_S + '_' + Method + '_Class2_All_Record_Data_' + Date_Range_STD + '.npz'
    All_Data_STD = np.load(All_Data_File_Name_STD)['arr_0']
    
    # DIFのビーム電流、HOM、該当するレコード名の列を抽出する。
    # DIFのビーム電流: BEAMCURRENT: 列番号 = 1
    Beam_Data = All_Data[:, 1:2]
    # DIFの計算に使用したビーム電流: BEAMCURRENT: 列番号 = 0
    Beam_Data_Sel = All_Data_Sel[:, 0:1]
    
    # DIFのHOM: (I*I/Nb)^2: 列番号 = 2
    HOM_Data = All_Data[:, 2:3]
    # DIFの計算に使用したHOM: (I*I/Nb)^2: 列番号 = 1
    HOM_Data_Sel = All_Data_Sel[:, 1:2]
    
    # STDのビーム電流: BEAMCURRENT: 列番号 = 1
    Beam_Data_STD = All_Data_STD[:, 1:2]
    # STDのHOM: (I*I/Nb)^2: 列番号 = 2
    HOM_Data_STD = All_Data_STD[:, 2:3]
    
    # DIF、STDのレコードの名前のブーリアンリスト：最後に'PRES'がある列
    Record_Name_bp = [bool(All_Data[0, i][-4:] == 'PRES') for i in range(len(All_Data[0]))]
    # DIF、STDのレコードの名前の列をブーリアンで抜き出す
    Record_Data_p = All_Data[:, Record_Name_bp] 
    Record_List = Record_Data_p[0]
    
    ''' #20240221コメントアウト
    # STEP 追加 2023/3/8 -----------------------------------------------------------------------------------
    Beam_Data_Title = Beam_Data[0]
    HOM_Data_Title = HOM_Data[0]
    Pre_Data_Title = Record_Data_p[0]
    
    Beam_Data_f = np.array([float(s) for s in np.delete(Beam_Data, 0)])
    Beam_Max = np.max(Beam_Data_f)
    HOM_Data_f = np.array([float(s) for s in np.delete(HOM_Data, 0)])
    Pre_Data_f = np.zeros((Record_Data_p.shape[0] - 1, Record_Data_p.shape[1]))
    for ii in range(1, Record_Data_p.shape[0]):
        for jj in range(Record_Data_p.shape[1]):
            Pre_Data_f[ii - 1, jj] = Record_Data_p[ii, jj]
            
    nstep_row = int(Beam_Max / 10)

    Beam_Data_f_Step = np.zeros((nstep_row + 1, 1))
    HOM_Data_f_Step = np.zeros((nstep_row + 1, 1))
    Pre_Data_f_Step_Mean = np.zeros((nstep_row + 1, Record_Data_p.shape[1]))
    Pre_Data_f_Step_Max = np.zeros((nstep_row + 1, Record_Data_p.shape[1]))
    for ii in range(nstep_row + 1):
        Beam_Data_f_Step[ii] = 10 * ii + 5
        n = 0
        for jj in range(HOM_Data_f.shape[0]):
            if((Beam_Data_f[jj] >= 10 * ii) and (Beam_Data_f[jj] < 10 * (ii + 1))):
                n = n + 1
                HOM_Data_f_Step[ii] = HOM_Data_f_Step[ii] + HOM_Data_f[jj]
                Pre_Data_f_Step_Mean[ii, :] = Pre_Data_f_Step_Mean[ii, :] + Pre_Data_f[jj, :]
                for kk in range(Record_Data_p.shape[1]):
                    if(Pre_Data_f_Step_Max[ii, kk] <= Pre_Data_f[jj, kk]):
                        Pre_Data_f_Step_Max[ii, kk] = Pre_Data_f[jj, kk]
        if(n > 0):
            HOM_Data_f_Step[ii] = HOM_Data_f_Step[ii] / n
            Pre_Data_f_Step_Mean[ii, :] = Pre_Data_f_Step_Mean[ii, :] / n
        
    # HOM>0のブーリアンリスト
    HOM_Data_Step_b = [bool(HOM_Data_f_Step[i] > 0) for i in range(nstep_row + 1)]
    # 行をブーリアンで抜き出す
    Beam_Data_f_Step_Sel = Beam_Data_f_Step[HOM_Data_Step_b] 
    HOM_Data_f_Step_Sel = HOM_Data_f_Step[HOM_Data_Step_b]
    Pre_Data_f_Step_Mean_Sel = Pre_Data_f_Step_Mean[HOM_Data_Step_b, :]
    Pre_Data_f_Step_Max_Sel = Pre_Data_f_Step_Max[HOM_Data_Step_b, :]
    
    # 先頭行を追加
    Beam_Data_Step_Sel = np.vstack((Beam_Data_Title, Beam_Data_f_Step_Sel))
    HOM_Data_Step_Sel = np.vstack((HOM_Data_Title, HOM_Data_f_Step_Sel))
    Pre_Data_Step_Mean_Sel = np.vstack((Pre_Data_Title, Pre_Data_f_Step_Mean_Sel))
    Pre_Data_Step_Max_Sel = np.vstack((Pre_Data_Title, Pre_Data_f_Step_Max_Sel))
    
    # ------------------------------------------------------------------------------------------------
    '''
    
    # 選別にFNNモデルの結果を使う(SDM)
    
    # if(Method == 'FNN'):
    #     Strg_Class_Result_File_Name = 'ALL_SDM_Class2_Result_Strg_WB.npy'
    #     Strg_Class_Result_Text_File_Name = 'ALL_SDM_Class2_Result_Strg_WB.txt'
    #     Strg_Class_Result_List = Check_Strg_Class_Result_File(Strg_Class_Result_File_Name, Strg_Class_Result_Text_File_Name)
    #     if(Strg_Class_Result_List.shape[0] == 0):
    #         print('no class result')
    #         tk.messagebox.showinfo('Error', '  No FNN (SDM) Strg_WB result.  ')
    #         return
    #         
    #     WV = Strg_Class_Result_List[:-1]

    # 202401変更
    Abnormal_Result_Strg_List_Tmp = np.empty((0, 26))
    Normal_Result_Strg_List_Tmp = np.empty((0, 26))
    
    # すべてのレコードについて
    for Record_Name in Record_List:
        # レコード名のブーリアン
        Record_Name_b = [bool(All_Data[0, i] == Record_Name) for i in range(len(All_Data[0]))]
        # レコードの名前の列をブーリアンで抜き出す
        Record_Data = All_Data[:, Record_Name_b]
        
        # DIFの計算に使用したレコード名のブーリアン
        Record_Name_b_Sel = [bool(All_Data_Sel[0, i] == Record_Name) for i in range(len(All_Data_Sel[0]))]
        # DIFの計算に使用したレコードの名前の列をブーリアンで抜き出す
        Record_Data_Sel = All_Data_Sel[:, Record_Name_b_Sel]
        
        # STDのレコードのブーリアン
        Record_Name_STD_b = [bool(All_Data_STD[0, i] == Record_Name) for i in range(len(All_Data_STD[0]))]
        # STDのレコードの名前の列をブーリアンで抜き出す
        Record_Data_STD = All_Data_STD[:, Record_Name_STD_b]

        # DIFの計算に使用した1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
        Record_Data_Sel_f = np.array([float(s) for s in np.delete(Record_Data_Sel, 0)])
        Beam_Data_Sel_f = np.array([float(s) for s in np.delete(Beam_Data_Sel, 0)])
        HOM_Data_Sel_f = np.array([float(s) for s in np.delete(HOM_Data_Sel, 0)])
        # DIFの圧力読み値(Record_Data_Sel_f)を3倍する
        Record_Data_Sel_f3 = Record_Data_Sel_f * 3.
        
        # STDの1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
        Record_Data_STD_f = np.array([float(s) for s in np.delete(Record_Data_STD, 0)])
        Beam_Data_STD_f = np.array([float(s) for s in np.delete(Beam_Data_STD, 0)])
        # HOM_Data_STD_f = np.array([float(s) for s in np.delete(HOM_Data_STD, 0)])
        # STDの圧力読み値(Record_Data_STD_f)を3倍する
        Record_Data_STD_f3 = Record_Data_STD_f * 3.
        # STDの最大値
        Record_STD_Max = np.max(Record_Data_STD_f3)
        
        ''' # 20240221 コメントアウト
        # Stepのレコードのブーリアン   追加2023/3/8----------------------------------
        Record_Name_Step_b = [bool(Pre_Data_Step_Mean_Sel[0, i] == Record_Name) for i in range(len(Pre_Data_Step_Mean_Sel[0]))]
        # Stepのレコードの名前の列をブーリアンで抜き出す
        Record_Data_Step_Mean_Sel = Pre_Data_Step_Mean_Sel[:, Record_Name_Step_b]
        Record_Data_Step_Max_Sel = Pre_Data_Step_Max_Sel[:, Record_Name_Step_b]
        # Stepの1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
        Record_Data_f_Step_Mean_Sel = np.array([float(s) for s in np.delete(Record_Data_Step_Mean_Sel, 0)])
        Record_Data_f_Step_Max_Sel = np.array([float(s) for s in np.delete(Record_Data_Step_Max_Sel, 0)])
        # STepの圧力読み値を3倍する
        Record_Data_f3_Step_Mean_Sel = Record_Data_f_Step_Mean_Sel * 3.
        Record_Data_f3_Step_Max_Sel = Record_Data_f_Step_Max_Sel * 3.
        '''
        
        # STDの該当するレコード名の解析結果を辞書から抽出する。
        list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_STD_Dict))
        dict_search = list_search[0]
        w0 = dict_search['W0(Beam current)']
        w1 = dict_search['W1(HOM^2)']
        w2 = dict_search['W2(Base)']
        rmse = dict_search['RMSE']
        maxrse = dict_search['MaxRSE']
        W =[float(w0), float(w1), float(w2)]
        
        # DIFの該当するレコード名の解析結果を辞書から抽出する。
        list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_DIF_Dict))
        dict_search = list_search[0]
        w0_dif = dict_search['W0(Beam current)']
        w1_dif = dict_search['W1(HOM^2)']
        w2_dif = dict_search['W2(Base)']
        rmse_dif = dict_search['RMSE']
        maxrse_dif = dict_search['MaxRSE']
        W_dif =[float(w0_dif), float(w1_dif), float(w2_dif)]
        
        # 20240120 変更
        if(float(rmse) <= 1.e-8):
            rmse = '1.e-8'
        
        # 20240302 変更
        if(float(rmse_dif) <= 1.e-8):
            rmse_dif = '1.e-8'
            
        # すべてのデータ
        Record_Data_f = np.array([float(s) for s in np.delete(Record_Data, 0)])
        Record_Data_f3 = Record_Data_f * 3.
        Beam_Data_f = np.array([float(s) for s in np.delete(Beam_Data, 0)])
        HOM_Data_f = np.array([float(s) for s in np.delete(HOM_Data, 0)])

        # 最大値
        Record_Max = np.max(Record_Data_f3)
        Beam_Max = np.max(Beam_Data_f)
        # 計算に用いた範囲の最大値
        Record_Sel_Max = np.max(Record_Data_Sel_f3)
        # 計算に用いた範囲の平均値
        Record_Sel_Avg = np.mean(Record_Data_Sel_f3)
        
        # STDの回帰曲線パラメータで計算した値
        Cal_f = Beam_Data_f * float(w0) + HOM_Data_f * float(w1) + float(w2)
        Cal_f_r = np.maximum(Cal_f, 3.e-8)
        Cal_Sel_f = Beam_Data_Sel_f * float(w0) + HOM_Data_Sel_f * float(w1) + float(w2)
        Cal_Sel_f_r = np.maximum(Cal_Sel_f, 3.e-8)
        # 最大値
        Cal_Max = np.max(Cal_f_r)
        # 計算に用いた範囲の最大
        Cal_Sel_Max = np.max(Cal_Sel_f_r)
        # 計算に用いた範囲の平均値
        Cal_Sel_Avg = np.mean(Cal_Sel_f_r)
        
        # 各レコードのMSE、RMSE、最大のSE、RSEの計算(DIF): STDの回帰曲線を使う
        mse_dif, maxse_dif = mse_plane(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3, W)
        rmse_cal = np.sqrt(mse_dif)
        maxrse_cal = np.sqrt(maxse_dif)
        
        ''' #20240221 コメントアウト
        # Stepでの各レコードのMSE、RMSE、最大のSE、RSEの計算(DIF): STDの回帰曲線を使う　追加 2023/3/8 ----------------
        Beam_Data_f_Step_Sel_1 =Beam_Data_f_Step_Sel.flatten()
        HOM_Data_f_Step_Sel_1 =HOM_Data_f_Step_Sel.flatten()
        
        mse_dif_step_mean, maxse_dif_step_mean = mse_plane(Beam_Data_f_Step_Sel_1, HOM_Data_f_Step_Sel_1, 
                                                           Record_Data_f3_Step_Mean_Sel, W)
        
        rmse_cal_step_mean = np.sqrt(mse_dif_step_mean)
        maxrse_cal_step_mean = np.sqrt(maxse_dif_step_mean)
        step_mean_avg = np.mean(Record_Data_f3_Step_Mean_Sel)
        
        mse_dif_step_max, maxse_dif_step_max = mse_plane(Beam_Data_f_Step_Sel_1, HOM_Data_f_Step_Sel_1, 
                                                           Record_Data_f3_Step_Max_Sel, W)
        rmse_cal_step_max = np.sqrt(mse_dif_step_max)
        maxrse_cal_step_max = np.sqrt(maxse_dif_step_max)
        step_max_avg = np.mean(Record_Data_f3_Step_Max_Sel)
        
        # ------------------------------------------------------------------------------------------------
        '''
        
        # 異常かどうかの判断
        Abnormal_Flag_FNN = 0 # 0 -> Normal, 1 -> Abnormal
        # Abnormal_Flag_Logi = 0 # 0 -> Normal, 1 -> Abnormal
        # Abnormal_Flag_All = 0 # 0 -> Normal, 1 -> Abnormal
        Abnormal_Flag_2 = 1 # 0 -> Normal, 1 -> Abnormal
        
        # FNN回帰モデルの結果を使う(SDM)
        if(Class_Method == 'SDM'):
            Strg_Class_Result_File_Name = 'ALL_SDM_Class2_Result_Strg_WB.npy'
            Strg_Class_Result_Text_File_Name = 'ALL_SDM_Class2_Result_Strg_WB.txt'
            Strg_Class_Result_List = Check_Strg_Class_Result_File(Strg_Class_Result_File_Name, Strg_Class_Result_Text_File_Name)
            if(Strg_Class_Result_List.shape[0] == 0):
                print('no class result')
                tk.messagebox.showinfo('Error', '  No FNN (SDM) Strg_WB result.  ')
                return
            
            WV = Strg_Class_Result_List[:-1]
            
            # divide zero対策
            if(float(rmse) < 1.e-8): # 2023/3/16 変更
                rmse = '1.e-8'
            x1 = np.log10(rmse_cal / float(rmse))
            x2 = np.log10(float(rmse_dif) / Record_Sel_Avg)
            x3 = np.log10(rmse_cal / Record_Sel_Avg) #2023/3/1変更
            
            M = 2
            K = 2
            CX = np.array([[x1, x2, x3]])
            CY, a, z, b =FNN(WV, M, K, CX)
            Cmax = np.argmax(CY)
            
            if(Cmax == 0):
                Abnormal_Flag_FNN = 1

            # 20240202変更
            if(Record_Max < 1.e-7) or (Record_Sel_Avg <= Cal_Sel_Avg * 1.1):
                Abnormal_Flag_2 = 0
            
            # ローカルリストの削除　20240221追加
            del x1, x2, x3, CX, CY, a, z, b

        # 20240204変更
        # FNN回帰モデルの結果を使う(Keras)
        if(Class_Method == 'Keras'):
            # 入力パラメータは5個)
            X1 = funclog1(rmse_cal)
            X2 = funclog1(float(rmse))
            X3 = funclog1(float(rmse_dif))
            X4 = funclog1(Record_Sel_Avg)
            X5 = funclog1(Record_Sel_Max)
            X6 = funclog1(Cal_Sel_Avg)
            
            X = np.hstack((X1, X2))
            X = np.hstack((X, X3))
            X = np.hstack((X, X4))
            X = np.hstack((X, X5))
            X = np.hstack((X, X6))
            
            X = X[None,:]
            
            # print(5881, 'before X', X)
            
            # 20240227追加
            # 標準化(Training dataのmeanとstdを使う)
            for i in range(6):
                me1 = float(sms_strg_wb[i * 2])
                si1 = float(sms_strg_wb[1 + i * 2])
                X[:, i] = standard_n1(X[:, i], me1, si1)
            
            # print(5890, 'after X', X)
            # print(me1, si1)
            
            Predictions = model_strg_wb.predict_on_batch(X)
            
            Cmax = Predictions[0].argmax()
            
            if(Cmax == 0):
                Abnormal_Flag_FNN = 1

            # 20240202変更
            if(Record_Max < 1.e-7) or (Record_Sel_Avg <= Cal_Sel_Avg * 1.1):
                Abnormal_Flag_2 = 0
            
            # 20240226追加
            keras.backend.clear_session()
            # ローカルリストの削除　20240221追加
            del X1, X2, X3, X4, X5, X6, X, Predictions
            gc.collect()
            
        if(Abnormal_Flag_2 == 0) and (Abnormal_Flag_FNN == 1):
            # print(Record_Name + ' Strg is normal! ', 'RMSE_CAL = {:.2e}, RMSE_STD = {:.2e}, Pre_Max = {:.2e}'.format(rmse_cal, 
            #                                                                                     float(rmse), float(Record_Max)))
            # 202401変更
            Normal_Result_Strg_List_Tmp = np.vstack((Normal_Result_Strg_List_Tmp, np.array([Date_Range_DIF, Date_Range_STD, 
                                Abort_Timing, Record_Name, '{:.2e}'.format(rmse_cal), '{:.2e}'.format(float(rmse)), 
                                '{:.2e}'.format(Beam_Max), '{:.3e}'.format(float(maxrse)), 
                                '{:.2e}'.format(Record_Max), 
                                '{:.2e}'.format(Record_Sel_Max), '{:.2e}'.format(Record_Sel_Avg), 
                                '{:.2e}'.format(Cal_Sel_Max), '{:.2e}'.format(Cal_Sel_Avg),
                                '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                '{:.2e}'.format(float(w0)), '{:.2e}'.format(float(w1)),
                                '{:.2e}'.format(float(w2)), '{:.2e}'.format(float(w0_dif)),
                                '{:.2e}'.format(float(w1_dif)), '{:.2e}'.format(float(w2_dif)),
                                '{:.2e}'.format(float(rmse_dif)), '{:.2e}'.format(float(maxrse_dif)),
                                '{:.2e}'.format(0.)])))
        if(Abnormal_Flag_2 == 1) and (Abnormal_Flag_FNN == 1):
            # print(Record_Name + ' Strg is abnormal! ', 'RMSE_CAL = {:.2e}, RMSE_STD = {:.2e}, Pre_Max = {:.2e}'.format(rmse_cal, 
            #                                                                                     float(rmse), float(Record_Max)))
            
            # 202401変更
            Abnormal_Result_Strg_List_Tmp = np.vstack((Abnormal_Result_Strg_List_Tmp, np.array([Date_Range_DIF, Date_Range_STD, 
                                Abort_Timing, Record_Name, '{:.2e}'.format(rmse_cal), '{:.2e}'.format(float(rmse)), 
                                '{:.2e}'.format(Beam_Max), '{:.3e}'.format(float(maxrse)), 
                                '{:.2e}'.format(Record_Max), 
                                '{:.2e}'.format(Record_Sel_Max), '{:.2e}'.format(Record_Sel_Avg), 
                                '{:.2e}'.format(Cal_Sel_Max), '{:.2e}'.format(Cal_Sel_Avg),
                                '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                '{:.2e}'.format(float(w0)), '{:.2e}'.format(float(w1)),
                                '{:.2e}'.format(float(w2)), '{:.2e}'.format(float(w0_dif)),
                                '{:.2e}'.format(float(w1_dif)), '{:.2e}'.format(float(w2_dif)),
                                '{:.2e}'.format(float(rmse_dif)), '{:.2e}'.format(float(maxrse_dif)),
                                '{:.2e}'.format(0.)])))
        
        # 20240207追加
        if(Record_Name == Check_Record_Name):
            rn0 = ['Date_Range_DIF', 'Date_Range_STD', 'Abort_Timing', 'Record Name', 
                                'RMSE_cal', 'RMSE_std', 'Beam_Max', 'MaxRSE_std', 'Max_Pre', 'Max_Sel_Pre', 
                                'Avg_Sel_Pre', 'Max_Cal', 'Avg_Cal', 'RMSE_STEP_Max', 'STEP_Max_Avg', 
                                'RMSE_STEP_Mean', 'STEP_Mean_Avg', 'W0_std', 'W1_std', 'W2_std', 'W0_dif', 
                                'W1_dif', 'W2_dif', 'RMSE_dif', 'MaxRMSE_dif', 'Cause_no']
            rn1 = [Date_Range_DIF, Date_Range_STD, 
                                Abort_Timing, Record_Name, '{:.2e}'.format(rmse_cal), '{:.2e}'.format(float(rmse)), 
                                '{:.2e}'.format(Beam_Max), '{:.3e}'.format(float(maxrse)), 
                                '{:.2e}'.format(Record_Max), 
                                '{:.2e}'.format(Record_Sel_Max), '{:.2e}'.format(Record_Sel_Avg), 
                                '{:.2e}'.format(Cal_Sel_Max), '{:.2e}'.format(Cal_Sel_Avg),
                                '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                '{:.2e}'.format(float(w0)), '{:.2e}'.format(float(w1)),
                                '{:.2e}'.format(float(w2)), '{:.2e}'.format(float(w0_dif)),
                                '{:.2e}'.format(float(w1_dif)), '{:.2e}'.format(float(w2_dif)),
                                '{:.2e}'.format(float(rmse_dif)), '{:.2e}'.format(float(maxrse_dif)),
                                '{:.2e}'.format(0.)]
            rn2 = np.vstack((rn0, rn1))
            print(rn2.T)
        
        # ローカルリストの削除　20240221 追加
        del Record_Name_b, Record_Data, Record_Name_b_Sel, Record_Data_Sel
        del Record_Name_STD_b, Record_Data_STD, Record_Data_Sel_f
        del Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3
        del Record_Data_STD_f, Beam_Data_STD_f, Record_Data_STD_f3, Record_Data_f
        del Record_Data_f3, Beam_Data_f, HOM_Data_f
        del Cal_f, Cal_Sel_f
    
    # 一時異常リストのプリント
    # print('Abnormal_Result_Strg_List_Tmp =', Abnormal_Result_Strg_List_Tmp)
    
    # もし同じDate_Range_DIFなら確認しない
    if(Abnormal_Result_Strg_List_Tmp.shape[0] > 0): # Abnormalなレコードが一つ以上あれば
        if(Abnormal_Result_Strg_List_Tmp[0][0] in Abnormal_Result_Strg_List[1:, 0]): # Date_Range_DIFが同じかどうか
            print('The same abort time Abnormal Strg data exist.')
            # print('The present Abnormal_Result_Strg_List = ', Abnormal_Result_Strg_List)
        
    # 行数が1行以上あり
    if(Abnormal_Result_Strg_List_Tmp.shape[0] > 0):
        # 同じ時間ではないなら
        if(Abnormal_Result_Strg_List_Tmp[0][0] not in Abnormal_Result_Strg_List[1:, 0]):
            Abnormal_Result_Strg_List = np.vstack((Abnormal_Result_Strg_List, Abnormal_Result_Strg_List_Tmp))
        else:
            print('The same abort time Abnormal Strg data exists.')
            # tk.messagebox.showinfo('Attention', 'The same abort time Abnormal Strg WB data exists.')
    else:
        print('No new abnormal Strg result')
        # tk.messagebox.showinfo('Note', 'No new abnormal Strg result')
    
    if(Normal_Result_Strg_List_Tmp.shape[0] > 0):
        # 同じ時間ではないなら
        if(Normal_Result_Strg_List_Tmp[0][0] not in Normal_Result_Strg_List[1:, 0]):
            Normal_Result_Strg_List = np.vstack((Normal_Result_Strg_List, Normal_Result_Strg_List_Tmp))
        else:
            print('The same abort time Normal Strg data exist.')
            # tk.messagebox.showinfo('Attention', 'The same abort time Normal Strg WB data exists.')
    else:
        print('No new Normal Strg result')
        # tk.messagebox.showinfo('Note', 'No new normal Strg result')
        
    # 結果のプリントと保存
    # print('Abnormal_Result_Strg_List =', Abnormal_Result_Strg_List)
    # print('Normal_Result_Strg_List =', Normal_Result_Strg_List)
    
    # Abnormalにもnormalにも新しいデータが無いならダミーをnormalに追加する。
    if(Abnormal_Result_Strg_List_Tmp.shape[0] == 0) and (Normal_Result_Strg_List_Tmp.shape[0] == 0):
        if(Method != 'Manual'):
            Normal_Result_Strg_List_Tmp = np.array([Date_Range_DIF, Date_Range_STD, Abort_Timing, 'Dummy', '1.00e-08', '1.00e-08', 
                                                    '{:.3e}'.format(Beam_Max), '1.00e-08', '1.00e-08', '1.00e-08', '1.00e-08',
                                                    '1.00e-08', '1.00e-08', '1.00e-08', '1.00e-08', '1.00e-08', '1.00e-08', 
                                                    '1.00e-08', '1.00e-08', '1.00e-08', '1.00e-08', '1.00e-08', '1.00e-08', 
                                                    '1.00e-08', '1.00e-08', '0.'])
            Normal_Result_Strg_List = np.vstack((Normal_Result_Strg_List, Normal_Result_Strg_List_Tmp))
    
    # Abnromal Recordの数
    N_rec = Abnormal_Result_Strg_List_Tmp.shape[0]
    
    if (File_Save_Para == 1):
        # 異常リストを保存する
        # 保存するリスト
        Array_to_Save = Abnormal_Result_Strg_List
        # arrayで保存するファイル名
        Array_File_Name_to_Save = Abnormal_Result_Strg_File_Name
        # textで保存するファイル名
        Text_File_Name_to_Save = Abnormal_Result_Strg_Text_File_Name
        # 保存する関数
        Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
        
        # 正常リストを保存する
        # 保存するリスト
        Array_to_Save = Normal_Result_Strg_List
        # arrayで保存するファイル名
        Array_File_Name_to_Save = Normal_Result_Strg_File_Name
        # textで保存するファイル名
        Text_File_Name_to_Save = Normal_Result_Strg_Text_File_Name
        # 保存する関数
        Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
        
        del Array_to_Save

    # ローカルリストの削除 20240221追加
    del Abnormal_Result_Strg_List, Normal_Result_Strg_List
    del All_Data, All_Data_Sel, All_Result, All_Result_DIF_Dict, All_Result_STD_Dict, All_Data_STD
    del Beam_Data, Beam_Data_Sel, HOM_Data, HOM_Data_Sel, Beam_Data_STD, HOM_Data_STD
    del Record_Data_p, Record_List
    del Abnormal_Result_Strg_List_Tmp, Normal_Result_Strg_List_Tmp
    if(Class_Method == 'Keras'):
        del model_strg_wb
    if(Class_Method == 'SDM'):
        del Strg_Class_Result_List, WV
        
    #メモリ開放
    gc.collect()
    
    return N_rec

# ==============================================================================================================
   
def Save_Manual_Strg(self, Method, Mode_Para, List_Para, Check_Record_Name, Save_Class, Abort_Timing):
    
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    
    # STDのモード
    Mode_Para_S = 'STD_Strg'
    
    # クラス化するためにデータを選別した方法
    Method_d = 'Manual'
    # Manualで選別したAbnormal, Normalのファイルがあるか？無ければ作り、あればロードする
    Abnormal_Result_Strg_File_Name = Ring_Name + '_' + Method_d +'_Abnormal_Class2_Result_Strg_WB.npy'
    Abnormal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method_d +'_Abnormal_Class2_Result_Strg_WB.txt'
    Abnormal_Result_Strg_List = Check_Strg_Abnormal_Result_File(Abnormal_Result_Strg_File_Name, 
                                                               Abnormal_Result_Strg_Text_File_Name)
    Normal_Result_Strg_File_Name = Ring_Name + '_' + Method_d +'_Normal_Class2_Result_Strg_WB.npy'
    Normal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method_d +'_Normal_Class2_Result_Strg_WB.txt'
    Normal_Result_Strg_List = Check_Strg_Normal_Result_File(Normal_Result_Strg_File_Name, 
                                                               Normal_Result_Strg_Text_File_Name)
    
    # ファイルからDate_Range_STDを読み込む    
    f = open(Ring_Name + "_Date_Range_STD_File.txt", "r")
    Date_Range_STD = f.readline()
    f.close()
    
    # ファイルからDate_Range_DIFを読み込む    
    f = open(Ring_Name + "_Date_Range_DIF_Strg_File.txt", "r")
    Date_Range_DIF_Strg = f.readline()
    Date_Range_DIF = Date_Range_DIF_Strg
    f.close()
    
    # 測定値と、STDの回帰曲線パラメータで計算した値でのRMSEを計算し、比べる(DIF_Strg)
    
    # DIFモードの、リングのすべてのレコードのデータをロードする(DIF_Strg)
    All_Data_File_Name = Ring_Name + '_' + Mode_Para + '_All_Record_Data_' + Date_Range_DIF + '.npz'
    All_Data = np.load(All_Data_File_Name)['arr_0']
    
    # DIFモードの、計算に用いたリングのすべてのレコードのデータをロードする。
    All_Data_File_Name_Sel = Ring_Name + '_' + Mode_Para + '_All_Record_Data_Sel_' + Date_Range_DIF + '.npz'
    All_Data_Sel = np.load(All_Data_File_Name_Sel)['arr_0']
    
    # DIFモードの、リングのすべてのレコードの解析結果をロードする。リスト型辞書(DIF_Strg)
    All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para + '_All_Result_Data_' + Date_Range_DIF + '_Dict.npz'
    All_Result = np.load(All_Result_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_DIF_Dict = All_Result.tolist()
    
    # STDモードの、リングのすべてのレコードの解析結果をロードする。リスト型辞書(STD_Strg)
    All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para_S + '_' + Method + '_Class2_All_Result_Data_' + Date_Range_STD + '_Dict.npz'
    All_Result = np.load(All_Result_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_STD_Dict = All_Result.tolist()
    
    # STDモードの、リングのすべてのレコードのデータをロードする。
    All_Data_File_Name_STD = Ring_Name + '_' + Mode_Para_S + '_' + Method + '_Class2_All_Record_Data_' + Date_Range_STD + '.npz'
    All_Data_STD = np.load(All_Data_File_Name_STD)['arr_0']
    
    # DIFのビーム電流、HOM、該当するレコード名の列を抽出する。
    # DIFのビーム電流: BEAMCURRENT: 列番号 = 1
    Beam_Data = All_Data[:, 1:2]
    # DIFの計算に使用したビーム電流: BEAMCURRENT: 列番号 = 0
    Beam_Data_Sel = All_Data_Sel[:, 0:1]
    
    # DIFのHOM: (I*I/Nb)^2: 列番号 = 2
    HOM_Data = All_Data[:, 2:3]
    # DIFの計算に使用したHOM: (I*I/Nb)^2: 列番号 = 1
    HOM_Data_Sel = All_Data_Sel[:, 1:2]
    
    # STDのビーム電流: BEAMCURRENT: 列番号 = 1
    Beam_Data_STD = All_Data_STD[:, 1:2]
    # STDのHOM: (I*I/Nb)^2: 列番号 = 2
    # HOM_Data_STD = All_Data_STD[:, 2:3]
    
    # DIF、STDのレコードの名前のブーリアンリスト：最後に'PRES'がある列
    Record_Name_bp = [bool(All_Data[0, i][-4:] == 'PRES') for i in range(len(All_Data[0]))]
    # DIF、STDのレコードの名前の列をブーリアンで抜き出す
    Record_Data_p = All_Data[:, Record_Name_bp] 
    Record_List = Record_Data_p[0]
    
    ''' 20240223コメントアウト
    # STEP 追加 2023/3/8 -----------------------------------------------------------------------------------
    Beam_Data_Title = Beam_Data[0]
    HOM_Data_Title = HOM_Data[0]
    Pre_Data_Title = Record_Data_p[0]
    
    Beam_Data_f = np.array([float(s) for s in np.delete(Beam_Data, 0)])
    Beam_Max = np.max(Beam_Data_f)
    HOM_Data_f = np.array([float(s) for s in np.delete(HOM_Data, 0)])
    Pre_Data_f = np.zeros((Record_Data_p.shape[0] - 1, Record_Data_p.shape[1]))
    for ii in range(1, Record_Data_p.shape[0]):
        for jj in range(Record_Data_p.shape[1]):
            Pre_Data_f[ii - 1, jj] = Record_Data_p[ii, jj]
            
    nstep_row = int(Beam_Max / 10)

    Beam_Data_f_Step = np.zeros((nstep_row + 1, 1))
    HOM_Data_f_Step = np.zeros((nstep_row + 1, 1))
    Pre_Data_f_Step_Mean = np.zeros((nstep_row + 1, Record_Data_p.shape[1]))
    Pre_Data_f_Step_Max = np.zeros((nstep_row + 1, Record_Data_p.shape[1]))
    for ii in range(nstep_row + 1):
        Beam_Data_f_Step[ii] = 10 * ii + 5
        n = 0
        for jj in range(HOM_Data_f.shape[0]):
            if((Beam_Data_f[jj] >= 10 * ii) and (Beam_Data_f[jj] < 10 * (ii + 1))):
                n = n + 1
                HOM_Data_f_Step[ii] = HOM_Data_f_Step[ii] + HOM_Data_f[jj]
                Pre_Data_f_Step_Mean[ii, :] = Pre_Data_f_Step_Mean[ii, :] + Pre_Data_f[jj, :]
                for kk in range(Record_Data_p.shape[1]):
                    if(Pre_Data_f_Step_Max[ii, kk] <= Pre_Data_f[jj, kk]):
                        Pre_Data_f_Step_Max[ii, kk] = Pre_Data_f[jj, kk]

        if(n > 0):
            HOM_Data_f_Step[ii] = HOM_Data_f_Step[ii] / n
            Pre_Data_f_Step_Mean[ii, :] = Pre_Data_f_Step_Mean[ii, :] / n
        
    # HOM>0のブーリアンリスト
    HOM_Data_Step_b = [bool(HOM_Data_f_Step[i] > 0) for i in range(nstep_row + 1)]
    # 行をブーリアンで抜き出す
    Beam_Data_f_Step_Sel = Beam_Data_f_Step[HOM_Data_Step_b] 
    HOM_Data_f_Step_Sel = HOM_Data_f_Step[HOM_Data_Step_b]
    Pre_Data_f_Step_Mean_Sel = Pre_Data_f_Step_Mean[HOM_Data_Step_b, :]
    Pre_Data_f_Step_Max_Sel = Pre_Data_f_Step_Max[HOM_Data_Step_b, :]
    
    # 先頭行を追加
    Beam_Data_Step_Sel = np.vstack((Beam_Data_Title, Beam_Data_f_Step_Sel))
    HOM_Data_Step_Sel = np.vstack((HOM_Data_Title, HOM_Data_f_Step_Sel))
    Pre_Data_Step_Mean_Sel = np.vstack((Pre_Data_Title, Pre_Data_f_Step_Mean_Sel))
    Pre_Data_Step_Max_Sel = np.vstack((Pre_Data_Title, Pre_Data_f_Step_Max_Sel))
    
    # ------------------------------------------------------------------------------------------------
    '''
    
    # 一時ファイルの定義
    Abnormal_Result_Strg_List_Tmp = np.empty((0, 26))
    Normal_Result_Strg_List_Tmp = np.empty((0, 26))
    
    # Record_Listの再定義(必要なrecord名のみ)
    Record_List = [Check_Record_Name]
    
    for Record_Name in Record_List:
        # レコード名のブーリアン
        Record_Name_b = [bool(All_Data[0, i] == Record_Name) for i in range(len(All_Data[0]))]
        # レコードの名前の列をブーリアンで抜き出す
        Record_Data = All_Data[:, Record_Name_b]
        
        # DIFの計算に使用したレコード名のブーリアン
        Record_Name_b_Sel = [bool(All_Data_Sel[0, i] == Record_Name) for i in range(len(All_Data_Sel[0]))]
        # DIFの計算に使用したレコードの名前の列をブーリアンで抜き出す
        Record_Data_Sel = All_Data_Sel[:, Record_Name_b_Sel]
        
        # STDのレコードのブーリアン
        Record_Name_STD_b = [bool(All_Data_STD[0, i] == Record_Name) for i in range(len(All_Data_STD[0]))]
        # STDのレコードの名前の列をブーリアンで抜き出す
        Record_Data_STD = All_Data_STD[:, Record_Name_STD_b]

        # DIFの計算に使用した1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
        Record_Data_Sel_f = np.array([float(s) for s in np.delete(Record_Data_Sel, 0)])
        Beam_Data_Sel_f = np.array([float(s) for s in np.delete(Beam_Data_Sel, 0)])
        HOM_Data_Sel_f = np.array([float(s) for s in np.delete(HOM_Data_Sel, 0)])
        # DIFの圧力読み値(Record_Data_Sel_f)を3倍する
        Record_Data_Sel_f3 = Record_Data_Sel_f * 3.
        
        # STDの1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
        Record_Data_STD_f = np.array([float(s) for s in np.delete(Record_Data_STD, 0)])
        Beam_Data_STD_f = np.array([float(s) for s in np.delete(Beam_Data_STD, 0)])
        # HOM_Data_STD_f = np.array([float(s) for s in np.delete(HOM_Data_STD, 0)])
        # STDの圧力読み値(Record_Data_STD_f)を3倍する
        Record_Data_STD_f3 = Record_Data_STD_f * 3.
        # STDの最大値
        Record_STD_Max = np.max(Record_Data_STD_f3)
        
        ''' 20240223コメントアウト
        # Stepのレコードのブーリアン   追加2023/3/8----------------------------------
        Record_Name_Step_b = [bool(Pre_Data_Step_Mean_Sel[0, i] == Record_Name) for i in range(len(Pre_Data_Step_Mean_Sel[0]))]
        # Stepのレコードの名前の列をブーリアンで抜き出す
        Record_Data_Step_Mean_Sel = Pre_Data_Step_Mean_Sel[:, Record_Name_Step_b]
        Record_Data_Step_Max_Sel = Pre_Data_Step_Max_Sel[:, Record_Name_Step_b]
        # Stepの1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
        Record_Data_f_Step_Mean_Sel = np.array([float(s) for s in np.delete(Record_Data_Step_Mean_Sel, 0)])
        Record_Data_f_Step_Max_Sel = np.array([float(s) for s in np.delete(Record_Data_Step_Max_Sel, 0)])
        # STepの圧力読み値を3倍する
        Record_Data_f3_Step_Mean_Sel = Record_Data_f_Step_Mean_Sel * 3.
        Record_Data_f3_Step_Max_Sel = Record_Data_f_Step_Max_Sel * 3.

        # -------------------------------------------------------------------------
        '''
        
        # STDの該当するレコード名の解析結果を辞書から抽出する。
        list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_STD_Dict))
        dict_search = list_search[0]
        w0 = dict_search['W0(Beam current)']
        w1 = dict_search['W1(HOM^2)']
        w2 = dict_search['W2(Base)']
        rmse = dict_search['RMSE']
        maxrse = dict_search['MaxRSE']
        W =[float(w0), float(w1), float(w2)]
        
        # DIFの該当するレコード名の解析結果を辞書から抽出する。
        list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_DIF_Dict))
        dict_search = list_search[0]
        w0_dif = dict_search['W0(Beam current)']
        w1_dif = dict_search['W1(HOM^2)']
        w2_dif = dict_search['W2(Base)']
        rmse_dif = dict_search['RMSE']
        maxrse_dif = dict_search['MaxRSE']
        W_dif =[float(w0_dif), float(w1_dif), float(w2_dif)]
        
        # divide zero対策
        # 20240120 変更
        if(float(rmse) <= 1.e-8):
            rmse = '1.e-8'
        
        # すべてのデータ
        Record_Data_f = np.array([float(s) for s in np.delete(Record_Data, 0)])
        Record_Data_f3 = Record_Data_f * 3.
        Beam_Data_f = np.array([float(s) for s in np.delete(Beam_Data, 0)])
        HOM_Data_f = np.array([float(s) for s in np.delete(HOM_Data, 0)])

        # 最大値
        Record_Max = np.max(Record_Data_f3)
        Beam_Max = np.max(Beam_Data_f)
        # 計算に用いた範囲の最大値
        Record_Sel_Max = np.max(Record_Data_Sel_f3)
        # 計算に用いた範囲の平均値
        Record_Sel_Avg = np.mean(Record_Data_Sel_f3)
        
        # STDの回帰曲線パラメータで計算した値 (w0, w1, w2は既に3倍されているはず)
        Cal_f = Beam_Data_f * float(w0) + HOM_Data_f * float(w1) + float(w2)
        Cal_f_r = np.maximum(Cal_f, 3.e-8)
        Cal_Sel_f = Beam_Data_Sel_f * float(w0) + HOM_Data_Sel_f * float(w1) + float(w2)
        Cal_Sel_f_r = np.maximum(Cal_Sel_f, 3.e-8)
        # 最大値
        Cal_Max = np.max(Cal_f_r)
        # 計算に用いた範囲の最大
        Cal_Sel_Max = np.max(Cal_Sel_f_r)
        # 計算に用いた範囲の平均値
        Cal_Sel_Avg = np.mean(Cal_Sel_f_r)
        
        # 各レコードのMSE、RMSE、最大のSE、RSEの計算(DIF): STDの回帰曲線を使う
        mse_cal, maxse_cal = mse_plane(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3, W)
        rmse_cal = np.sqrt(mse_cal)
        maxrse_cal = np.sqrt(maxse_cal)
        
        ''' 20240223コメントアウト
        # Stepでの各レコードのMSE、RMSE、最大のSE、RSEの計算(DIF): STDの回帰曲線を使う　追加 2023/3/8 ----------------
        Beam_Data_f_Step_Sel_1 =Beam_Data_f_Step_Sel.flatten()
        HOM_Data_f_Step_Sel_1 =HOM_Data_f_Step_Sel.flatten()
        
        mse_dif_step_mean, maxse_dif_step_mean = mse_plane(Beam_Data_f_Step_Sel_1, HOM_Data_f_Step_Sel_1, 
                                                           Record_Data_f3_Step_Mean_Sel, W)
        
        rmse_cal_step_mean = np.sqrt(mse_dif_step_mean)
        maxrse_cal_step_mean = np.sqrt(maxse_dif_step_mean)
        step_mean_avg = np.mean(Record_Data_f3_Step_Mean_Sel)
        
        mse_dif_step_max, maxse_dif_step_max = mse_plane(Beam_Data_f_Step_Sel_1, HOM_Data_f_Step_Sel_1, 
                                                           Record_Data_f3_Step_Max_Sel, W)
        rmse_cal_step_max = np.sqrt(mse_dif_step_max)
        maxrse_cal_step_max = np.sqrt(maxse_dif_step_max)
        step_max_avg = np.mean(Record_Data_f3_Step_Max_Sel)
        
        # ------------------------------------------------------------------------------------------------
        '''
        
        if(Save_Class == 'Nor'):
            Normal_Result_Strg_List_Tmp = np.vstack((Normal_Result_Strg_List_Tmp, np.array([Date_Range_DIF, Date_Range_STD, 
                                    Abort_Timing, Record_Name, '{:.2e}'.format(rmse_cal), '{:.2e}'.format(float(rmse)), 
                                    '{:.2e}'.format(Beam_Max), '{:.3e}'.format(float(maxrse)), 
                                    '{:.2e}'.format(Record_Max), 
                                    '{:.2e}'.format(Record_Sel_Max), '{:.2e}'.format(Record_Sel_Avg), 
                                    '{:.2e}'.format(Cal_Sel_Max), '{:.2e}'.format(Cal_Sel_Avg),
                                    '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                    '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                    '{:.2e}'.format(float(w0)), '{:.2e}'.format(float(w1)),
                                    '{:.2e}'.format(float(w2)), '{:.2e}'.format(float(w0_dif)),
                                    '{:.2e}'.format(float(w1_dif)), '{:.2e}'.format(float(w2_dif)),
                                    '{:.2e}'.format(float(rmse_dif)), '{:.2e}'.format(float(maxrse_dif)),
                                    '{:.2e}'.format(0.)])))
        
        if(Save_Class == 'Abn'):
            # subwindowで推定原因を尋ねる 20240218変更　-----------------------------------------------------------------------
            # subwindowのタイトル
            wtitle = Record_Name + ' (Abnormal Strg_WB)'
            
            # Strg_WBのpossible cause選択肢
            self.Cause_List = ['Leak or Pump failure?', 'Over heating or Discharge?', 'Abnormal orbit or Leak?']
            
            # subwindowをつくる
            sub_win = self.getSubWindow(self.Cause_List, wtitle)
            
            # subwindowが閉じられるのを待つ。
            root.wait_window(sub_win) # サブウィンドウが閉じられるのを待つ
            
            # Possible Causeのインデックス
            mpc = self.Cau_var.get()

            Abnormal_Result_Strg_List_Tmp = np.vstack((Abnormal_Result_Strg_List_Tmp, np.array([Date_Range_DIF, Date_Range_STD, 
                                    Abort_Timing, Record_Name, '{:.2e}'.format(rmse_cal), '{:.2e}'.format(float(rmse)), 
                                    '{:.2e}'.format(Beam_Max), '{:.3e}'.format(float(maxrse)), 
                                    '{:.2e}'.format(Record_Max), 
                                    '{:.2e}'.format(Record_Sel_Max), '{:.2e}'.format(Record_Sel_Avg), 
                                    '{:.2e}'.format(Cal_Sel_Max), '{:.2e}'.format(Cal_Sel_Avg),
                                    '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                    '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                    '{:.2e}'.format(float(w0)), '{:.2e}'.format(float(w1)),
                                    '{:.2e}'.format(float(w2)), '{:.2e}'.format(float(w0_dif)),
                                    '{:.2e}'.format(float(w1_dif)), '{:.2e}'.format(float(w2_dif)),
                                    '{:.2e}'.format(float(rmse_dif)), '{:.2e}'.format(float(maxrse_dif)),
                                    '{:.2e}'.format(float(mpc))])))
                
    # 行数が1行以上あれば
    if(Abnormal_Result_Strg_List_Tmp.shape[0] > 0):
        # 同じ時間ではないなら
        # if(Abnormal_Result_Strg_List_Tmp[0][0] not in Abnormal_Result_Strg_List[1:, 0]):
        #     Abnormal_Result_Strg_List = np.vstack((Abnormal_Result_Strg_List, Abnormal_Result_Strg_List_Tmp))
        # else:
        #     print('The same abort time Abnormal Strg data exist.')
        #     tk.messagebox.showinfo('Attention', 'The same abort time Abnormal Strg WB data exists.')
        
        # 20240402
        # 同じ時間、同じレコード名でないなら
        nrowtmp = Abnormal_Result_Strg_List.shape[0]
        hit = 0
        for i in range(nrowtmp):
            if(Abnormal_Result_Strg_List_Tmp[0][0] == Abnormal_Result_Strg_List[i, 0]):
                if(Abnormal_Result_Strg_List_Tmp[0][3] == Abnormal_Result_Strg_List[i, 3]):
                    hit = 1
        
        if(hit == 0):
            Abnormal_Result_Strg_List = np.vstack((Abnormal_Result_Strg_List, Abnormal_Result_Strg_List_Tmp))
        else:
            print('The same abort time and record name exist in Abnormal Strg data.')
            tk.messagebox.showinfo('Attention', 'The same abort time and record name exist in Abnormal Strg data.')
            
        # 異常リストを保存する
        # 保存するリスト
        Array_to_Save = Abnormal_Result_Strg_List
        # arrayで保存するファイル名
        Array_File_Name_to_Save = Abnormal_Result_Strg_File_Name
        # textで保存するファイル名
        Text_File_Name_to_Save = Abnormal_Result_Strg_Text_File_Name
        # 保存する関数
        Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
        
        del Array_to_Save
    
    # 行数が1行以上あれば
    if(Normal_Result_Strg_List_Tmp.shape[0] > 0):
        # 同じ時間ではないなら
        # if(Normal_Result_Strg_List_Tmp[0][0] not in Normal_Result_Strg_List[1:, 0]):
        #     Normal_Result_Strg_List = np.vstack((Normal_Result_Strg_List, Normal_Result_Strg_List_Tmp))
        # else:
        #     print('The same abort time and record name exist in Normal Strg data.')
        #     tk.messagebox.showinfo('Attention', 'The same abort time and record name exist in Normal Strg data.')
        
        # 20240402
        # 同じ時間、同じレコード名でないなら
        nrowtmp = Normal_Result_Strg_List.shape[0]
        hit = 0
        for i in range(nrowtmp):
            if(Normal_Result_Strg_List_Tmp[0][0] == Normal_Result_Strg_List[i, 0]):
                if(Normal_Result_Strg_List_Tmp[0][3] == Normal_Result_Strg_List[i, 3]):
                    hit = 1
        
        if(hit == 0):
            Normal_Result_Strg_List = np.vstack((Normal_Result_Strg_List, Normal_Result_Strg_List_Tmp))
        else:
            print('The same abort time and record name exist in Normal Strg data.')
            tk.messagebox.showinfo('Attention', 'The same abort time and record name exist in Normal Strg data.')
            
        # 正常リストを保存する
        # 保存するリスト
        Array_to_Save = Normal_Result_Strg_List
        # arrayで保存するファイル名
        Array_File_Name_to_Save = Normal_Result_Strg_File_Name
        # textで保存するファイル名
        Text_File_Name_to_Save = Normal_Result_Strg_Text_File_Name
        # 保存する関数
        Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
        
        del Array_to_Save

    # 20240223追加　ローカルリストの削除
    del Abnormal_Result_Strg_List, Normal_Result_Strg_List
    del All_Data, All_Data_Sel, All_Result, All_Result_DIF_Dict
    del All_Result_STD_Dict, All_Data_STD, Beam_Data, HOM_Data_Sel, Beam_Data_STD, Record_Name_bp
    del Record_Data_p, Record_List
    del Abnormal_Result_Strg_List_Tmp, Normal_Result_Strg_List_Tmp
    del Record_Name_b, Record_Data, Record_Name_b_Sel, Record_Data_Sel
    del Record_Name_STD_b, Record_Data_STD
    del Record_Data_Sel_f, Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3
    del Record_Data_STD_f, Beam_Data_STD_f, Record_Data_STD_f3
    del Record_Data_f, Record_Data_f3, Beam_Data_f, HOM_Data_f
    del Cal_f, Cal_f_r, Cal_Sel_f, Cal_Sel_f_r

    # メモリーの開放
    gc.collect()
    
    return

# ==============================================================================================================
   
def Save_Manual_Strg_NB(self, Method, Mode_Para, List_Para, Check_Record_Name, Save_Class, Abort_Timing):
    
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    
    # STDのモード
    Mode_Para_S = 'STD_Strg'
    
    # クラス化するためにデータを選別した方法
    Method_d = 'Manual'
    # Manualで選別したAbnormal, Normalのファイルがあるか？無ければ作り、あればロードする
    Abnormal_Result_Strg_File_Name = Ring_Name + '_' + Method_d +'_Abnormal_Class2_Result_Strg_NB.npy'
    Abnormal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method_d +'_Abnormal_Class2_Result_Strg_NB.txt'
    Abnormal_Result_Strg_List = Check_Strg_Abnormal_Result_File(Abnormal_Result_Strg_File_Name, 
                                                               Abnormal_Result_Strg_Text_File_Name)
    Normal_Result_Strg_File_Name = Ring_Name + '_' + Method_d +'_Normal_Class2_Result_Strg_NB.npy'
    Normal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method_d +'_Normal_Class2_Result_Strg_NB.txt'
    Normal_Result_Strg_List = Check_Strg_Normal_Result_File(Normal_Result_Strg_File_Name, 
                                                               Normal_Result_Strg_Text_File_Name)
    
    # ファイルからDate_Range_STDを読み込む    
    f = open(Ring_Name + "_Date_Range_STD_File.txt", "r")
    Date_Range_STD = f.readline()
    f.close()
    
    # ファイルからDate_Range_DIFを読み込む    
    f = open(Ring_Name + "_Date_Range_DIF_Strg_File.txt", "r")
    Date_Range_DIF_Strg = f.readline()
    Date_Range_DIF = Date_Range_DIF_Strg
    f.close()
    
    # 測定値と、STDの回帰曲線パラメータで計算した値でのRMSEを計算し、比べる(DIF_Strg_NB)
    
    # DIFモードの、リングのすべてのレコードのデータをロードする(DIF_Strg)
    All_Data_File_Name = Ring_Name + '_' + Mode_Para + '_All_Record_Data_' + Date_Range_DIF + '.npz'
    All_Data = np.load(All_Data_File_Name)['arr_0']
    
    # DIFモードの、計算に用いたリングのすべてのレコードのデータをロードする。
    All_Data_File_Name_Sel = Ring_Name + '_' + Mode_Para + '_All_Record_Data_Sel_' + Date_Range_DIF + '.npz'
    All_Data_Sel = np.load(All_Data_File_Name_Sel)['arr_0']
    
    # DIFモードの、リングのすべてのレコードの解析結果をロードする。リスト型辞書(DIF_Strg)
    All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para + '_All_Result_Data_' + Date_Range_DIF + '_Dict.npz'
    All_Result = np.load(All_Result_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_DIF_Dict = All_Result.tolist()
    
    # STDモードの、リングのすべてのレコードの解析結果をロードする。リスト型辞書(STD_Strg_NB)
    All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para_S + '_' + Method + '_Class2_All_Result_Data_' + \
    Date_Range_STD + '_Dict_NB.npz'
    All_Result = np.load(All_Result_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_STD_Dict = All_Result.tolist()
    
    # STDモードの、リングのすべてのレコードのデータをロードする。
    All_Data_File_Name_STD = Ring_Name + '_' + Mode_Para_S + '_' + Method + '_Class2_All_Record_Data_' + \
    Date_Range_STD + '.npz'
    All_Data_STD = np.load(All_Data_File_Name_STD)['arr_0']
    
    # STDモードの、リングの解析に使用したすべてのデータをロードする。
    All_Data_File_Name_STD_Sel =  Ring_Name + '_' + Mode_Para_S + '_' + Method + '_Class2_All_Record_Data_Sel_' + \
    Date_Range_STD + '_NB.npz'
    All_Data_STD_Sel = np.load(All_Data_File_Name_STD_Sel)['arr_0']
    
    # DIFのビーム電流、HOM、該当するレコード名の列を抽出する。
    # DIFのビーム電流: BEAMCURRENT: 列番号 = 1
    Beam_Data = All_Data[:, 1:2]
    # DIFの計算に使用したビーム電流: BEAMCURRENT: 列番号 = 0
    Beam_Data_Sel = All_Data_Sel[:, 0:1]
    
    # DIFのHOM: (I*I/Nb)^2: 列番号 = 2
    HOM_Data = All_Data[:, 2:3]
    # DIFの計算に使用したHOM: (I*I/Nb)^2: 列番号 = 1
    HOM_Data_Sel = All_Data_Sel[:, 1:2]
    
    # STDのビーム電流: BEAMCURRENT: 列番号 = 1
    Beam_Data_STD = All_Data_STD[:, 1:2]
    # STDの計算に使用したビーム電流: BEAMCURRENT: 列番号 = 0
    Beam_Data_STD_Sel = All_Data_STD_Sel[:, 0:1]
    # STDのHOM: (I*I/Nb)^2: 列番号 = 2
    HOM_Data_STD = All_Data_STD[:, 2:3]
    # STDの計算に使用したHOM: (I*I/Nb)^2: 列番号 = 1
    HOM_Data_STD_Sel = All_Data_STD_Sel[:, 1:2]
    
    # DIF、STDのレコードの名前のブーリアンリスト：最後に'PRES'がある列
    Record_Name_bp = [bool(All_Data[0, i][-4:] == 'PRES') for i in range(len(All_Data[0]))]
    # DIF、STDのレコードの名前の列をブーリアンで抜き出す
    Record_Data_p = All_Data[:, Record_Name_bp] 
    Record_List = Record_Data_p[0]

    # 一時ファイル
    Abnormal_Result_Strg_List_Tmp = np.empty((0, 26))
    Normal_Result_Strg_List_Tmp = np.empty((0, 26))
    
    # Record_Listの再定義(必要なrecord名のみ)
    Record_List = [Check_Record_Name]
    
    for Record_Name in Record_List:
        # レコード名のブーリアン
        Record_Name_b = [bool(All_Data[0, i] == Record_Name) for i in range(len(All_Data[0]))]
        # レコードの名前の列をブーリアンで抜き出す
        Record_Data = All_Data[:, Record_Name_b]
        
        # DIFの計算に使用したレコード名のブーリアン
        Record_Name_b_Sel = [bool(All_Data_Sel[0, i] == Record_Name) for i in range(len(All_Data_Sel[0]))]
        # DIFの計算に使用したレコードの名前の列をブーリアンで抜き出す
        Record_Data_Sel = All_Data_Sel[:, Record_Name_b_Sel]
        
        # STDのレコードのブーリアン
        Record_Name_STD_b = [bool(All_Data_STD[0, i] == Record_Name) for i in range(len(All_Data_STD[0]))]
        # STDのレコードの名前の列をブーリアンで抜き出す
        Record_Data_STD = All_Data_STD[:, Record_Name_STD_b]

        # STDの計算に使用したレコードのブーリアン
        Record_Name_STD_Sel_b = [bool(All_Data_STD_Sel[0, i] == Record_Name) for i in range(len(All_Data_STD_Sel[0]))]
        # STDの計算に使用したレコードの名前の列をブーリアンで抜き出す
        Record_Data_STD_Sel = All_Data_STD_Sel[:, Record_Name_STD_Sel_b]
        
        # DIFの計算に使用した1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
        Record_Data_Sel_f = np.array([float(s) for s in np.delete(Record_Data_Sel, 0)])
        Beam_Data_Sel_f = np.array([float(s) for s in np.delete(Beam_Data_Sel, 0)])
        HOM_Data_Sel_f = np.array([float(s) for s in np.delete(HOM_Data_Sel, 0)])
        # DIFの圧力読み値(Record_Data_Sel_f)を3倍する
        Record_Data_Sel_f3 = Record_Data_Sel_f * 3.
        
        # STDの1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
        Record_Data_STD_f = np.array([float(s) for s in np.delete(Record_Data_STD, 0)])
        Beam_Data_STD_f = np.array([float(s) for s in np.delete(Beam_Data_STD, 0)])
        # HOM_Data_STD_f = np.array([float(s) for s in np.delete(HOM_Data_STD, 0)])
        # STDの圧力読み値(Record_Data_STD_f)を3倍する
        Record_Data_STD_f3 = Record_Data_STD_f * 3.
        # STDの最大値
        Record_STD_Max = np.max(Record_Data_STD_f3)
        
        # STDの該当するレコード名の解析結果を辞書から抽出する。
        list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_STD_Dict))
        dict_search = list_search[0]
        w0 = dict_search['W0(Beam current)']
        w1 = dict_search['W1(HOM^2)']
        w2 = dict_search['W2(Base)']
        rmse = dict_search['RMSE']
        maxrse = dict_search['MaxRSE']
        W =[float(w0), float(w1), float(w2)]
        
        # DIFの該当するレコード名の解析結果を辞書から抽出する。
        list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_DIF_Dict))
        dict_search = list_search[0]
        w0_dif = dict_search['W0(Beam current)']
        w1_dif = dict_search['W1(HOM^2)']
        w2_dif = dict_search['W2(Base)']
        rmse_dif = dict_search['RMSE']
        maxrse_dif = dict_search['MaxRSE']
        W_dif =[float(w0_dif), float(w1_dif), float(w2_dif)]
        
        # divide zero対策
        # 20240120 変更
        if(float(rmse) <= 1.e-8):
            rmse = '1.e-8'
        
        # すべてのデータ
        Record_Data_f = np.array([float(s) for s in np.delete(Record_Data, 0)])
        Record_Data_f3 = Record_Data_f * 3.
        Beam_Data_f = np.array([float(s) for s in np.delete(Beam_Data, 0)])
        HOM_Data_f = np.array([float(s) for s in np.delete(HOM_Data, 0)])

        # 最大値
        Record_Max = np.max(Record_Data_f3)
        Beam_Max = np.max(Beam_Data_f)
        # 計算に用いた範囲の最大値
        Record_Sel_Max = np.max(Record_Data_Sel_f3)
        # 計算に用いた範囲の平均値
        Record_Sel_Avg = np.mean(Record_Data_Sel_f3)
        
        # STDの回帰曲線パラメータで計算した値
        Cal_f = Beam_Data_Sel_f * float(w0) + HOM_Data_Sel_f * float(w1) + float(w2)
        Cal_f_r = np.maximum(Cal_f, 3.e-8)
        Cal_Sel_f = Beam_Data_Sel_f * float(w0) + HOM_Data_Sel_f * float(w1) + float(w2)
        Cal_Sel_f_r = np.maximum(Cal_Sel_f, 3.e-8)
        # 最大値
        Cal_Max = np.max(Cal_f_r)
        # 計算に用いた範囲の最大
        Cal_Sel_Max = np.max(Cal_Sel_f_r)
        # 計算に用いた範囲の平均値
        Cal_Sel_Avg = np.mean(Cal_Sel_f_r)
        
        # 各レコードのMSE、RMSE、最大のSE、RSEの計算(DIF): STDの回帰曲線を使う
        mse_cal, maxse_cal = mse_plane(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3, W)
        rmse_cal = np.sqrt(mse_cal)
        maxrse_cal = np.sqrt(maxse_cal)
        
        # STDのパラメータで計算したrms
        Record_rms = rmse_cal
        # for i in range(len(Record_Data_Sel_f3) - 1):
        #     Record_rms = Record_rms + (Record_Data_Sel_f3[i] - float(w2))**2
        # Record_rms = np.sqrt(Record_rms / (len(Record_Data_Sel_f3)))
        
        # 一時ファイルへの追加
        if(Save_Class == 'Nor'):
            Normal_Result_Strg_List_Tmp = np.vstack((Normal_Result_Strg_List_Tmp, np.array([Date_Range_DIF, Date_Range_STD, 
                                    Abort_Timing, Record_Name, '{:.2e}'.format(rmse_cal), '{:.2e}'.format(float(rmse)), 
                                    '{:.2e}'.format(Beam_Max), '{:.3e}'.format(float(maxrse)), 
                                    '{:.2e}'.format(Record_Max), 
                                    '{:.2e}'.format(Record_Sel_Max), '{:.2e}'.format(Record_Sel_Avg), 
                                    '{:.2e}'.format(Cal_Sel_Max), '{:.2e}'.format(Cal_Sel_Avg),
                                    '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                    '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                    '{:.2e}'.format(float(w0)), '{:.2e}'.format(float(w1)),
                                    '{:.2e}'.format(float(w2)), '{:.2e}'.format(float(w0_dif)),
                                    '{:.2e}'.format(float(w1_dif)), '{:.2e}'.format(float(w2_dif)),
                                    '{:.2e}'.format(float(rmse_dif)), '{:.2e}'.format(float(maxrse_dif)),
                                    '{:.2e}'.format(0.)])))
        if(Save_Class == 'Abn'):
            # subwindowで推定原因を尋ねる 20240218変更　-----------------------------------------------------------------------
            # subwindowのタイトル
            wtitle = Record_Name + ' (Abnormal Strg_NB)'
            
            # Strg_NBのpossible cause選択肢
            self.Cause_List = ['Leak or Pump failure?', 'Leak or CCG abnormal?', 'Pumping down or Leak?']
            
            # subwindowをつくる
            sub_win = self.getSubWindow(self.Cause_List, wtitle)
            
            # subwindowが閉じられるのを待つ。
            root.wait_window(sub_win) # サブウィンドウが閉じられるのを待つ
            
            # Possible Causeのインデックス
            mpc = self.Cau_var.get()

            Abnormal_Result_Strg_List_Tmp = np.vstack((Abnormal_Result_Strg_List_Tmp, np.array([Date_Range_DIF, Date_Range_STD, 
                                    Abort_Timing, Record_Name, '{:.2e}'.format(rmse_cal), '{:.2e}'.format(float(rmse)), 
                                    '{:.2e}'.format(Beam_Max), '{:.3e}'.format(float(maxrse)), 
                                    '{:.2e}'.format(Record_Max), 
                                    '{:.2e}'.format(Record_Sel_Max), '{:.2e}'.format(Record_Sel_Avg), 
                                    '{:.2e}'.format(Cal_Sel_Max), '{:.2e}'.format(Cal_Sel_Avg),
                                    '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                    '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                    '{:.2e}'.format(float(w0)), '{:.2e}'.format(float(w1)),
                                    '{:.2e}'.format(float(w2)), '{:.2e}'.format(float(w0_dif)),
                                    '{:.2e}'.format(float(w1_dif)), '{:.2e}'.format(float(w2_dif)),
                                    '{:.2e}'.format(float(rmse_dif)), '{:.2e}'.format(float(maxrse_dif)),
                                    '{:.2e}'.format(float(mpc))])))
    
    # 行数が1行以上あれば
    if(Abnormal_Result_Strg_List_Tmp.shape[0] > 0):
        # 同じ時間ではないなら
        # if(Abnormal_Result_Strg_List_Tmp[0][0] not in Abnormal_Result_Strg_List[1:, 0]):
        #     Abnormal_Result_Strg_List = np.vstack((Abnormal_Result_Strg_List, Abnormal_Result_Strg_List_Tmp))
        # else:
        #     print('The same abort time Abnormal Strg data exist.')
        #     tk.messagebox.showinfo('Attention', 'The same abort time Abnormal Strg NB data exists.')
        
        # 20240402
        # 同じ時間、同じレコード名でないなら
        nrowtmp = Abnormal_Result_Strg_List.shape[0]
        hit = 0
        for i in range(nrowtmp):
            if(Abnormal_Result_Strg_List_Tmp[0][0] == Abnormal_Result_Strg_List[i, 0]):
                if(Abnormal_Result_Strg_List_Tmp[0][3] == Abnormal_Result_Strg_List[i, 3]):
                    hit = 1

        if(hit == 0):
            Abnormal_Result_Strg_List = np.vstack((Abnormal_Result_Strg_List, Abnormal_Result_Strg_List_Tmp))
        else:
            print('The same abort time and record name exit in Abnormal Strg data.')
            tk.messagebox.showinfo('Attention', 'The same abort time and record name exit in Abnormal Strg data.')
            
        # 異常リストを保存する
        # 保存するリスト
        Array_to_Save = Abnormal_Result_Strg_List
        # arrayで保存するファイル名
        Array_File_Name_to_Save = Abnormal_Result_Strg_File_Name
        # textで保存するファイル名
        Text_File_Name_to_Save = Abnormal_Result_Strg_Text_File_Name
        # 保存する関数
        Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
        
        del Array_to_Save
    
    if(Normal_Result_Strg_List_Tmp.shape[0] > 0):
        # 同じ時間ではないなら
        # f(Normal_Result_Strg_List_Tmp[0][0] not in Normal_Result_Strg_List[1:, 0]):
        #     Normal_Result_Strg_List = np.vstack((Normal_Result_Strg_List, Normal_Result_Strg_List_Tmp))
        # else:
        #     print('The same abort time Normal Strg data exist.')
        #     tk.messagebox.showinfo('Attention', 'The same abort time Normal Strg NB data exists.')
        
        # 20240402
        # 同じ時間、同じレコード名でないなら
        nrowtmp = Normal_Result_Strg_List.shape[0]
        hit = 0
        for i in range(nrowtmp):
            if(Normal_Result_Strg_List_Tmp[0][0] == Normal_Result_Strg_List[i, 0]):
                if(Normal_Result_Strg_List_Tmp[0][3] == Normal_Result_Strg_List[i, 3]):
                    hit = 1
                    
        if(hit == 0):
            Normal_Result_Strg_List = np.vstack((Normal_Result_Strg_List, Normal_Result_Strg_List_Tmp))
        else:
            print('The same abort time and record name exit in Normal Strg data.')
            tk.messagebox.showinfo('Attention', 'The same abort time and record name exit in Normal Strg data.')
            
        # 正常リストを保存する
        # 保存するリスト
        Array_to_Save = Normal_Result_Strg_List
        # arrayで保存するファイル名
        Array_File_Name_to_Save = Normal_Result_Strg_File_Name
        # textで保存するファイル名
        Text_File_Name_to_Save = Normal_Result_Strg_Text_File_Name
        # 保存する関数
        Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
        
        del Array_to_Save

    #20240223追加　ローカルリストの削除
    del Abnormal_Result_Strg_List, Normal_Result_Strg_List
    del All_Data, All_Data_Sel, All_Result, All_Result_DIF_Dict
    del All_Result_STD_Dict, All_Data_STD, All_Data_STD_Sel
    del Beam_Data, Beam_Data_Sel, HOM_Data, HOM_Data_Sel
    del Beam_Data_STD, Beam_Data_STD_Sel, HOM_Data_STD, HOM_Data_STD_Sel
    del Record_Name_bp, Record_Data_p, Record_List
    del Abnormal_Result_Strg_List_Tmp, Normal_Result_Strg_List_Tmp
    del Record_Name_b, Record_Data, Record_Name_b_Sel, Record_Data_Sel
    del Record_Name_STD_b, Record_Data_STD, Record_Name_STD_Sel_b, Record_Data_STD_Sel
    del Record_Data_Sel_f, Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3
    del Record_Data_STD_f, Beam_Data_STD_f, Record_Data_STD_f3
    
    #　メモリー開放
    gc.collect()
    
    return

# ==============================================================================================================
# ==============================================================================================================
   
def Find_Abnormal_Strg_NB(Method, Mode_Para, List_Para, File_Save_Para, Plot_Para, Abort_Timing, Class_Method, Check_Record_Name):
    # ビームなし、Method = FNN
    
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    
    # STDのモード
    Mode_Para_S = 'STD_Strg'
    
    # Kerasの結果をロードする
    if(Class_Method == 'Keras'):
        # path = 'model_strg_nb.keras'
        path = 'model_strg_nb.h5'
        is_file = os.path.isfile(path)
        if(is_file == False):
            tk.messagebox.showinfo('File Error', path + 'does not exist')
            return
        # model_strg_nb = keras.models.load_model('model_strg_nb.keras')
        model_strg_nb = keras.models.load_model('model_strg_nb.h5')
        
        # 20240227追加
        # meanとstdをロードする
        # データのファイルがあるか？
        path = 'sms_strg_nb.txt'
        is_file = os.path.isfile(path)
        if(is_file == False):
            tk.messagebox.showinfo('File Error', path + ' does not exist')
            return
        # データのファイルからmeanとstdを読み込む 
        content =[]
        with open (path, encoding = 'utf8', newline = '') as f:
            csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
            for row in csvreader:
                content = content + row #contentは1行のリストになる
        sms_strg_nb = content
        
    # Abnormal, Normalのファイルがあるか？無ければ作り、あればロードする
    Abnormal_Result_Strg_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_NB.npy'
    Abnormal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_NB.txt'
    Abnormal_Result_Strg_List = Check_Strg_Abnormal_Result_File(Abnormal_Result_Strg_File_Name, 
                                                               Abnormal_Result_Strg_Text_File_Name)
    Normal_Result_Strg_File_Name = Ring_Name + '_' + Method +'_Normal_Class2_Result_Strg_NB.npy'
    Normal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method +'_Normal_Class2_Result_Strg_NB.txt'
    Normal_Result_Strg_List = Check_Strg_Normal_Result_File(Normal_Result_Strg_File_Name, 
                                                               Normal_Result_Strg_Text_File_Name)
    
    # Date_Range_STDをロードする
    # データのファイルがあるか？
    path = Ring_Name + '_Date_Range_STD_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        tk.messagebox.showinfo('File Error', path + 'does not exist')
        return
    # データのファイルからDate_Range_STDを読み込む    
    f = open(Ring_Name + "_Date_Range_STD_File.txt", "r")
    Date_Range_STD = f.readline()
    f.close()
    
    # Date_Range_DIFをロードする
    # データのファイルがあるか？
    path = Ring_Name + '_Date_Range_DIF_Strg_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        tk.messagebox.showinfo('File Error', path + 'does not exist')
        return
    # データのファイルからDate_Range_DIFを読み込む    
    f = open(Ring_Name + "_Date_Range_DIF_Strg_File.txt", "r")
    Date_Range_DIF_Strg = f.readline()
    Date_Range_DIF = Date_Range_DIF_Strg
    f.close()
    
    # 違いを調べる　
    # 測定値と、STDの計算した値での平均値を計算し、比べる(DIF_Strg_NB)
    
    # DIFモードの、リングのすべてのレコードのデータをロードする(DIF_Strg)
    All_Data_File_Name = Ring_Name + '_' + Mode_Para + '_All_Record_Data_' + Date_Range_DIF + '.npz'
    All_Data = np.load(All_Data_File_Name)['arr_0']
    
    # DIFモードの、計算に用いたリングのすべてのレコードのデータをロードする。
    All_Data_File_Name_Sel = Ring_Name + '_' + Mode_Para + '_All_Record_Data_Sel_' + Date_Range_DIF + '.npz'
    All_Data_Sel = np.load(All_Data_File_Name_Sel)['arr_0']
    
    # DIFモードの、リングのすべてのレコードの解析結果をロードする。リスト型辞書(DIF_Strg)
    All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para + '_All_Result_Data_' + Date_Range_DIF + '_Dict.npz'
    All_Result = np.load(All_Result_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_DIF_Dict = All_Result.tolist()
    
    # STDモードの、リングのすべてのレコードの解析結果をロードする。リスト型辞書(STD_Strg)(NB)
    All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para_S + '_' + Method + '_Class2_All_Result_Data_' + \
    Date_Range_STD + '_Dict_NB.npz'
    All_Result = np.load(All_Result_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_STD_Dict = All_Result.tolist()
    
    # STDモードの、リングのすべてのレコードのデータをロードする。(NBに依らず)
    All_Data_File_Name_STD = Ring_Name + '_' + Mode_Para_S + '_' + Method + '_Class2_All_Record_Data_' + Date_Range_STD + '.npz'
    All_Data_STD = np.load(All_Data_File_Name_STD)['arr_0']
    
    # STDモードの、リングの解析に使用したすべてのデータを行列として保存する。
    All_Data_File_Name_STD_Sel =  Ring_Name + '_' + Mode_Para_S + '_' + Method + '_Class2_All_Record_Data_Sel_' + \
    Date_Range_STD + '_NB.npz'
    All_Data_STD_Sel = np.load(All_Data_File_Name_STD_Sel)['arr_0']
    
    # DIFのビーム電流、HOM、該当するレコード名の列を抽出する。
    # DIFのビーム電流: BEAMCURRENT: 列番号 = 1
    Beam_Data = All_Data[:, 1:2]
    # DIFの計算に使用したビーム電流: BEAMCURRENT: 列番号 = 0
    Beam_Data_Sel = All_Data_Sel[:, 0:1]
    
    # DIFのHOM: (I*I/Nb)^2: 列番号 = 2
    HOM_Data = All_Data[:, 2:3]
    # DIFの計算に使用したHOM: (I*I/Nb)^2: 列番号 = 1
    HOM_Data_Sel = All_Data_Sel[:, 1:2]
    
    # STDのビーム電流: BEAMCURRENT: 列番号 = 1
    Beam_Data_STD = All_Data_STD[:, 1:2]
    # STDのHOM: (I*I/Nb)^2: 列番号 = 2
    HOM_Data_STD = All_Data_STD[:, 2:3]
    
    # DIF、STDのレコードの名前のブーリアンリスト：最後に'PRES'がある列
    Record_Name_bp = [bool(All_Data[0, i][-4:] == 'PRES') for i in range(len(All_Data[0]))]
    # DIF、STDのレコードの名前の列をブーリアンで抜き出す
    Record_Data_p = All_Data[:, Record_Name_bp] 
    Record_List = Record_Data_p[0]

    # 選別にFNNモデルの結果を使う(SDM)
    
    # if(Method == 'FNN'):
    #     Strg_Class_Result_File_Name = 'ALL_SDM_Class2_Result_Strg_NB.npy'
    #     Strg_Class_Result_Text_File_Name = 'ALL_SDM_Class2_Result_Strg_NB.txt'
    #     Strg_Class_Result_List = Check_Strg_Class_Result_File(Strg_Class_Result_File_Name, Strg_Class_Result_Text_File_Name)
    #     if(Strg_Class_Result_List.shape[0] == 0):
    #         print('no class result')
    #         tk.messagebox.showinfo('Error', '  No FNN (SDM) Strg_NB result.  ')
    #         return
    #         
    #     WV = Strg_Class_Result_List[:-1]
            
    # 202401変更
    Abnormal_Result_Strg_List_Tmp = np.empty((0, 26))
    Normal_Result_Strg_List_Tmp = np.empty((0, 26))
    
    for Record_Name in Record_List:
        # レコード名のブーリアン
        Record_Name_b = [bool(All_Data[0, i] == Record_Name) for i in range(len(All_Data[0]))]
        # レコードの名前の列をブーリアンで抜き出す
        Record_Data = All_Data[:, Record_Name_b]
        
        # DIFの計算に使用したレコード名のブーリアン
        Record_Name_b_Sel = [bool(All_Data_Sel[0, i] == Record_Name) for i in range(len(All_Data_Sel[0]))]
        # DIFの計算に使用したレコードの名前の列をブーリアンで抜き出す
        Record_Data_Sel = All_Data_Sel[:, Record_Name_b_Sel]
        
        # STDのレコードのブーリアン
        Record_Name_STD_b = [bool(All_Data_STD[0, i] == Record_Name) for i in range(len(All_Data_STD[0]))]
        # STDのレコードの名前の列をブーリアンで抜き出す
        Record_Data_STD = All_Data_STD[:, Record_Name_STD_b]

        # DIFの計算に使用した1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
        Record_Data_Sel_f = np.array([float(s) for s in np.delete(Record_Data_Sel, 0)])
        Beam_Data_Sel_f = np.array([float(s) for s in np.delete(Beam_Data_Sel, 0)])
        HOM_Data_Sel_f = np.array([float(s) for s in np.delete(HOM_Data_Sel, 0)])
        # DIFの圧力読み値(Record_Data_Sel_f)を3倍する
        Record_Data_Sel_f3 = Record_Data_Sel_f * 3.
        
        # STDの1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する
        Record_Data_STD_f = np.array([float(s) for s in np.delete(Record_Data_STD, 0)])
        Beam_Data_STD_f = np.array([float(s) for s in np.delete(Beam_Data_STD, 0)])
        HOM_Data_STD_f = np.array([float(s) for s in np.delete(HOM_Data_STD, 0)])
        # STDの圧力読み値(Record_Data_STD_f)を3倍する
        Record_Data_STD_f3 = Record_Data_STD_f * 3.
        # STDの最大値
        Record_STD_Max = np.max(Record_Data_STD_f3)
        
        # STDの該当するレコード名の解析結果を辞書から抽出する。
        list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_STD_Dict))
        dict_search = list_search[0]
        w0 = dict_search['W0(Beam current)']
        w1 = dict_search['W1(HOM^2)']
        w2 = dict_search['W2(Base)']
        rmse = dict_search['RMSE']
        maxrse = dict_search['MaxRSE']
        W =[float(w0), float(w1), float(w2)]
        
        # DIFの該当するレコード名の解析結果を辞書から抽出する。
        list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_DIF_Dict))
        dict_search = list_search[0]
        w0_dif = dict_search['W0(Beam current)']
        w1_dif = dict_search['W1(HOM^2)']
        w2_dif = dict_search['W2(Base)']
        rmse_dif = dict_search['RMSE']
        maxrse_dif = dict_search['MaxRSE']
        W_dif =[float(w0_dif), float(w1_dif), float(w2_dif)]
        
        # divide zero対策
        # 202401変更
        if(float(rmse) <= 1.e-8):
            rmse = '1.e-8'
        
        # 20240302 変更
        if(float(rmse_dif) <= 1.e-8):
            rmse_dif = '1.e-8'
            
        # すべてのデータ
        Record_Data_f = np.array([float(s) for s in np.delete(Record_Data, 0)])
        Record_Data_f3 = Record_Data_f * 3.
        Beam_Data_f = np.array([float(s) for s in np.delete(Beam_Data, 0)])
        HOM_Data_f = np.array([float(s) for s in np.delete(HOM_Data, 0)])

        # 最大値
        Record_Max = np.max(Record_Data_f3)
        Beam_Max = np.max(Beam_Data_Sel_f)
        # 計算に用いた範囲の最大値
        Record_Sel_Max = np.max(Record_Data_Sel_f3)
        # 計算に用いた範囲の平均値
        Record_Sel_Avg = np.mean(Record_Data_Sel_f3)
        
        # STDの回帰曲線パラメータで計算した値
        Cal_f = Beam_Data_Sel_f * float(w0) + HOM_Data_Sel_f * float(w1) + float(w2)
        Cal_f_r = np.maximum(Cal_f, 3.e-8)
        Cal_Sel_f = Beam_Data_Sel_f * float(w0) + HOM_Data_Sel_f * float(w1) + float(w2)
        Cal_Sel_f_r = np.maximum(Cal_Sel_f, 3.e-8)
        # 最大値
        Cal_Max = np.max(Cal_f_r)
        # 計算に用いた範囲の最大
        Cal_Sel_Max = np.max(Cal_Sel_f_r)
        # 計算に用いた範囲の平均値
        Cal_Sel_Avg = np.mean(Cal_Sel_f_r)
        
        # 各レコードのMSE、RMSE、最大のSE、RSEの計算(DIF): STDの回帰曲線を使う
        mse_cal, maxse_cal = mse_plane(Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3, W)
        rmse_cal = np.sqrt(mse_cal)
        maxrse_cal = np.sqrt(maxse_cal)
        
        # STDのパラメータで計算したrms
        Record_rms = rmse_cal

        # 20240201変更
        # 異常かどうかの判断
        Abnormal_Flag_FNN = 0 # 0 -> Normal, 1 -> Abnormal
        Abnormal_Flag_2 = 1 # 0 -> Normal, 1 -> Abnormal
        
        # FNN回帰モデルの結果を使う(SDM)
        if(Class_Method == 'SDM'):
            Strg_Class_Result_File_Name = 'ALL_SDM_Class2_Result_Strg_NB.npy'
            Strg_Class_Result_Text_File_Name = 'ALL_SDM_Class2_Result_Strg_NB.txt'
            Strg_Class_Result_List = Check_Strg_Class_Result_File(Strg_Class_Result_File_Name, Strg_Class_Result_Text_File_Name)
            if(Strg_Class_Result_List.shape[0] == 0):
                print('no class result')
                tk.messagebox.showinfo('Error', '  No FNN (SDM) Strg_NB result.  ')
                return
            
            WV = Strg_Class_Result_List[:-1]
        
            # divide zero対策
            if(float(rmse) < 1.e-8): 
                rmse = '1.e-8'
            x1 = np.log10(rmse_cal / float(rmse))
            x2 = np.log10(rmse_cal / Record_Sel_Avg)
            x3 = np.log10(Record_Sel_Avg / Cal_Sel_Avg)
            
            M = 2
            K = 2
            CX = np.array([[x1, x2, x3]])
            CY, a, z, b =FNN(WV, M, K, CX)
            Cmax = np.argmax(CY)
            
            if(Cmax == 0):
                Abnormal_Flag_FNN = 1
                
            #　共通の条件
            # 20240202変更
            if(Record_Max < 8.e-8) or (Record_Sel_Avg <= Cal_Sel_Avg * 1.1):
                Abnormal_Flag_2 = 0
            
            # ローカルリストの削除　20240221追加
            del x1, x2, x3, CX, CY, a, z, b
            
        # 20240204変更
        # FNN回帰モデルの結果を使う(Keras)
        if(Class_Method == 'Keras'):
            # 入力パラメータは4個)
            X1 = funclog1(rmse_cal)
            X2 = funclog1(float(rmse))
            X3 = funclog1(float(rmse_dif))
            X4 = funclog1(Record_Sel_Avg)
            X5 = funclog1(Record_Sel_Max)
            X6 = funclog1(Cal_Sel_Avg)
            
            X = np.hstack((X1, X2))
            X = np.hstack((X, X3))
            X = np.hstack((X, X4))
            X = np.hstack((X, X5))
            X = np.hstack((X, X6))
            
            X = X[None,:]
            
            # 20240227追加
            # 標準化(Training dataのmeanとstdを使う)
            for i in range(6):
                me1 = float(sms_strg_nb[i * 2])
                si1 = float(sms_strg_nb[1 + i * 2])
                X[:, i] = standard_n1(X[:, i], me1, si1)
                
            Predictions = model_strg_nb.predict_on_batch(X)
            
            Cmax = Predictions[0].argmax()
            
            if(Cmax == 0):
                Abnormal_Flag_FNN = 1

            # 20240202変更
            if(Record_Max < 8.e-8) or (Record_Sel_Avg <= Cal_Sel_Avg * 1.1):
                Abnormal_Flag_2 = 0
            
            # 20240226追加
            keras.backend.clear_session()
            # ローカルリストの削除　20240221追加
            del X1, X2, X3, X4, X5, X6, X, Predictions
            gc.collect()
            
        # 20240201変更
        # 異常だったら一時異常リストに追加
        if(Abnormal_Flag_FNN == 1) and (Abnormal_Flag_2 == 0):
            # print(Record_Name + ' Strg is Normal! ', 'RMSE_CAL = {:.2e}, RMSE_STD = {:.2e}, \
            #         Pre_Max = {:.2e}'.format(Record_rms, float(rmse), float(Record_Max)))
            
            Normal_Result_Strg_List_Tmp = np.vstack((Normal_Result_Strg_List_Tmp, np.array([Date_Range_DIF, Date_Range_STD, 
                                Abort_Timing, Record_Name, '{:.2e}'.format(rmse_cal), '{:.2e}'.format(float(rmse)), 
                                '{:.2e}'.format(Beam_Max), '{:.3e}'.format(float(maxrse)), 
                                '{:.2e}'.format(Record_Max), 
                                '{:.2e}'.format(Record_Sel_Max), '{:.2e}'.format(Record_Sel_Avg), 
                                '{:.2e}'.format(Cal_Sel_Max), '{:.2e}'.format(Cal_Sel_Avg),
                                '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                '{:.2e}'.format(float(w0)), '{:.2e}'.format(float(w1)),
                                '{:.2e}'.format(float(w2)), '{:.2e}'.format(float(w0_dif)),
                                '{:.2e}'.format(float(w1_dif)), '{:.2e}'.format(float(w2_dif)),
                                '{:.2e}'.format(float(rmse_dif)), '{:.2e}'.format(float(maxrse_dif)),
                                '{:.2e}'.format(0.)])))
                
        if(Abnormal_Flag_FNN == 1) and (Abnormal_Flag_2 == 1):
            # print(Record_Name + ' Strg is abnormal! ', 'RMSE_CAL = {:.2e}, RMSE_STD = {:.2e}, \
            #         Pre_Max = {:.2e}'.format(Record_rms, float(rmse), float(Record_Max)))
            
            Abnormal_Result_Strg_List_Tmp = np.vstack((Abnormal_Result_Strg_List_Tmp, np.array([Date_Range_DIF, Date_Range_STD, 
                                Abort_Timing, Record_Name, '{:.2e}'.format(rmse_cal), '{:.2e}'.format(float(rmse)), 
                                '{:.2e}'.format(Beam_Max), '{:.3e}'.format(float(maxrse)), 
                                '{:.2e}'.format(Record_Max), 
                                '{:.2e}'.format(Record_Sel_Max), '{:.2e}'.format(Record_Sel_Avg), 
                                '{:.2e}'.format(Cal_Sel_Max), '{:.2e}'.format(Cal_Sel_Avg),
                                '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                '{:.2e}'.format(float(w0)), '{:.2e}'.format(float(w1)),
                                '{:.2e}'.format(float(w2)), '{:.2e}'.format(float(w0_dif)),
                                '{:.2e}'.format(float(w1_dif)), '{:.2e}'.format(float(w2_dif)),
                                '{:.2e}'.format(float(rmse_dif)), '{:.2e}'.format(float(maxrse_dif)),
                                '{:.2e}'.format(0.)])))
        # 20240207追加
        if(Record_Name == Check_Record_Name):
            rn0 = ['Date_Range_DIF', 'Date_Range_STD', 'Abort_Timing', 'Record Name', 
                                'RMSE_cal', 'RMSE_std', 'Beam_Max', 'MaxRSE_std', 'Max_Pre', 'Max_Sel_Pre', 
                                'Avg_Sel_Pre', 'Max_Cal', 'Avg_Cal', 'RMSE_STEP_Max', 'STEP_Max_Avg', 
                                'RMSE_STEP_Mean', 'STEP_Mean_Avg', 'W0_std', 'W1_std', 'W2_std', 'W0_dif', 
                                'W1_dif', 'W2_dif', 'RMSE_dif', 'MaxRMSE_dif', 'Cause_no']
            rn1 = [Date_Range_DIF, Date_Range_STD, 
                                Abort_Timing, Record_Name, '{:.2e}'.format(rmse_cal), '{:.2e}'.format(float(rmse)), 
                                '{:.2e}'.format(Beam_Max), '{:.3e}'.format(float(maxrse)), 
                                '{:.2e}'.format(Record_Max), 
                                '{:.2e}'.format(Record_Sel_Max), '{:.2e}'.format(Record_Sel_Avg), 
                                '{:.2e}'.format(Cal_Sel_Max), '{:.2e}'.format(Cal_Sel_Avg),
                                '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                '{:.2e}'.format(0.0), '{:.2e}'.format(0.0),
                                '{:.2e}'.format(float(w0)), '{:.2e}'.format(float(w1)),
                                '{:.2e}'.format(float(w2)), '{:.2e}'.format(float(w0_dif)),
                                '{:.2e}'.format(float(w1_dif)), '{:.2e}'.format(float(w2_dif)),
                                '{:.2e}'.format(float(rmse_dif)), '{:.2e}'.format(float(maxrse_dif)),
                                '{:.2e}'.format(0.)]
            rn2 = np.vstack((rn0, rn1))
            print(rn2.T)
        
        # ローカルリストの削除　20240221追加
        del Record_Name_b, Record_Data, Record_Name_b_Sel, Record_Data_Sel, Record_Name_STD_b, 
        del Record_Data_STD, Record_Data_Sel_f, Beam_Data_Sel_f, HOM_Data_Sel_f, Record_Data_Sel_f3
        del Record_Data_STD_f, Beam_Data_STD_f, HOM_Data_STD_f, Record_Data_STD_f3
        del Record_Data_f, Record_Data_f3, Beam_Data_f, HOM_Data_f
        del Cal_f, Cal_Sel_f
            
    # 一時異常リストのプリント
    # print('Abnormal_Result_Strg_List_Tmp =', Abnormal_Result_Strg_List_Tmp)
    
    # もし同じDate_Range_DIFなら確認しない
    if(Abnormal_Result_Strg_List_Tmp.shape[0] > 0): # Abnormalなレコードが一つ以上あれば
        if(Abnormal_Result_Strg_List_Tmp[0][0] in Abnormal_Result_Strg_List[1:, 0]): # Date_Range_DIFが同じかどうか
            print('The same abort time Abnormal Strg data exist.')
            # print('The present Abnormal_Result_Strg_List = ', Abnormal_Result_Strg_List)
            # tk.messagebox.showinfo('Note', 'The same abort time Abnormal Strg NB data exists.')
        
    # 行数が1行以上あり
    if(Abnormal_Result_Strg_List_Tmp.shape[0] > 0):
        # 同じ時間ではないなら
        if(Abnormal_Result_Strg_List_Tmp[0][0] not in Abnormal_Result_Strg_List[1:, 0]):
            Abnormal_Result_Strg_List = np.vstack((Abnormal_Result_Strg_List, Abnormal_Result_Strg_List_Tmp))
        else:
            print('The same abort time Abnormal Strg data exists.')
            # tk.messagebox.showinfo('Attention', 'The same abort time Abnormal Strg NB data exists.')
    else:
        print('No new Abnormal Strg result')
        # tk.messagebox.showinfo('Note', 'No new abnormal Strg result')
    
    if(Normal_Result_Strg_List_Tmp.shape[0] > 0):
        # 同じ時間ではないなら
        if(Normal_Result_Strg_List_Tmp[0][0] not in Normal_Result_Strg_List[1:, 0]):
            Normal_Result_Strg_List = np.vstack((Normal_Result_Strg_List, Normal_Result_Strg_List_Tmp))
        else:
            print('The same abort time Normal Strg data exist.')
            # tk.messagebox.showinfo('Attention', 'The same abort time Normal Strg NB data exists.')
    else:
        print('No new Normal Strg result')
        # tk.messagebox.showinfo('Note', 'No new normal Strg result')
        
    # 結果のプリントと保存
    # print('Abnormal_Result_Strg_List =', Abnormal_Result_Strg_List)
    # print('Normal_Result_Strg_List =', Normal_Result_Strg_List)
    
    # Abnormalにもnormalにも新しいデータが無いならダミーをnormalに追加する。
    if(Abnormal_Result_Strg_List_Tmp.shape[0] == 0) and (Normal_Result_Strg_List_Tmp.shape[0] == 0):
        print('Add dummy')
        if(Method != 'Manual'):
            Normal_Result_Strg_List_Tmp = np.array([Date_Range_DIF, Date_Range_STD, Abort_Timing, 'Dummy', '1.00e-08', '1.00e-08', 
                                                    '{:.3e}'.format(Beam_Max), '1.00e-08', '1.00e-08', '1.00e-08', '1.00e-08', 
                                                    '1.00e-08', '1.00e-08', '1.00e-08', '1.00e-08', '1.00e-08', '1.00e-08', 
                                                    '1.00e-08', '1.00e-08', '1.00e-08', '1.00e-08', '1.00e-08', '1.00e-08', 
                                                    '1.00e-08', '1.00e-08', '0.'])
            Normal_Result_Strg_List = np.vstack((Normal_Result_Strg_List, Normal_Result_Strg_List_Tmp))
            # print('Dummy was added')
            
    # Abnromal Recordの数
    N_rec = Abnormal_Result_Strg_List_Tmp.shape[0]
    
    if (File_Save_Para == 1):
        # 異常リストを保存する
        # 保存するリスト
        Array_to_Save = Abnormal_Result_Strg_List
        # arrayで保存するファイル名
        Array_File_Name_to_Save = Abnormal_Result_Strg_File_Name
        # textで保存するファイル名
        Text_File_Name_to_Save = Abnormal_Result_Strg_Text_File_Name
        # 保存する関数
        Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
        
        # 正常リストを保存する
        # 保存するリスト
        Array_to_Save = Normal_Result_Strg_List
        # arrayで保存するファイル名
        Array_File_Name_to_Save = Normal_Result_Strg_File_Name
        # textで保存するファイル名
        Text_File_Name_to_Save = Normal_Result_Strg_Text_File_Name
        # 保存する関数
        Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
        
        del Array_to_Save
        
    # ローカルリストの削除　20240221追加
    del Abnormal_Result_Strg_List, Normal_Result_Strg_List
    del All_Data, All_Data_Sel, All_Result, All_Result_DIF_Dict, All_Result_STD_Dict, All_Data_STD, All_Data_STD_Sel
    del Beam_Data, Beam_Data_Sel, HOM_Data, HOM_Data_Sel, Beam_Data_STD, HOM_Data_STD
    del Record_Data_p, Record_List
    del Abnormal_Result_Strg_List_Tmp, Normal_Result_Strg_List_Tmp
    if(Class_Method == 'SDM'):
        del Strg_Class_Result_List, WV
    if(Class_Method == 'Keras'):
        del model_strg_nb
    # メモリー開放
    gc.collect()
    
    return N_rec
    
# ==============================================================================================================

def Find_Abnormal_Tail(Method, Mode_Para, List_Para, File_Save_Para, Plot_Para, Abort_Timing, Class_Method, Check_Record_Name):
    # Method = FNN
    
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    
    # STDのモード
    Mode_Para_S = 'STD_Tail'
    
    # Kerasの結果をロードする
    if(Class_Method == 'Keras'):
        # path = 'model_tail_wb.keras'
        path = 'model_tail_wb.h5'
        is_file = os.path.isfile(path)
        if(is_file == False):
            tk.messagebox.showinfo('File Error', path + 'does not exist')
            return
        # model_tail_wb = keras.models.load_model('model_tail_wb.keras')
        model_tail_wb = keras.models.load_model('model_tail_wb.h5')
        
        # 20240227追加
        # meanとstdをロードする
        # データのファイルがあるか？
        path = 'sms_tail_wb.txt'
        is_file = os.path.isfile(path)
        if(is_file == False):
            tk.messagebox.showinfo('File Error', path + 'does not exist')
            return
        # データのファイルからmeanとstdを読み込む 
        content =[]
        with open (path, encoding = 'utf8', newline = '') as f:
            csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
            for row in csvreader:
                content = content + row #contentは1行のリストになる
        sms_tail_wb = content
        
    # Abnormal, Normalのファイルがあるか？
    Abnormal_Result_Tail_File_Name = Ring_Name +'_' + Method +'_Abnormal_Class2_Result_Tail_WB.npy'
    Abnormal_Result_Tail_Text_File_Name = Ring_Name +'_' + Method +'_Abnormal_Class2_Result_Tail_WB.txt'
    Abnormal_Result_Tail_List = Check_Tail_Abnormal_Result_File(Abnormal_Result_Tail_File_Name, 
                                                               Abnormal_Result_Tail_Text_File_Name)
    Normal_Result_Tail_File_Name = Ring_Name +'_' + Method +'_Normal_Class2_Result_Tail_WB.npy'
    Normal_Result_Tail_Text_File_Name = Ring_Name +'_' + Method + '_Normal_Class2_Result_Tail_WB.txt'
    Normal_Result_Tail_List = Check_Tail_Normal_Result_File(Normal_Result_Tail_File_Name, 
                                                                Normal_Result_Tail_Text_File_Name)

    # Date_Range_DIFをロードする
    # データのファイルがあるか？
    path = Ring_Name + '_Date_Range_DIF_Strg_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        tk.messagebox.showinfo('File Error', path + 'does not exist')
        return
    # データのファイルからDate_Range_DIFを読み込む    
    f = open(Ring_Name + "_Date_Range_DIF_Strg_File.txt", "r")
    Date_Range_DIF = f.readline()
    f.close()
    
    # Date_Range_STDをロードする
    # データのファイルがあるか？
    path = Ring_Name + '_Date_Range_STD_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        tk.messagebox.showinfo('File Error', path + 'does not exist')
        return
    # データのファイルからDate_Range_STDを読み込む    
    f = open(Ring_Name + "_Date_Range_STD_File.txt", "r")
    Date_Range_STD = f.readline()
    f.close()
    
    # 違いを調べる　
    # 測定値とSTDの回帰曲線パラメータで計算した値を比べ、RMSEを計算する(DIF_Tail)
    
    # DIFモードのリングのすべてのレコードのデータをロードする（DIF_Tail)。
    All_Data_File_Name =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_' + Date_Range_DIF + '.npz'
    All_Data = np.load(All_Data_File_Name)['arr_0']
    # 規格化していない生データ
    All_Data_File_Name_B =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_B_' + Date_Range_DIF + '.npz'
    All_Data_B = np.load(All_Data_File_Name_B)['arr_0']
    
    # DIFモードの、リングのすべてのレコードの解析結果をロードする。リスト型辞書(DIF_Tail)
    All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para + '_All_Result_Data_' + Date_Range_DIF + '_Dict.npz'
    All_Result = np.load(All_Result_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_DIF_Dict = All_Result.tolist()
    
    # リングのすべてのSTDのレコードのデータをロードする（STD_Tail)。
    All_Data_STD_File_Name =  Ring_Name + '_' + Mode_Para_S +'_' + Method + '_Class2_All_Record_Data_' + Date_Range_STD + '.npz'
    All_Data_STD = np.load(All_Data_STD_File_Name)['arr_0']
    # 規格化していない生データをロードする
    All_Data_STD_File_Name_B =  Ring_Name + '_' + Mode_Para_S +'_' + Method + '_Class2_All_Record_Data_B_' + Date_Range_STD + '.npz'
    All_Data_STD_B = np.load(All_Data_STD_File_Name_B)['arr_0']
    
    # リングのすべてのSTDのレコードの解析結果をロードする。
    All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para_S +'_' + Method + '_Class2_All_Result_Data_' + Date_Range_STD + '_Dict.npz'
    All_Result = np.load(All_Result_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_STD_Dict = All_Result.tolist()
    
    # アボート後の時間ステップ、該当するレコード名の列を抽出する。
    # ステップ: 列番号 = 0
    Time_Step_Data = All_Data[:, 0:1]
    Time_Step_Data_STD = All_Data_STD[:, 0:1]
    
    # レコードの名前のブーリアンリスト：最後に'PRES'がある列
    Record_Name_bp = [bool(All_Data[0, i][-4:] == 'PRES') for i in range(len(All_Data[0]))]
    # レコードの名前の列をブーリアンで抜き出す
    Record_Data_p = All_Data[:, Record_Name_bp] 
    Record_List = Record_Data_p[0]
    
    # Strg中の最大ビーム電流
    # DIFモードの、リングのすべてのレコードのデータをロードする(DIF_Strg)
    All_Data_File_Name_s = Ring_Name + '_DIF_Strg_All_Record_Data_' + Date_Range_DIF + '.npz'
    All_Data_s = np.load(All_Data_File_Name_s)['arr_0']

    # DIFのビーム電流、HOM、該当するレコード名の列を抽出する。
    # DIFのビーム電流: BEAMCURRENT: 列番号 = 1
    Beam_Data_s = All_Data_s[:, 1:2]

    Beam_Data_s_f = np.array([float(s) for s in np.delete(Beam_Data_s, 0)])
    Beam_Max = np.max(Beam_Data_s_f)

    # FNN回帰モデルの結果を使う時 (SDM)
    
    # if(Method == 'FNN'):
    #     Tail_Class_Result_File_Name = 'ALL_SDM_Class2_Result_Tail_WB.npy'
    #     Tail_Class_Result_Text_File_Name = 'ALL_SDM_Class2_Result_Tail_WB.txt'
    #     Tail_Class_Result_List = Check_Tail_Class_Result_File(Tail_Class_Result_File_Name, Tail_Class_Result_Text_File_Name)
    #     if(Tail_Class_Result_List.shape[0] == 0):
    #         print('no class result')
    #         tk.messagebox.showinfo('Error', '  No FNN (SDM) Tail_WB result.  ')
    #         return
    #         
    #     WV = Tail_Class_Result_List[:-1]

    Abnormal_Result_Tail_List_Tmp = np.empty((0, 29))
    Normal_Result_Tail_List_Tmp = np.empty((0, 29))
    
    for Record_Name in Record_List:
        # レコード名のブーリアンリスト：Record_Nameの列(DIF_Tail)
        Record_Name_b = [bool(All_Data[0, i] == Record_Name) for i in range(len(All_Data[0]))]
        # レコード名の列をブーリアンで抜き出す：Record_Nameの列
        Record_Data = All_Data[:, Record_Name_b] 
        # レコード名のブーリアンリスト：Record_Nameの列(DIF_Tail)
        Record_Name_B_b = [bool(All_Data[0, i] == Record_Name) for i in range(len(All_Data_B[0]))]
        # レコード名の列をブーリアンで抜き出す：Record_Nameの列
        Record_Data_B = All_Data_B[:, Record_Name_B_b] 
    
        # 1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する(DIF_Tail)
        Record_Data_f = np.array([float(s) for s in np.delete(Record_Data, 0)])
        Time_Step_Data_f = np.array([float(s) for s in np.delete(Time_Step_Data, 0)])
        Record_Data_B_f = np.array([float(s) for s in np.delete(Record_Data_B, 0)]) - 1.e-8
        Record_Data_B_f3 = Record_Data_B_f * 3.
        # 規格化データの最大値
        Record_Max = np.max(Record_Data_f)
        # 規格化データの平均値
        Record_Avg = np.mean(Record_Data_f)
        # 生データの最大値 (読み値-1.e-8 の3倍）
        Record_B_Max = np.max(Record_Data_B_f3)
        # 生データの平均値 (読み値-1.e-8 の3倍)
        Record_B_Avg = np.mean(Record_Data_B_f3)
        
        # レコード名のブーリアンリスト：Record_Nameの列(STD_Tail)
        Record_Name_b_STD = [bool(All_Data_STD[0, i] == Record_Name) for i in range(len(All_Data_STD[0]))]
        # レコード名の列をブーリアンで抜き出す：Record_Nameの列
        Record_Data_STD = All_Data_STD[:, Record_Name_b_STD] 
        # レコードの名のブーリアンリスト：Record_Nameの列(STD_Tail)
        Record_Name_b_STD_B = [bool(All_Data_STD[0, i] == Record_Name) for i in range(len(All_Data_STD_B[0]))]
        # レコード名の列をブーリアンで抜き出す：Record_Nameの列
        Record_Data_STD_B = All_Data_STD_B[:, Record_Name_b_STD_B] 
    
        # 1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する(STD_Tail)
        Record_Data_STD_f = np.array([float(s) for s in np.delete(Record_Data_STD, 0)])
        Time_Step_Data_STD_f = np.array([float(s) for s in np.delete(Time_Step_Data_STD, 0)])
        Record_Data_STD_B_f = np.array([float(s) for s in np.delete(Record_Data_STD_B, 0)]) - 1.e-8
        # STDの規格化する前のデータ-1.e-8　の3倍
        Record_Data_STD_B_f3 = Record_Data_STD_B_f * 3.
        # STDの規格化データの最大値
        Record_STD_Max = np.max(Record_Data_STD_f)
        # STDの規格化データの平均値
        Record_STD_Avg = np.mean(Record_Data_STD_f)
        # STDの生データの最大値(読み値-1.e-8 の3倍)
        Record_STD_B_Max = np.max(Record_Data_STD_B_f3)
        # STDの生データの平均値(読み値-1.e-8 の3倍)
        Record_STD_B_Avg = np.mean(Record_Data_STD_B_f3)
        
        # 該当するレコード名の解析結果を辞書から抽出する。(STD_Tail)
        list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_STD_Dict))
        dict_search = list_search[0]
    
        # 202401変更
        # model_C
        w0 = dict_search['W0']
        w1 = dict_search['W1']
        w2 = dict_search['W2']
        w3 = dict_search['W3']
        w4 = dict_search['W4']
        w4 = '0'
        rmse = dict_search['RMSE']
        maxrse = dict_search['MaxRSE']
        W = [float(w0), float(w1), float(w2), float(w3)]
    
        # 該当するレコード名の解析結果を辞書から抽出する。(DIF_Tail)
        list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_DIF_Dict))
        dict_search = list_search[0]
    
        # model_C
        w0_dif = dict_search['W0']
        w1_dif = dict_search['W1']
        w2_dif = dict_search['W2']
        w3_dif = dict_search['W3']
        # w3_dif = '0'
        w4_dif = dict_search['W4']
        w4_dif = '0'
        rmse_dif = dict_search['RMSE']
        maxrse_dif = dict_search['MaxRSE']
        W_dif = [float(w0_dif), float(w1_dif), float(w2_dif), float(w3_dif)]
        
        if(float(rmse) <= 1.e-2):
            rmse = '1.e-2'
        
        # 20240302 変更
        if(float(rmse_dif) <= 1.e-2):
            rmse_dif = '1.e-2'
            
        # 各レコードのMSEの計算
        mse_cal = mse_model_C(W, Time_Step_Data_f, Record_Data_f)
        rmse_cal = np.sqrt(mse_cal)
        
        # 各レコードの最大SEの計算
        maxse_cal = maxse_model_C(W, Time_Step_Data_f, Record_Data_f)
        maxrse_cal = np.sqrt(maxse_cal)
        
        # 回帰曲線の最大値
        Cal_Max = np.max(model_C(Time_Step_Data_f, W))
        # 回帰曲線の平均値
        Cal_Avg = np.mean(model_C(Time_Step_Data_f, W))
        
        # 異常かどうかの判断
        Abnormal_Flag_FNN = 0 # 0 -> Normal, 1 -> Abnormal
        Abnormal_Flag_2 = 1 # 0 -> Normal, 1 -> Abnormal
        
        # 20240201変更
        # FNN回帰モデルの結果を使う(SDM)
        if(Class_Method == 'SDM'):
            Tail_Class_Result_File_Name = 'ALL_SDM_Class2_Result_Tail_WB.npy'
            Tail_Class_Result_Text_File_Name = 'ALL_SDM_Class2_Result_Tail_WB.txt'
            Tail_Class_Result_List = Check_Tail_Class_Result_File(Tail_Class_Result_File_Name, Tail_Class_Result_Text_File_Name)
            if(Tail_Class_Result_List.shape[0] == 0):
                print('no class result')
                tk.messagebox.showinfo('Error', '  No FNN (SDM) Tail_WB result.  ')
                return
            
            WV = Tail_Class_Result_List[:-1]
        
            # divide zero対策
            if(float(rmse) < 1.e-8):
                rmse = '1.e-8'
            x1 = np.log10(rmse_cal / float(rmse))
            x2 = np.log10(rmse_cal)
            
            # 中間層 M = 2 次元入力(バイアスを入れて3個）、K = 2 クラス分け
            M = 2
            K = 2
            CX = np.array([[x1, x2]])
            CY, a, z, b = FNN(WV, M, K, CX)
            # 確率が最大となるクラス(インデックス): 0ならAbnormal
            Cmax = np.argmax(CY)
            if(Cmax == 0):
                Abnormal_Flag_FNN = 1
                
            # 共通の条件
            # 規格化する前の値(読み値-1.e-8 の3倍)
            # 20240202変更
            if(Record_B_Max < 1.e-7) or (Record_Max < Record_STD_Max * 1.2):
                Abnormal_Flag_2 = 0
            
            # ローカルリストの削除 20240221追加
            del x1, x2, CX, CY, a, z, b
            gc.collect()
            
        # 20240204変更
        # FNN回帰モデルの結果を使う(Keras)
        if(Class_Method == 'Keras'):
            # 入力パラメータは4個)
            X1 = funclog1(rmse_cal)
            X2 = funclog1(float(rmse))
            X3 = funclog1(float(rmse_dif))
            X4 = funclog1(Record_Avg)
            X5 = funclog1(Record_Max)
            X6 = funclog1(Cal_Avg)
            
            X = np.hstack((X1, X2))
            X = np.hstack((X, X3))
            X = np.hstack((X, X4))
            X = np.hstack((X, X5))
            X = np.hstack((X, X6))
            
            X = X[None,:]
            
            # 20240227追加
            # 標準化(Training dataのmeanとstdを使う)
            for i in range(6):
                me1 = float(sms_tail_wb[i * 2])
                si1 = float(sms_tail_wb[1 + i * 2])
                X[:, i] = standard_n1(X[:, i], me1, si1)
                
            Predictions = model_tail_wb.predict_on_batch(X)
            
            Cmax = Predictions[0].argmax()
            
            if(Cmax == 0):
                Abnormal_Flag_FNN = 1

            # 20240207変更
            if(Record_B_Max < 1.e-7) or (Record_Max < Record_STD_Max * 1.2):
                Abnormal_Flag_2 = 0
            
            # 20240226追加
            keras.backend.clear_session()
            # ローカルリストの削除 20240221追加
            del X1, X2, X3, X4, X5, X6, X, Predictions
            gc.collect()
            
        # 異常だったら一時異常リストに追加
        if(Abnormal_Flag_2 == 0) and (Abnormal_Flag_FNN == 1):
            # print(Record_Name + ' Tail is normal! ', 'RMSE_CAL = {:.2e}, RMSE_STD = {:.2e}, Max = {:.2e}'.format(rmse_cal, 
            #                                                                      float(rmse), float(Record_Max)))
            
            Normal_Result_Tail_List_Tmp = np.vstack((Normal_Result_Tail_List_Tmp, np.array([Date_Range_DIF, Date_Range_STD, 
                            Abort_Timing, Record_Name, '{:.2e}'.format(rmse_cal), '{:.2e}'.format(float(rmse)), 
                            '{:.2e}'.format(Beam_Max), '{:.3e}'.format(float(maxrse)), '{:.2e}'.format(Record_Max),
                            '{:.2e}'.format(Record_Avg), '{:.2e}'.format(Cal_Max), '{:.2e}'.format(Cal_Avg), 
                            '{:.2e}'.format(Record_B_Max), '{:.2e}'.format(Record_B_Avg),
                            '{:.2e}'.format(Record_STD_B_Max), '{:.2e}'.format(Record_STD_B_Avg),
                            '{:.2e}'.format(float(w0)), '{:.2e}'.format(float(w1)), '{:.2e}'.format(float(w2)), 
                            '{:.2e}'.format(float(w3)), '{:.2e}'.format(float(w4)), '{:.2e}'.format(float(w0_dif)), 
                            '{:.2e}'.format(float(w1_dif)), '{:.2e}'.format(float(w2_dif)), '{:.2e}'.format(float(w3_dif)), 
                            '{:.2e}'.format(float(w4_dif)), '{:.2e}'.format(float(rmse_dif)), 
                            '{:.2e}'.format(float(maxrse_dif)), '{:.2e}'.format(0.0)])))
            
        if(Abnormal_Flag_2 == 1) and (Abnormal_Flag_FNN == 1):
            # print(Record_Name + ' Tail is abnormal! ', 'RMSE_CAL = {:.2e}, RMSE_STD = {:.2e}, Max = {:.2e}'.format(rmse_cal, 
            #                                                                     float(rmse), float(Record_Max)))
            
            Abnormal_Result_Tail_List_Tmp = np.vstack((Abnormal_Result_Tail_List_Tmp, np.array([Date_Range_DIF, Date_Range_STD, 
                            Abort_Timing, Record_Name, '{:.2e}'.format(rmse_cal), '{:.2e}'.format(float(rmse)), 
                            '{:.2e}'.format(Beam_Max), '{:.3e}'.format(float(maxrse)), '{:.2e}'.format(Record_Max),
                            '{:.2e}'.format(Record_Avg), '{:.2e}'.format(Cal_Max), '{:.2e}'.format(Cal_Avg), 
                            '{:.2e}'.format(Record_B_Max), '{:.2e}'.format(Record_B_Avg),
                            '{:.2e}'.format(Record_STD_B_Max), '{:.2e}'.format(Record_STD_B_Avg),
                            '{:.2e}'.format(float(w0)), '{:.2e}'.format(float(w1)), '{:.2e}'.format(float(w2)), 
                            '{:.2e}'.format(float(w3)), '{:.2e}'.format(float(w4)), '{:.2e}'.format(float(w0_dif)), 
                            '{:.2e}'.format(float(w1_dif)), '{:.2e}'.format(float(w2_dif)), '{:.2e}'.format(float(w3_dif)), 
                            '{:.2e}'.format(float(w4_dif)), '{:.2e}'.format(float(rmse_dif)), 
                            '{:.2e}'.format(float(maxrse_dif)), '{:.2e}'.format(0.0)])))
        # 20240207追加
        if(Record_Name == Check_Record_Name):
            rn0 = ['Date_Range_DIF', 'Date_Range_STD', 'Abort_Timing', 'Record Name', 'RMSE_cal', 
                            'RMSE_std','Beam_Max', 'MaxRSE_std', 'Max_Pre', 'Avg_Pre', 'Max_Cal', 
                             'Avg_Cal', 'Max_Raw_Pre', 'Avg_Raw_Pre', 'Max_STD_Raw_Pre', 'Avg_STD_Raw_Pre',
                             'W0_std', 'W1_std', 'W2_std', 'W3_std', 'W4_std', 'W0_dif', 'W1_dif', 'W2_dif',
                             'W3_dif', 'W4_dif', 'RMSE_dif', 'MaxRSE_dif', 'Cause_no']
            rn1 = [Date_Range_DIF, Date_Range_STD, 
                            Abort_Timing, Record_Name, '{:.2e}'.format(rmse_cal), '{:.2e}'.format(float(rmse)), 
                            '{:.2e}'.format(Beam_Max), '{:.3e}'.format(float(maxrse)), '{:.2e}'.format(Record_Max),
                            '{:.2e}'.format(Record_Avg), '{:.2e}'.format(Cal_Max), '{:.2e}'.format(Cal_Avg), 
                            '{:.2e}'.format(Record_B_Max), '{:.2e}'.format(Record_B_Avg),
                            '{:.2e}'.format(Record_STD_B_Max), '{:.2e}'.format(Record_STD_B_Avg),
                            '{:.2e}'.format(float(w0)), '{:.2e}'.format(float(w1)), '{:.2e}'.format(float(w2)), 
                            '{:.2e}'.format(float(w3)), '{:.2e}'.format(float(w4)), '{:.2e}'.format(float(w0_dif)), 
                            '{:.2e}'.format(float(w1_dif)), '{:.2e}'.format(float(w2_dif)), '{:.2e}'.format(float(w3_dif)), 
                            '{:.2e}'.format(float(w4_dif)), '{:.2e}'.format(float(rmse_dif)), 
                            '{:.2e}'.format(float(maxrse_dif)), '{:.2e}'.format(0.0)]
            rn2 = np.vstack((rn0, rn1))
            print(rn2.T)
            
        # ローカルリストの削除 20240221追加
        del Record_Name_b, Record_Data, Record_Name_B_b, Record_Data_B
        del Record_Data_f, Time_Step_Data_f, Record_Data_B_f, Record_Data_B_f3
        del Record_Name_b_STD, Record_Data_STD,  Record_Name_b_STD_B, Record_Data_STD_B
        del Record_Data_STD_f, Time_Step_Data_STD_f, Record_Data_STD_B_f, Record_Data_STD_B_f3
        del list_search, dict_search

    # 一時異常リストのプリント
    # print('Abnormal_Result_Tail_List_Tmp =', Abnormal_Result_Tail_List_Tmp)
    
    # もし同じDate_Range_DIFなら確認しない
    if(Abnormal_Result_Tail_List_Tmp.shape[0] > 0): # Abnormalなレコードが一つ以上あれば
        if(Abnormal_Result_Tail_List_Tmp[0][0] in Abnormal_Result_Tail_List[1:, 0]): # Date_Range_DIFが同じかどうか
            print('The same abort time abnormal Tail data exist.')
            # print('The present Abnormal_Result_Tail_List = ', Abnormal_Result_Tail_List)
        
    # 行数が1行以上あり
    if(Abnormal_Result_Tail_List_Tmp.shape[0] > 0):
        # 同じ時間ではないなら
        if(Abnormal_Result_Tail_List_Tmp[0][0] not in Abnormal_Result_Tail_List[1:, 0]):
            Abnormal_Result_Tail_List = np.vstack((Abnormal_Result_Tail_List, Abnormal_Result_Tail_List_Tmp))
        else:
            print('The same abort time Abnormal Tail data exist.')
            # tk.messagebox.showinfo('Note', 'The same abort time Abnormal Tail data exists.')
    else:
        print('No new abnormal Tail result')

    if(Normal_Result_Tail_List_Tmp.shape[0] > 0):
        # 同じ時間ではないなら
        if(Normal_Result_Tail_List_Tmp[0][0] not in Normal_Result_Tail_List[1:, 0]):
            Normal_Result_Tail_List = np.vstack((Normal_Result_Tail_List, Normal_Result_Tail_List_Tmp))
        else:
            print('The same abort time Normal Tail data exist.')
            # tk.messagebox.showinfo('Note', 'The same abort time Normal Tail data exists.')
    else:
        print('No new normal Tail result')
        # tk.messagebox.showinfo('Note', 'No new normal Tail result')
    
    # Abnormalにもnormalにも新しいデータが無いならダミーをnormalに追加する。
    if(Abnormal_Result_Tail_List_Tmp.shape[0] == 0) and (Normal_Result_Tail_List_Tmp.shape[0] == 0):
        if(Method != 'Manual'):
            Normal_Result_Tail_List_Tmp = np.array([Date_Range_DIF, Date_Range_STD, Abort_Timing, 'Dummy', '1.00e-01', '1.00e-02', 
                                                    '{:.3e}'.format(Beam_Max), '1.00e-01','1.00e-01', '1.00e-01', '1.00e-01', 
                                                    '1.00e-01', '1.00e-07', '1.00e-07', '1.00e-07', '1.00e-07', '1.00e-07', 
                                                    '1.00e-07', '1.00e-07', '1.00e-07', '0.', '1.00e-07', '1.00e-07', 
                                                    '1.00e-07', '1.00e-07', '0.', '1.00e-07', '1.00e-07', '0'])
            Normal_Result_Tail_List = np.vstack((Normal_Result_Tail_List, Normal_Result_Tail_List_Tmp))
            
    # Abnromal Recordの数
    N_rec = Abnormal_Result_Tail_List_Tmp.shape[0]
    
    if (File_Save_Para == 1):
        # 異常リストを保存する
        # 保存するリスト
        Array_to_Save = Abnormal_Result_Tail_List
        # arrayで保存するファイル名
        Array_File_Name_to_Save = Abnormal_Result_Tail_File_Name
        # textで保存するファイル名
        Text_File_Name_to_Save = Abnormal_Result_Tail_Text_File_Name
        # 保存する関数
        Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
        # 正常リストを保存する
        # 保存するリスト
        Array_to_Save = Normal_Result_Tail_List
        # arrayで保存するファイル名
        Array_File_Name_to_Save = Normal_Result_Tail_File_Name
        # textで保存するファイル名
        Text_File_Name_to_Save = Normal_Result_Tail_Text_File_Name
        # 保存する関数
        Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
        
        del Array_to_Save
    
    # ローカルリストの削除 20240221追加
    del Abnormal_Result_Tail_List,  Normal_Result_Tail_List
    del All_Data, All_Data_B, All_Result, All_Result_DIF_Dict
    del All_Data_STD, All_Data_STD_B, All_Result_STD_Dict, Time_Step_Data, Time_Step_Data_STD
    del Record_Data_p, Record_List, All_Data_s, Beam_Data_s, Beam_Data_s_f
    del Abnormal_Result_Tail_List_Tmp, Normal_Result_Tail_List_Tmp
    if(Class_Method == 'Keras'):
        del model_tail_wb
    if(Class_Method == 'SDM'):
        del Tail_Class_Result_List, WV
        
    # メモリーの開放
    gc.collect()
    
    return N_rec
    
# ==============================================================================================================

def Find_Abnormal_Tail_NB(Method, Mode_Para, List_Para, File_Save_Para, Plot_Para, Abort_Timing):
    
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    
    # STDのモード
    Mode_Para_S = 'STD_Tail'
    
    # クラス化するためにデータを選別した方法
    # Method_d = 'Manual'
    
    # Abnormal, Normalのファイルがあるか？
    Abnormal_Result_Tail_File_Name = Ring_Name +'_' + Method +'_Abnormal_Class2_Result_Tail_NB.npy'
    Abnormal_Result_Tail_Text_File_Name = Ring_Name +'_' + Method +'_Abnormal_Class2_Result_Tail_NB.txt'
    Abnormal_Result_Tail_List = Check_Tail_Abnormal_Result_File(Abnormal_Result_Tail_File_Name, 
                                                               Abnormal_Result_Tail_Text_File_Name)
    Normal_Result_Tail_File_Name = Ring_Name +'_' + Method +'_Normal_Class2_Result_Tail_NB.npy'
    Normal_Result_Tail_Text_File_Name = Ring_Name +'_' + Method + '_Normal_Class2_Result_Tail_NB.txt'
    Normal_Result_Tail_List = Check_Tail_Normal_Result_File(Normal_Result_Tail_File_Name, 
                                                                Normal_Result_Tail_Text_File_Name)

    # Date_Range_DIFをロードする
    # データのファイルがあるか？
    path = Ring_Name + '_Date_Range_DIF_Strg_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print(3361, 'No Date_Range_DIF_File')
        sys.exit()
    # データのファイルからDate_Range_DIFを読み込む    
    f = open(Ring_Name + "_Date_Range_DIF_Strg_File.txt", "r")
    Date_Range_DIF = f.readline()
    f.close()
    
    # Date_Range_STDをロードする
    # データのファイルがあるか？
    path = Ring_Name + '_Date_Range_STD_File.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        print('No Date_Range_STD_File')
        sys.exit()
    # データのファイルからDate_Range_STDを読み込む    
    f = open(Ring_Name + "_Date_Range_STD_File.txt", "r")
    Date_Range_STD = f.readline()
    f.close()
    
    # 最大電流はゼロ
    Beam_Max = 0.
    
    # Abnormalにもnormalにも新しいデータが無いならダミーをnormalに追加する。
    Normal_Result_Tail_List_Tmp = np.array([Date_Range_DIF, Date_Range_STD, Abort_Timing, 'Dummy', '1.00e-01', '1.00e-02', 
                                                    '{:.3e}'.format(Beam_Max), '1.00e-01', '1.00e-01', '1.00e-01', '1.00e-01', 
                                                    '1.00e-01', '1.00e-07', '1.00e-07', '1.00e-07', '1.00e-07', '1.00e-07', 
                                                    '1.00e-07', '1.00e-07', '1.00e-07', '0.', '1.00e-07', '1.00e-07', 
                                                    '1.00e-07', '1.00e-07', '0.', '1.00e-07', '1.00e-07', '0'])
    Normal_Result_Tail_List = np.vstack((Normal_Result_Tail_List, Normal_Result_Tail_List_Tmp))
            
    # print('Abnormal data were not saved')
    if (File_Save_Para == 1):
        # 異常リストを保存する
        # 保存するリスト
        Array_to_Save = Abnormal_Result_Tail_List
        # arrayで保存するファイル名
        Array_File_Name_to_Save = Abnormal_Result_Tail_File_Name
        # textで保存するファイル名
        Text_File_Name_to_Save = Abnormal_Result_Tail_Text_File_Name
        # 保存する関数
        Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
        # 正常リストを保存する
        # 保存するリスト
        Array_to_Save = Normal_Result_Tail_List
        # arrayで保存するファイル名
        Array_File_Name_to_Save = Normal_Result_Tail_File_Name
        # textで保存するファイル名
        Text_File_Name_to_Save = Normal_Result_Tail_Text_File_Name
        # 保存する関数
        Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
        
        del Array_to_Save
        
    N_rec = 0
    
    # ローカルリストの削除 20240223追加
    del Abnormal_Result_Tail_List,  Normal_Result_Tail_List
    
    # メモリーの開放
    gc.collect()
    
    return N_rec
    
# ==============================================================================================================

def Save_Manual_Tail(self, Method, Mode_Para, List_Para, Check_Record_Name, Save_Class, Abort_Timing):
    
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    
    # STDのモード
    Mode_Para_S = 'STD_Tail'
    
    # クラス化するためにデータを選別した方法
    Method_d = 'Manual'
    
    # Abnormal, Normalのファイルがあるか？
    Abnormal_Result_Tail_File_Name = Ring_Name +'_' + Method_d +'_Abnormal_Class2_Result_Tail_WB.npy'
    Abnormal_Result_Tail_Text_File_Name = Ring_Name +'_' + Method_d +'_Abnormal_Class2_Result_Tail_WB.txt'
    Abnormal_Result_Tail_List = Check_Tail_Abnormal_Result_File(Abnormal_Result_Tail_File_Name, 
                                                               Abnormal_Result_Tail_Text_File_Name)
    Normal_Result_Tail_File_Name = Ring_Name +'_' + Method_d +'_Normal_Class2_Result_Tail_WB.npy'
    Normal_Result_Tail_Text_File_Name = Ring_Name +'_' + Method_d + '_Normal_Class2_Result_Tail_WB.txt'
    Normal_Result_Tail_List = Check_Tail_Normal_Result_File(Normal_Result_Tail_File_Name, 
                                                                Normal_Result_Tail_Text_File_Name)

    # データのファイルからDate_Range_DIFを読み込む    
    f = open(Ring_Name + "_Date_Range_DIF_Strg_File.txt", "r")
    Date_Range_DIF = f.readline()
    f.close()
    
    # データのファイルからDate_Range_STDを読み込む    
    f = open(Ring_Name + "_Date_Range_STD_File.txt", "r")
    Date_Range_STD = f.readline()
    f.close()
    
    # 測定値とSTDの回帰曲線パラメータで計算した値を比べ、RMSEを計算する(DIF_Tail)
    
    # DIFモードのリングのすべてのレコードのデータをロードする（DIF_Tail)。
    All_Data_File_Name =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_' + Date_Range_DIF + '.npz'
    All_Data = np.load(All_Data_File_Name)['arr_0']
    # 規格化していない生データ
    All_Data_File_Name_B =  Ring_Name + '_' + Mode_Para + '_All_Record_Data_B_' + Date_Range_DIF + '.npz'
    All_Data_B = np.load(All_Data_File_Name_B)['arr_0']
    
    # DIFモードの、リングのすべてのレコードの解析結果をロードする。リスト型辞書(DIF_Tail)
    All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para + '_All_Result_Data_' + Date_Range_DIF + '_Dict.npz'
    All_Result = np.load(All_Result_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_DIF_Dict = All_Result.tolist()
    
    # リングのすべてのSTDのレコードのデータをロードする（STD_Tail)。
    All_Data_STD_File_Name =  Ring_Name + '_' + Mode_Para_S +'_' + Method + '_Class2_All_Record_Data_' + Date_Range_STD + '.npz'
    All_Data_STD = np.load(All_Data_STD_File_Name)['arr_0']
    # 規格化していない生データをロードする
    All_Data_STD_File_Name_B =  Ring_Name + '_' + Mode_Para_S +'_' + Method + '_Class2_All_Record_Data_B_' + Date_Range_STD + '.npz'
    All_Data_STD_B = np.load(All_Data_STD_File_Name_B)['arr_0']
    
    # リングのすべてのSTDのレコードの解析結果をロードする。
    All_Result_Data_File_Name = Ring_Name + '_' + Mode_Para_S +'_' + Method + '_Class2_All_Result_Data_' + Date_Range_STD + '_Dict.npz'
    All_Result = np.load(All_Result_Data_File_Name, allow_pickle = 'TRUE')['arr_0']
    All_Result_STD_Dict = All_Result.tolist()
    
    # アボート後の時間ステップ、該当するレコード名の列を抽出する。
    # ステップ: 列番号 = 0
    Time_Step_Data = All_Data[:, 0:1]
    Time_Step_Data_STD = All_Data_STD[:, 0:1]
    
    # レコードの名前のブーリアンリスト：最後に'PRES'がある列
    Record_Name_bp = [bool(All_Data[0, i][-4:] == 'PRES') for i in range(len(All_Data[0]))]
    # レコードの名前の列をブーリアンで抜き出す
    Record_Data_p = All_Data[:, Record_Name_bp] 
    Record_List = Record_Data_p[0]
    
    # Strg中の最大ビーム電流
    # DIFモードの、リングのすべてのレコードのデータをロードする(DIF_Strg)
    All_Data_File_Name_s = Ring_Name + '_DIF_Strg_All_Record_Data_' + Date_Range_DIF + '.npz'
    All_Data_s = np.load(All_Data_File_Name_s)['arr_0']

    # DIFのビーム電流、HOM、該当するレコード名の列を抽出する。
    # DIFのビーム電流: BEAMCURRENT: 列番号 = 1
    Beam_Data_s = All_Data_s[:, 1:2]

    Beam_Data_s_f = np.array([float(s) for s in np.delete(Beam_Data_s, 0)])
    Beam_Max = np.max(Beam_Data_s_f)

    # 一時ファイルの定義
    Abnormal_Result_Tail_List_Tmp = np.empty((0, 29))
    Normal_Result_Tail_List_Tmp = np.empty((0, 29))
    
    # Record_Listの再定義
    Record_List = [Check_Record_Name]
    
    for Record_Name in Record_List:
        # レコード名のブーリアンリスト：Record_Nameの列(DIF_Tail)
        Record_Name_b = [bool(All_Data[0, i] == Record_Name) for i in range(len(All_Data[0]))]
        # レコード名の列をブーリアンで抜き出す：Record_Nameの列
        Record_Data = All_Data[:, Record_Name_b] 
        # レコード名のブーリアンリスト：Record_Nameの列(DIF_Tail)
        Record_Name_B_b = [bool(All_Data[0, i] == Record_Name) for i in range(len(All_Data_B[0]))]
        # レコード名の列をブーリアンで抜き出す：Record_Nameの列
        Record_Data_B = All_Data_B[:, Record_Name_B_b] 
    
        # 1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する(DIF_Tail)
        Record_Data_f = np.array([float(s) for s in np.delete(Record_Data, 0)])
        Time_Step_Data_f = np.array([float(s) for s in np.delete(Time_Step_Data, 0)])
        Record_Data_B_f = np.array([float(s) for s in np.delete(Record_Data_B, 0)]) - 1.e-8
        Record_Data_B_f3 = Record_Data_B_f * 3.
        # 規格化データの最大値
        Record_Max = np.max(Record_Data_f)
        # 規格化データの平均値
        Record_Avg = np.mean(Record_Data_f)
        # 生データの最大値 (読み値-1.e-8 の3倍）
        Record_B_Max = np.max(Record_Data_B_f3)
        # 生データの平均値 (読み値-1.e-8 の3倍)
        Record_B_Avg = np.mean(Record_Data_B_f3)
        
        # レコード名のブーリアンリスト：Record_Nameの列(STD_Tail)
        Record_Name_b_STD = [bool(All_Data_STD[0, i] == Record_Name) for i in range(len(All_Data_STD[0]))]
        # レコード名の列をブーリアンで抜き出す：Record_Nameの列
        Record_Data_STD = All_Data_STD[:, Record_Name_b_STD] 
        # レコードの名のブーリアンリスト：Record_Nameの列(STD_Tail)
        Record_Name_b_STD_B = [bool(All_Data_STD[0, i] == Record_Name) for i in range(len(All_Data_STD_B[0]))]
        # レコード名の列をブーリアンで抜き出す：Record_Nameの列
        Record_Data_STD_B = All_Data_STD_B[:, Record_Name_b_STD_B] 
    
        # 1番目のデータ(2個目のデータ)以降を各floatにし、ベクトルにして、1次元化する(STD_Tail)
        Record_Data_STD_f = np.array([float(s) for s in np.delete(Record_Data_STD, 0)])
        Time_Step_Data_STD_f = np.array([float(s) for s in np.delete(Time_Step_Data_STD, 0)])
        Record_Data_STD_B_f = np.array([float(s) for s in np.delete(Record_Data_STD_B, 0)]) - 1.e-8
        # STDの規格化する前のデータ-1.e-8　の3倍
        Record_Data_STD_B_f3 = Record_Data_STD_B_f * 3.
        # STDの規格化データの最大値
        Record_STD_Max = np.max(Record_Data_STD_f)
        # STDの規格化データの平均値
        Record_STD_Avg = np.mean(Record_Data_STD_f)
        # STDの生データの最大値(読み値-1.e-8 の3倍)
        Record_STD_B_Max = np.max(Record_Data_STD_B_f3)
        # STDの生データの平均値(読み値-1.e-8 の3倍)
        Record_STD_B_Avg = np.mean(Record_Data_STD_B_f3)
        
        # STDの該当するレコード名の解析結果を辞書から抽出する。(STD_Tail)
        list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_STD_Dict))
        dict_search = list_search[0]
    
        # 202401変更
        # model_C
        w0 = dict_search['W0']
        w1 = dict_search['W1']
        w2 = dict_search['W2']
        w3 = dict_search['W3']
        w4 = dict_search['W4']
        w4 = '0'
        rmse = dict_search['RMSE']
        maxrse = dict_search['MaxRSE']
        W = [float(w0), float(w1), float(w2), float(w3)]
    
        # 該当するレコード名の解析結果を辞書から抽出する。(DIF_Tail)
        list_search = list(filter(lambda item : item['Record Name'] == Record_Name, All_Result_DIF_Dict))
        dict_search = list_search[0]
    
        # model_C
        w0_dif = dict_search['W0']
        w1_dif = dict_search['W1']
        w2_dif = dict_search['W2']
        w3_dif = dict_search['W3']
        # w3_dif = '0'
        w4_dif = dict_search['W4']
        w4_dif = '0'
        rmse_dif = dict_search['RMSE']
        maxrse_dif = dict_search['MaxRSE']
        W_dif = [float(w0_dif), float(w1_dif), float(w2_dif), float(w3_dif)]
        
        if(float(rmse) <= 1.e-2):
            rmse = '1.e-2'
        
        # 各レコードのMSEの計算
        mse_cal = mse_model_C(W, Time_Step_Data_f, Record_Data_f)
        rmse_cal = np.sqrt(mse_cal)
        
        # 各レコードの最大SEの計算
        maxse_cal = maxse_model_C(W, Time_Step_Data_f, Record_Data_f)
        maxrse_cal = np.sqrt(maxse_cal)
        
        # 回帰曲線の最大値
        Cal_Max = np.max(model_C(Time_Step_Data_f, W))
        # 回帰曲線の平均値
        Cal_Avg = np.mean(model_C(Time_Step_Data_f, W))
        
        # 一時異常リストに追加
        if(Save_Class == 'Nor'):
            Normal_Result_Tail_List_Tmp = np.vstack((Normal_Result_Tail_List_Tmp, np.array([Date_Range_DIF, Date_Range_STD, 
                                Abort_Timing, Record_Name, '{:.2e}'.format(rmse_cal), '{:.2e}'.format(float(rmse)), 
                                '{:.2e}'.format(Beam_Max), '{:.3e}'.format(float(maxrse)), '{:.2e}'.format(Record_Max),
                                '{:.2e}'.format(Record_Avg), '{:.2e}'.format(Cal_Max), '{:.2e}'.format(Cal_Avg), 
                                '{:.2e}'.format(Record_B_Max), '{:.2e}'.format(Record_B_Avg),
                                '{:.2e}'.format(Record_STD_B_Max), '{:.2e}'.format(Record_STD_B_Avg),
                                '{:.2e}'.format(float(w0)), '{:.2e}'.format(float(w1)), '{:.2e}'.format(float(w2)), 
                                '{:.2e}'.format(float(w3)), '{:.2e}'.format(float(w4)), '{:.2e}'.format(float(w0_dif)), 
                                '{:.2e}'.format(float(w1_dif)), '{:.2e}'.format(float(w2_dif)), '{:.2e}'.format(float(w3_dif)), 
                                '{:.2e}'.format(float(w4_dif)), '{:.2e}'.format(float(rmse_dif)), 
                                '{:.2e}'.format(float(maxrse_dif)), '{:.2e}'.format(0.0)])))
        if(Save_Class == 'Abn'):
            # subwindowで推定原因を尋ねる 20240218変更　-----------------------------------------------------------------------
            # subwindowのタイトル
            wtitle = Record_Name + ' (Abnormal Tail)'
            
            # Strg_NBのpossible cause選択肢
            self.Cause_List = ['Leak or Pump failure?', 'Overheating or Discharge?', 'Pressure burst?']
            
            # subwindowをつくる
            sub_win = self.getSubWindow(self.Cause_List, wtitle)
            
            # subwindowが閉じられるのを待つ。
            root.wait_window(sub_win) # サブウィンドウが閉じられるのを待つ
            
            # Possible Causeのインデックス
            mpc = self.Cau_var.get()

            Abnormal_Result_Tail_List_Tmp = np.vstack((Abnormal_Result_Tail_List_Tmp, np.array([Date_Range_DIF, Date_Range_STD, 
                                Abort_Timing, Record_Name, '{:.2e}'.format(rmse_cal), '{:.2e}'.format(float(rmse)), 
                                '{:.2e}'.format(Beam_Max), '{:.3e}'.format(float(maxrse)), '{:.2e}'.format(Record_Max),
                                '{:.2e}'.format(Record_Avg), '{:.2e}'.format(Cal_Max), '{:.2e}'.format(Cal_Avg), 
                                '{:.2e}'.format(Record_B_Max), '{:.2e}'.format(Record_B_Avg),
                                '{:.2e}'.format(Record_STD_B_Max), '{:.2e}'.format(Record_STD_B_Avg),
                                '{:.2e}'.format(float(w0)), '{:.2e}'.format(float(w1)), '{:.2e}'.format(float(w2)), 
                                '{:.2e}'.format(float(w3)), '{:.2e}'.format(float(w4)), '{:.2e}'.format(float(w0_dif)), 
                                '{:.2e}'.format(float(w1_dif)), '{:.2e}'.format(float(w2_dif)), '{:.2e}'.format(float(w3_dif)), 
                                '{:.2e}'.format(float(w4_dif)), '{:.2e}'.format(float(rmse_dif)), 
                                '{:.2e}'.format(float(maxrse_dif)), '{:.2e}'.format(float(mpc))])))
            
    # 行数が1行以上あり
    if(Abnormal_Result_Tail_List_Tmp.shape[0] > 0):
        # 同じ時間ではないなら
        # if(Abnormal_Result_Tail_List_Tmp[0][0] not in Abnormal_Result_Tail_List[1:, 0]):
        #     Abnormal_Result_Tail_List = np.vstack((Abnormal_Result_Tail_List, Abnormal_Result_Tail_List_Tmp))
        # else:
        #     print('The same abort time Abnormal Tail data exist.')
        #     tk.messagebox.showinfo('Note', 'No new abnormal Tail result')
    
        # 20240402
        # 同じ時間、同じレコード名でないなら
        nrowtmp = Abnormal_Result_Tail_List.shape[0]
        hit = 0
        for i in range(nrowtmp):
            if(Abnormal_Result_Tail_List_Tmp[0][0] == Abnormal_Result_Tail_List[i, 0]):
                if(Abnormal_Result_Tail_List_Tmp[0][3] == Abnormal_Result_Tail_List[i, 3]):
                    hit = 1

        if(hit == 0):
            Abnormal_Result_Tail_List = np.vstack((Abnormal_Result_Tail_List, Abnormal_Result_Tail_List_Tmp))
        else:
            print('The same abort time and record name exist in Abnormal Tail data.')
            tk.messagebox.showinfo('Attention', 'The same abort time and record name exist in Abnormal Tail data.')
            
        # 異常リストを保存する
        # 保存するリスト
        Array_to_Save = Abnormal_Result_Tail_List
        # arrayで保存するファイル名
        Array_File_Name_to_Save = Abnormal_Result_Tail_File_Name
        # textで保存するファイル名
        Text_File_Name_to_Save = Abnormal_Result_Tail_Text_File_Name
        # 保存する関数
        Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
        
        del Array_to_Save
        
    if(Normal_Result_Tail_List_Tmp.shape[0] > 0):
        # 同じ時間ではないなら
        # if(Normal_Result_Tail_List_Tmp[0][0] not in Normal_Result_Tail_List[1:, 0]):
        #     Normal_Result_Tail_List = np.vstack((Normal_Result_Tail_List, Normal_Result_Tail_List_Tmp))
        # else:
        #     print('The same abort time Normal Tail data exist.')
        #     tk.messagebox.showinfo('Note', 'No new normal Tail result')
        
        # 20240402
        # 同じ時間、同じレコード名でないなら
        nrowtmp = Normal_Result_Tail_List.shape[0]
        hit = 0
        for i in range(nrowtmp):
            if(Normal_Result_Tail_List_Tmp[0][0] == Normal_Result_Tail_List[i, 0]):
                if(Normal_Result_Tail_List_Tmp[0][3] == Normal_Result_Tail_List[i, 3]):
                    hit = 1

        if(hit == 0):
            Normal_Result_Tail_List = np.vstack((Normal_Result_Tail_List, Normal_Result_Tail_List_Tmp))
        else:
            print('The same abort time and record name exist in Normal Tail data.')
            tk.messagebox.showinfo('Attention', 'The same abort time and record name exist in Normal Tail data.')
            
        # 正常リストを保存する
        # 保存するリスト
        Array_to_Save = Normal_Result_Tail_List
        # arrayで保存するファイル名
        Array_File_Name_to_Save = Normal_Result_Tail_File_Name
        # textで保存するファイル名
        Text_File_Name_to_Save = Normal_Result_Tail_Text_File_Name
        # 保存する関数
        Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
        
        del Array_to_Save
        
    # 20240223追加　ローカルリストの削除
    del Abnormal_Result_Tail_List, Normal_Result_Tail_List
    del All_Data, All_Data_B, All_Result, All_Result_DIF_Dict
    del All_Data_STD, All_Data_STD_B, All_Result_STD_Dict, Time_Step_Data, Time_Step_Data_STD
    del Record_Name_bp, Record_Data_p, Record_List, All_Data_s, Beam_Data_s, Beam_Data_s_f
    del Abnormal_Result_Tail_List_Tmp, Normal_Result_Tail_List_Tmp
    del Record_Name_b, Record_Data, Record_Name_B_b, Record_Data_B
    del Record_Data_f, Time_Step_Data_f, Record_Data_B_f, Record_Data_B_f3
    del Record_Name_b_STD, Record_Data_STD, Record_Name_b_STD_B, Record_Data_STD_B
    del Record_Data_STD_f, Time_Step_Data_STD_f, Record_Data_STD_B_f, Record_Data_STD_B_f3
    
    #　メモリーの開放
    gc.collect()
    
    return

# ==============================================================================================================
       
def Record_Freq_plot(self, List_Para, Method, nplot, nfill, nrecord, No_Beam, Cn_Beam):
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    
    # プロット定義
    fig = plt.figure(figsize = (9, 4), tight_layout=True)
    
    #CCGの説明リストボックスのクリア
    if(Ring_Name == 'LER'):
        # 20240108変更
        # self.LER_CCG_Strg_list.delete(0, tk.END)
        # self.LER_CCG_Tail_list.delete(0, tk.END)
        self.LER_CCG_Strg_listbox.delete(0, tk.END)
        self.LER_CCG_Tail_listbox.delete(0, tk.END)
    elif(Ring_Name == 'HER'):
        # 20240108変更
        # self.HER_CCG_Strg_list.delete(0, tk.END)
        # self.HER_CCG_Tail_list.delete(0, tk.END)
        self.HER_CCG_Strg_listbox.delete(0, tk.END)
        self.HER_CCG_Tail_listbox.delete(0, tk.END)
    
    # 最大頻度のレコード名の初期値
    Max_Record_Name_Strg = 'none'
    Max_Record_Name_Tail = 'none'
    
    # 20240123変更
    Cause_Record_Name_Strg = np.zeros((5, 3))# 5x3の２次元配列を生成。
    Cause_Record_Name_Strg = [list(map(str, row)) for row in Cause_Record_Name_Strg]
    Cause_Record_Name_Tail = np.zeros((5, 3))# 5x3の２次元配列を生成。
    Cause_Record_Name_Tail = [list(map(str, row)) for row in Cause_Record_Name_Tail]
            
    if(No_Beam == 0) and (Cn_Beam == 0): # ビームがあってTailもあったら
        m_list = ['Strg', 'Tail']
    else: # ビームがなかったら
        m_list = ['Strg']
        
    # abnormalの個数の最大値
    abmax = 0
    for Mode_Para in m_list:
        # Abnormalのデータ
        if(No_Beam == 0):
            path = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_' + Mode_Para + '_WB.txt'
        else:
            path = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_' + Mode_Para + '_NB.txt'
            
        is_file = os.path.isfile(path)
        if(is_file == True):
            content =[]
            with open (path, encoding = 'utf8', newline = '') as f:
                csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
                for row in csvreader:
                    content = content + row #contentは1行のリストになる
            # もとのデータのリスト'content'をベクトルのデータ'data'にする
            if(Mode_Para == 'Strg'):
                data1 = np.array(content).reshape(-1, 26)
            elif(Mode_Para == 'Tail'):
                data1 = np.array(content).reshape(-1, 29)
        else:
            print('No abnormal data list')
            tk.messagebox.showinfo('No abnormal data list', '     Check save button     ' + Mode_Para)
            abmax = -1
            Cause_Record_Name_Strg = np.zeros((5, 3))# 5x3の２次元配列を生成。
            Cause_Record_Name_Strg = [list(map(str, row)) for row in Cause_Record_Name_Strg]
            Cause_Record_Name_Tail = np.zeros((5, 3))# 5x3の２次元配列を生成。
            Cause_Record_Name_Tail = [list(map(str, row)) for row in Cause_Record_Name_Tail]
            return abmax, Cause_Record_Name_Strg, Cause_Record_Name_Tail

        # Normalのデータ
        if(No_Beam == 0):
            path = Ring_Name + '_' + Method + '_Normal_Class2_Result_' + Mode_Para + '_WB.txt'
        else:
            path = Ring_Name + '_' + Method + '_Normal_Class2_Result_' + Mode_Para + '_NB.txt'
        is_file = os.path.isfile(path)
        if(is_file == True):
            content =[]
            with open (path, encoding = 'utf8', newline = '') as f:
                csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
                for row in csvreader:
                    content = content + row #contentは1行のリストになる
            # もとのデータのリスト'content'をベクトルのデータ'data'にする
            if(Mode_Para == 'Strg'):
                data2 = np.array(content).reshape(-1, 26)
            elif(Mode_Para == 'Tail'):
                data2 = np.array(content).reshape(-1, 29)
        else:
            print('No normal data list')
            tk.messagebox.showinfo('No normal data list', '   Check save button   ' + Mode_Para)
            abmax = -1
            Cause_Record_Name_Strg = np.zeros((5, 3))# 5x3の２次元配列を生成。
            Cause_Record_Name_Strg = [list(map(str, row)) for row in Cause_Record_Name_Strg]
            Cause_Record_Name_Tail = np.zeros((5, 3))# 5x3の２次元配列を生成。
            Cause_Record_Name_Tail = [list(map(str, row)) for row in Cause_Record_Name_Tail]
            return abmax, Cause_Record_Name_Strg, Cause_Record_Name_Tail
    
        # Abnormalデータから、時刻(kblogrd形式)とレコード名のリストをつくる。
        # この時刻はAbort Timingの3分後--> 2分後 #20240207変更
        List_Date1 = data1[:, 0]
        List_Name1 = data1[:, 3]
        List_Beam1 = data1[:, 6]
        Sum_Abnormal_List = np.array([['Date', 'Record Name']])
        Sum_Beam1_List = np.array([['Date', 'Beam Current']])
        if(List_Date1.shape[0] > 1):
            for i in range(1, List_Date1.shape[0]):
                Sum_Abnormal_List = np.vstack((Sum_Abnormal_List, [List_Date1[i][15:], List_Name1[i]]))
                Sum_Beam1_List = np.vstack((Sum_Beam1_List, [List_Date1[i][15:], List_Beam1[i]]))
        else:
            print('No abnormal data')
        
        # 20240107追加
        # Abnormalデータから、List_Date_Range_DIF、List_Date_Range_STD、List_Record_Nameのリストをつくる
        List_Date_Range_DIF = data1[:, 0]
        List_Date_Range_STD = data1[:, 1]
        List_Record_Name = data1[:, 3]
        Sum_Abnormal_Record_List = np.array([['Date_Range_DIF', 'Date_Range_STD', 'Record_Name']])
        if(List_Date1.shape[0] > 1):
            for i in range(1, List_Date1.shape[0]):
                Sum_Abnormal_Record_List = np.vstack((Sum_Abnormal_Record_List, [List_Date_Range_DIF[i], List_Date_Range_STD[i], 
                                                                          List_Record_Name[i]]))
        else:
            print('No abnormal data')
            
        # Normalデータから、時刻(kblogrd形式)とレコード名のリストをつくる。
        # この時刻はAbort Timingの3分後--> 2分後 #20240207変更
        List_Date2 = data2[:, 0]
        List_Name2 = data2[:, 3]
        List_Beam2 = data2[:, 6]
        Sum_Normal_List = np.array([['Date', 'Record Name']])
        Sum_Beam2_List = np.array([['Date', 'Beam Current']])
        if(List_Date2.shape[0] > 1):
            for i in range(1, List_Date2.shape[0]):
                Sum_Normal_List = np.vstack((Sum_Normal_List, [List_Date2[i][15:], List_Name2[i]]))
                Sum_Beam2_List = np.vstack((Sum_Beam2_List, [List_Date2[i][15:], List_Beam2[i]]))
        else:
            print('No normal data')
            
        # AbnormalデータとNormalデータを合体する
        if(Sum_Normal_List.shape[0] > 1) or (Sum_Abnormal_List.shape[0] > 1):
            Sum_Date_List = np.vstack((Sum_Abnormal_List, Sum_Normal_List[1:]))
            Sum_Beam_List = np.vstack((Sum_Beam1_List, Sum_Beam2_List[1:]))
        if(Sum_Date_List.shape[0] == 1):
            print('No abnormal and normal data list')
            tk.messagebox.showinfo('No abnormal and normal data list', 'Check save button')
            abmax = -1
            Cause_Record_Name_Strg = np.zeros((5, 3))# 5x3の２次元配列を生成。
            Cause_Record_Name_Strg = [list(map(str, row)) for row in Cause_Record_Name_Strg]
            Cause_Record_Name_Tail = np.zeros((5, 3))# 5x3の２次元配列を生成。
            Cause_Record_Name_Tail = [list(map(str, row)) for row in Cause_Record_Name_Tail]
            return abmax, Cause_Record_Name_Strg, Cause_Record_Name_Tail

        # ユニークな時刻のリストにしてアボートタイム(3分後--> 2分後 #20240207変更)のリストを作る。:新しいもの順
        Sum_Date_List_Uni = np.unique(Sum_Date_List[1:, 0])
        Abort_Time_List = np.array(sorted(Sum_Date_List_Uni, reverse = True))
        # print(7773, Abort_Time_List[:5])
        
        # アボートタイミング(本当)のリスト
        Abort_Time_List_dt = []
        # アボートタイミングの3分前--> 2分前 #20240207変更
        minit_advance = -2
        for item in Abort_Time_List:
            date_dt = Convert_Kblogrd_to_Dtime(item)
            dtime = date_dt + datetime.timedelta(minutes = minit_advance)
            Abort_Time_List_dt.append(str(dtime))
        
        # Strgの時
        # アボートタイミング(本当)のリストを保存する
        if(Mode_Para == 'Strg'):
            Array_to_Save = Abort_Time_List_dt
            # arrayで保存するファイル名
            Array_File_Name_to_Save = Ring_Name + '_Abort_Time_List.npy'
            # textで保存するファイル名
            Text_File_Name_to_Save = Ring_Name + '_Abort_Time_List.txt'
            # 保存する関数
            Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
            
            del Array_to_Save
        
        # ------------------------------------------------------------------------------------------------------------------
        # 最近のnplotフィル前から(nplotアボート)時、あるいは、
        # nplotフィルなければnplotフィル前からnfillフィルのnrecordレコード名リストを作る(重複あり)
        # nplot = 12
        # nfill = 8
        # nrecord = 5

        # 初期設定
        Record_Freq_List_Sum = np.full((nrecord + 1, nplot * 2), '0')
        nlist = len(Abort_Time_List)
        Beam_List =[]
        
        for n in range(nplot): # プロットする数
            x0 = np.min([n, nlist])
            x1 = np.min([n + nfill - 1, nlist - 1])
    
            # 頻度を計算する最初のフィルの時刻(アボート後3分)と最後のフィルの時刻
            if(x0 <= x1):
                First_Abort_Time = Abort_Time_List[x1]
                Last_Abort_Time = Abort_Time_List[x0]
            else:
                First_Abort_Time = 'none'
    
            if(First_Abort_Time != 'none'):
                Sum_Abnormal_List_Sel = np.array([['Date', 'Record name']])
                if(Sum_Abnormal_List.shape[0] > 1):
                    for i in range(1, len(Sum_Abnormal_List)):
                        if(Sum_Abnormal_List[i, 0] >= First_Abort_Time) and (Sum_Abnormal_List[i, 0] <= Last_Abort_Time):
                            Sum_Abnormal_List_Sel = np.vstack((Sum_Abnormal_List_Sel, Sum_Abnormal_List[i]))
                
                for i in range(Sum_Beam_List.shape[0]):
                    if(Last_Abort_Time == Sum_Beam_List[i, 0]):
                        Beam_List = np.append(Beam_List, Sum_Beam_List[i, 1])
                        break
                
                if(Sum_Abnormal_List_Sel.shape[0] > 1):
                    # ユニークなレコード名とその出現頻度
                    u, counts = np.unique(Sum_Abnormal_List_Sel[1:, 1], return_counts=True)

                    Record_Freq_List = np.array(['Counts', 'Record name'])
                    for i in range(len(u)):
                        Record_Freq_List = np.vstack((Record_Freq_List, [counts[i], u[i]]))
                    Record_Freq_List_1 = np.delete(Record_Freq_List, 0, 0).tolist()
                    Record_Freq_List_Sort = np.array(sorted(Record_Freq_List_1, reverse = True)).reshape(-1, 2)
                    nrow = Record_Freq_List_Sort.shape[0]
        
                    if(nrow >= nrecord):
                        Record_Freq_List = np.vstack((Record_Freq_List[0], Record_Freq_List_Sort[:nrecord]))
                    else:
                        Record_Freq_List = np.vstack((Record_Freq_List[0], Record_Freq_List_Sort[:nrow]))
                        for j in range(nrow, nrecord):
                            Record_Freq_List = np.vstack((Record_Freq_List, ['0', '0']))
                else:
                    Record_Freq_List = np.array(['Counts', 'Record name'])
                    for j in range(nrecord):
                        Record_Freq_List = np.vstack((Record_Freq_List, ['0', '0']))
            else:
                Record_Freq_List = np.array(['Counts', 'Record name'])
                for j in range(nrecord):
                    Record_Freq_List = np.vstack((Record_Freq_List, ['0', '0']))
    
            # 列を後ろからつぎ足す。最初の2列を削除する。
            Record_Freq_List_Sum = np.hstack((Record_Freq_List_Sum, Record_Freq_List))
            Record_Freq_List_Sum = np.delete(Record_Freq_List_Sum, slice(0, 2), axis = 1)
        
        # Beam Listの長さ
        while(len(Beam_List) <= nplot):
            Beam_List = np.append(Beam_List, ['0'])

        # ------------------------------------------------------------------------------------------------------------------
        # プロットする
        # nfillのレコードから頻度の高いものから並べる
        Record_List = []

        for i in range(1, 1 * 2, 2): # 最近の1xnfillフィルで頻度の高いレコードを用いるため
            for j in range(1, nrecord + 1):
                if(Record_Freq_List_Sum[j, i] != '0'):
                    k = int(Record_Freq_List_Sum[j, i - 1])
                    for m in range(k + 1):
                        Record_List = np.append(Record_List, Record_Freq_List_Sum[j, i])

        u, count = np.unique(Record_List, return_counts=True)
        Record_List_Uni = u[np.argsort(count)]
        Record_List_Sort = np.flip(Record_List_Uni) # 頻度の高いもの順のリスト

        nrecord_a = len(Record_List_Sort)
        if(nrecord_a == 0):
            print(203, 'no record to plot', Ring_Name, Mode_Para)
            
        rname = [] # プロットするレコード名リスト
        x = np.full(nplot, 0.)
        y = np.full(nplot, 0.)
        y2 = np.full(nplot, 0.)
        cl = ['red', 'orange', 'green', 'deepskyblue', 'blue']
        if Mode_Para == 'Strg':
            ax = fig.add_subplot(1, 2, 1)
        elif Mode_Para == 'Tail':
            ax = fig.add_subplot(1, 2, 2)
        plt.subplots_adjust(hspace = 0.3)
        Abort_Time_dtime = Convert_Kblogrd_to_Dtime(Abort_Time_List[0])
        # 20240301修正 -3 -> -2
        minit_advance = -2
        Abort_Time_dtime = Abort_Time_dtime + datetime.timedelta(minutes = minit_advance)
        ax.set_title(Mode_Para + ': Abort Time = '+ str(Abort_Time_dtime), fontsize = 10)
        ax.set_xlabel('Latest check', size = 14, weight = 'light')
        ax.set_ylabel('Counts of anomalies tallied during' + '\n' + 'the last eight checks', size = 14, weight = 'light')
        ax.set_xlim(-nplot + 1, 1)
        ax.set_ylim(0, nfill + 2)
        ax.tick_params(direction = 'inout', length = 5, colors = 'black')
        if(Ring_Name == 'HER'):
            if(Mode_Para == 'Strg'):
                ax.set_facecolor((0.96, 0.96, 1.0))
                ax.set_alpha(0.8)
            else:
                ax.set_facecolor((0.96, 1.0, 0.96))
                ax.set_alpha(0.8)
        elif(Ring_Name == 'LER'):
            if(Mode_Para == 'Strg'):
                ax.set_facecolor((1.0, 0.96, 0.96))
                ax.set_alpha(0.8)
            else:
                ax.set_facecolor((1.0, 1.0, 0.96))
                ax.set_alpha(0.8)
        
        if(nrecord_a == 0):
            for j in range(nplot):
                x[j] = -j
                y[j] = 0
            ax.plot(x, y, color = cl[0], zorder = 10, label = 'no record')
            ax.scatter(x, y, s = 10, zorder = 20, color = cl[0])
            ax.legend(loc = 'upper left', framealpha = 0.3, fontsize = 8)
        
        else:
            for i in range(np.min((nrecord_a, nrecord))):
                rname.append(Record_List_Sort[i])

                for j in range(nplot):
                    x[j] = -j
                    result = rname[i] in Record_Freq_List_Sum[:, 1 + j * 2]

                    if(result == True):
                        list1 = Record_Freq_List_Sum[:, 1 + j * 2].tolist()
                        k = list1.index(rname[i])
                        y[j] = float(Record_Freq_List_Sum[k, j * 2]) + np.maximum(0, 0.25 - 0.05 * i)
                    else:
                        y[j] = np.maximum(0, 0.25 - 0.05 * i)
                if(abmax < y[0]):
                    abmax = y[0]
                ax.plot(x, y, color = cl[i], zorder = 10 - i, label = rname[i])
                ax.scatter(x, y, s = 10, zorder = 20 - i, color = cl[i])
                ax.legend(loc = 'upper left', framealpha = 0.3, fontsize = 9)
            
                #CCGの説明リストボックス追記
                if(Ring_Name == 'LER') and (Mode_Para == 'Strg'):
                    for k in range(0, 12):
                        if(rname[i] in self.LER_Record_Name_List_box[k]):
                            indx = self.LER_Record_Name_List_box[k].index(rname[i])
                            rname_place = self.LER_Record_Name_Place_List_box[k][indx]
                            # 20240108変更
                            # self.LER_CCG_Strg_list.insert(tk.END, rname[i][7:] + ' : ' + rname_place)
                            # self.LER_CCG_Strg_list.itemconfigure(tk.END, foreground = cl[i])
                            # self.LER_CCG_Strg_list.see(tk.END)
                            self.LER_CCG_Strg_listbox.insert(tk.END, rname[i][7:] + ' : ' + rname_place)
                            self.LER_CCG_Strg_listbox.itemconfigure(tk.END, foreground = cl[i])
                            self.LER_CCG_Strg_listbox.see(tk.END)
                            break
                elif(Ring_Name == 'LER') and (Mode_Para == 'Tail'):
                    for k in range(0, 12):
                        if(rname[i] in self.LER_Record_Name_List_box[k]):
                            indx = self.LER_Record_Name_List_box[k].index(rname[i])
                            rname_place = self.LER_Record_Name_Place_List_box[k][indx]
                            # 20240108変更
                            # self.LER_CCG_Tail_list.insert(tk.END, rname[i][7:] + ' : ' + rname_place)
                            # self.LER_CCG_Tail_list.itemconfigure(tk.END, foreground = cl[i])
                            # self.LER_CCG_Tail_list.see(tk.END)
                            self.LER_CCG_Tail_listbox.insert(tk.END, rname[i][7:] + ' : ' + rname_place)
                            self.LER_CCG_Tail_listbox.itemconfigure(tk.END, foreground = cl[i])
                            self.LER_CCG_Tail_listbox.see(tk.END)
                            break
                elif(Ring_Name == 'HER') and (Mode_Para == 'Strg'):
                    for k in range(0, 12):
                        if(rname[i] in self.HER_Record_Name_List_box[k]):
                            indx = self.HER_Record_Name_List_box[k].index(rname[i])
                            rname_place = self.HER_Record_Name_Place_List_box[k][indx]
                            # 20240108変更
                            # self.HER_CCG_Strg_list.insert(tk.END, rname[i][7:] + ' : ' + rname_place)
                            # self.HER_CCG_Strg_list.itemconfigure(tk.END, foreground = cl[i])
                            # self.HER_CCG_Strg_list.see(tk.END)
                            self.HER_CCG_Strg_listbox.insert(tk.END, rname[i][7:] + ' : ' + rname_place)
                            self.HER_CCG_Strg_listbox.itemconfigure(tk.END, foreground = cl[i])
                            self.HER_CCG_Strg_listbox.see(tk.END)
                            break
                elif(Ring_Name == 'HER') and (Mode_Para == 'Tail'):
                    for k in range(0, 12):
                        if(rname[i] in self.HER_Record_Name_List_box[k]):
                            indx = self.HER_Record_Name_List_box[k].index(rname[i])
                            rname_place = self.HER_Record_Name_Place_List_box[k][indx]
                            # 20240108変更
                            # self.HER_CCG_Tail_list.insert(tk.END, rname[i][7:] + ' : ' + rname_place)
                            # self.HER_CCG_Tail_list.itemconfigure(tk.END, foreground = cl[i])
                            # self.HER_CCG_Tail_list.see(tk.END)
                            self.HER_CCG_Tail_listbox.insert(tk.END, rname[i][7:] + ' : ' + rname_place)
                            self.HER_CCG_Tail_listbox.itemconfigure(tk.END, foreground = cl[i])
                            self.HER_CCG_Tail_listbox.see(tk.END)
                            break
       
        # アボート時刻をテキストとして記入する
        xl = np.full(nplot, 0.)
        yl = np.full(nplot, 0.)
        for j in range(0, nplot, 2):
            if(j <= len(Abort_Time_List_dt) - 1):
                xl[j] = -j - 0.8
                yl[j] = 0.2
                stext = Abort_Time_List_dt[j][11:]
                ax.text(xl[j], yl[j], stext, fontsize = 7)
                
        # ビーム電流を2軸にプロットする。
        ax2 = ax.twinx()
        for j in range(nplot):
            if(No_Beam == 0):
                y2[j] = float(Beam_List[j])
            else:
                y2[j] = 0.
        # ax2.set_ylim(0, np.max(y2) * 1.2 + 1)
        ax2.set_ylim(0, np.maximum(np.max(y2) * 1.2 + 1, 50.))
        ax2.plot(x, y2, color = 'black', linestyle = "dotted", label = 'Beam' )
        ax2.legend(loc = 'upper right', framealpha = 0.3, fontsize = 8)
        ax2.set_ylabel('Beam current', size = 14, weight = 'light')
        
        # 20240107追記
        # Abnormal record names
        if (nrecord_a > 0) and (Mode_Para == 'Strg'):
            for i in range(np.min((nrecord_a, nrecord))):
                for k in reversed(range(Sum_Abnormal_Record_List.shape[0])):
                    if(rname[i] == Sum_Abnormal_Record_List[k, 2]):
                        Cause_Record_Name_Strg[i][:] = Sum_Abnormal_Record_List[k][:]
                        break
            Cause_Record_Name_Strg = np.array(Cause_Record_Name_Strg)
            
        elif (nrecord_a > 0) and (Mode_Para == 'Tail'):
            for i in range(np.min((nrecord_a, nrecord))):
                for k in reversed(range(Sum_Abnormal_Record_List.shape[0])):
                    if(rname[i] == Sum_Abnormal_Record_List[k, 2]):
                        Cause_Record_Name_Tail[i][:] = Sum_Abnormal_Record_List[k][:]
                        break 
            Cause_Record_Name_Tail = np.array(Cause_Record_Name_Tail)
    
    # 20240223追加　ローカルリストの削除
    del content, data1, List_Date1, List_Name1, List_Beam1, Sum_Abnormal_List, Sum_Beam1_List
    del List_Date_Range_DIF, List_Date_Range_STD, List_Record_Name, Sum_Abnormal_Record_List
    del List_Date2, List_Name2, List_Beam2, Sum_Normal_List, Sum_Beam2_List, Sum_Date_List, 
    del Sum_Beam_List, Sum_Date_List_Uni, Abort_Time_List, Abort_Time_List_dt
    del Record_Freq_List_Sum, Beam_List, u
    del Record_List_Uni, Record_List_Sort
    del x, y, y2, rname
    
    
    # メモリーの開放
    gc.collect()
    
    return abmax, Cause_Record_Name_Strg, Cause_Record_Name_Tail

# ==============================================================================================================
       
# Abnormalの原因を推定する
def Find_Possible_Cause(self, List_Para, Method, Cause_Record_Name_Strg, Cause_Record_Name_Tail, No_Beam, Cn_Beam):
    # リングの名前
    Ring_Name = List_Para[0][:3] # LERかHER
    
    # 初期値
    Text_Cause = ''
    Max_Record_Name = ''
    
    # Kerasの結果をロードする
    # path = 'model_pc_strg_wb.keras'
    path = 'model_pc_strg_wb.h5'
    is_file = os.path.isfile(path)
    if(is_file == False):
        tk.messagebox.showinfo('File Error', path + 'does not exist')
        return
    # model_strg_wb = keras.models.load_model('model_pc_strg_wb.keras')
    model_pc_strg_wb = keras.models.load_model('model_pc_strg_wb.h5')
    
    # 20240227追加
    # meanとstdをロードする
    # データのファイルがあるか？
    path = 'sms_pc_strg_wb.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        tk.messagebox.showinfo('File Error', path + 'does not exist')
        return
    # データのファイルからmeanとstdを読み込む 
    content =[]
    with open (path, encoding = 'utf8', newline = '') as f:
        csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
        for row in csvreader:
            content = content + row #contentは1行のリストになる
    sms_pc_strg_wb = content
    
    # path = 'model_pc_strg_nb.keras'
    path = 'model_pc_strg_nb.h5'
    is_file = os.path.isfile(path)
    if(is_file == False):
        tk.messagebox.showinfo('File Error', path + 'does not exist')
        return
    # model_strg_nb = keras.models.load_model('model_pc_strg_nb.keras')
    model_pc_strg_nb = keras.models.load_model('model_pc_strg_nb.h5')
    
    # 20240227追加
    # meanとstdをロードする
    # データのファイルがあるか？
    path = 'sms_pc_strg_nb.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        tk.messagebox.showinfo('File Error', path + 'does not exist')
        return
    # データのファイルからmeanとstdを読み込む 
    content =[]
    with open (path, encoding = 'utf8', newline = '') as f:
        csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
        for row in csvreader:
            content = content + row #contentは1行のリストになる
    sms_pc_strg_nb = content
    
    # path = 'model_pc_tail_wb.keras'
    path = 'model_pc_tail_wb.h5'
    is_file = os.path.isfile(path)
    if(is_file == False):
        tk.messagebox.showinfo('File Error', path + 'does not exist')
        return
    # model_tail_wb = keras.models.load_model('model_pc_tail_wb.keras')
    model_pc_tail_wb = keras.models.load_model('model_pc_tail_wb.h5')
    
    # 20240227追加
    # meanとstdをロードする
    # データのファイルがあるか？
    path = 'sms_pc_tail_wb.txt'
    is_file = os.path.isfile(path)
    if(is_file == False):
        tk.messagebox.showinfo('File Error', path + 'does not exist')
        return
    # データのファイルからmeanとstdを読み込む 
    content =[]
    with open (path, encoding = 'utf8', newline = '') as f:
        csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
        for row in csvreader:
            content = content + row #contentは1行のリストになる
    sms_pc_tail_wb = content
    
    for k in range(5):
        # Strgのレコードがある場合
        # print(7585, k, Cause_Record_Name_Strg[k][0])
        if(Cause_Record_Name_Strg[k][0] != '0.0'):

            Date_Range_DIF = Cause_Record_Name_Strg[k][0]
            Date_Range_STD = Cause_Record_Name_Strg[k][1]
            Max_Record_Name = Cause_Record_Name_Strg[k][2]
        
            # ビームがあったら (No_Beam = 0)
            if(No_Beam == 0):
                # Abnormalのリストを読み込む
                Abnormal_Result_Strg_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_WB.npy'
                Abnormal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_WB.txt'
                Abnormal_Result_Strg_List = Check_Strg_Abnormal_Result_File(Abnormal_Result_Strg_File_Name, 
                                                               Abnormal_Result_Strg_Text_File_Name)
                
                # Max_Record_Nameの結果を読み出す(STD_Strg)
                # Abnormal_Result_Strg_List = np.array([['Date_Range_DIF', 'Date_Range_STD', 'Abort_Timing', 'Record Name', 
                #                                    'RMSE_cal', 'RMSE_std', 'Beam_Max', 'MaxRSE_std', 'Max_Pre', 'Max_Sel_Pre', 
                #                                    'Avg_Sel_Pre', 'Max_Cal', 'Avg_Cal', 'RMSE_STEP_Max', 'STEP_Max_Avg', 
                #                                    'RMSE_STEP_Mean', 'STEP_Mean_Avg', 'W0_std', 'W1_std', 'W2_std', 'W0_dif', 
                #                                    'W1_dif', 'W2_dif', 'RMSE_dif', 'MaxRMSE_dif', 'Cause_no']])
                
                #逆側から探す
                hit = 0
                # nhit = 0
                for i in reversed(range(Abnormal_Result_Strg_List.shape[0])):
                    if (Max_Record_Name == Abnormal_Result_Strg_List[i][3]): # Record_Name = 3　
                        W0_std = Abnormal_Result_Strg_List[i][17]
                        W1_std = Abnormal_Result_Strg_List[i][18]
                        W2_std = Abnormal_Result_Strg_List[i][19]
                        W0_dif = Abnormal_Result_Strg_List[i][20]
                        W1_dif = Abnormal_Result_Strg_List[i][21]
                        W2_dif = Abnormal_Result_Strg_List[i][22]
                        Max_Pre = Abnormal_Result_Strg_List[i][8]
                        Max_Cal = Abnormal_Result_Strg_List[i][11]
                        hit = 1
                    # nhit = nhit +1
                    if (hit ==1):
                        break
                if (hit == 0): # 該当するレコードが無かったら
                    Text_Cause = 'no old result file'
                else:
                    W_STD =[float(W0_std), float(W1_std), float(W2_std)]
                    
                    # 入力パラメータは8個)
                    X1 = funclog1(float(W0_std))
                    X2 = funclog1(float(W1_std))
                    X3 = funclog1(float(W2_std))
                    X4 = funclog1(float(W0_dif))
                    X5 = funclog1(float(W1_dif))
                    X6 = funclog1(float(W2_dif))
                    X7 = funclog1(float(Max_Pre))
                    X8 = funclog1(float(Max_Cal))

                    # 列を加える(入力パラメータは8個)
                    X = np.hstack((X1, X2, X3, X4, X5, X6, X7, X8))
                    X = X[None,:]
                    
                    # 20240227追加
                    # 標準化(Training dataのmeanとstdを使う)
                    for i in range(8):
                        me1 = float(sms_pc_strg_wb[i * 2])
                        si1 = float(sms_pc_strg_wb[1 + i * 2])
                        X[:, i] = standard_n1(X[:, i], me1, si1)
                    
                    # 推定する
                    Predictions = model_pc_strg_wb.predict_on_batch(X)
                    # 最有力なもの
                    Cmax = Predictions[0].argmax()
                    # 候補
                    Cause_List = ['Leak or Pump failure', 'Over heating or Discharge', 'Abnormal orbit or Leak']
                
                    Text_Cause = Max_Record_Name[7:] + ' ' + Cause_List[Cmax]
                    
                    # 20240226追加
                    keras.backend.clear_session()
                    del X, Predictions
                    gc.collect()

            # ビームが無かったら (No_Beam = 1)
            if(No_Beam == 1):
                # Abnormalのリストを読み込む
                Abnormal_Result_Strg_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_NB.npy'
                Abnormal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_NB.txt'
                Abnormal_Result_Strg_List = Check_Strg_Abnormal_Result_File(Abnormal_Result_Strg_File_Name, 
                                                               Abnormal_Result_Strg_Text_File_Name)
                
                # Max_Record_Nameの結果を読み出す(STD_Strg)
                # Abnormal_Result_Strg_List = np.array([['Date_Range_DIF', 'Date_Range_STD', 'Abort_Timing', 'Record Name', 
                #                                    'RMSE_cal', 'RMSE_std', 'Beam_Max', 'MaxRSE_std', 'Max_Pre', 'Max_Sel_Pre', 
                #                                    'Avg_Sel_Pre', 'Max_Cal', 'Avg_Cal', 'RMSE_STEP_Max', 'STEP_Max_Avg', 
                #                                    'RMSE_STEP_Mean', 'STEP_Mean_Avg', 'W0_std', 'W1_std', 'W2_std', 'W0_dif', 
                #                                    'W1_dif', 'W2_dif', 'RMSE_dif', 'MaxRMSE_dif', 'Cause_no']])
                
                #逆側から探す
                hit = 0
                # nhit = 0
                for i in reversed(range(Abnormal_Result_Strg_List.shape[0])):
                    if (Max_Record_Name == Abnormal_Result_Strg_List[i][3]): # Record_Name = 3　
                        W0_std = Abnormal_Result_Strg_List[i][17]
                        W1_std = Abnormal_Result_Strg_List[i][18]
                        W2_std = Abnormal_Result_Strg_List[i][19]
                        W0_dif = Abnormal_Result_Strg_List[i][20]
                        W1_dif = Abnormal_Result_Strg_List[i][21]
                        W2_dif = Abnormal_Result_Strg_List[i][22]
                        Max_Pre = Abnormal_Result_Strg_List[i][8]
                        # Avg_Pre = Abnormal_Result_Strg_List[i][10]
                        Max_Cal = Abnormal_Result_Strg_List[i][11]
                        RMSE_dif = Abnormal_Result_Strg_List[i][23]
                        hit = 1
                    # nhit = nhit +1
                    if (hit ==1):
                        break
                if (hit == 0): # 該当するレコードが無かったら
                    Text_Cause = 'no old result file'
                else:
                    W_STD =[float(W0_std), float(W1_std), float(W2_std)]
                    
                    # 入力パラメータは9個)
                    X1 = funclog1(float(W0_std))
                    X2 = funclog1(float(W1_std))
                    X3 = funclog1(float(W2_std))
                    X4 = funclog1(float(W0_dif))
                    X5 = funclog1(float(W1_dif))
                    X6 = funclog1(float(W2_dif))
                    X7 = funclog1(float(Max_Pre))
                    # X8 = funclog1(float(Avg_Pre))
                    X8 = funclog1(float(Max_Cal))
                    X9 = funclog1(float(RMSE_dif))

                    # 列を加える(入力パラメータは9個)
                    X = np.hstack((X1, X2, X3, X4, X5, X6, X7, X8, X9))
                    X = X[None,:]
                    
                    # 20240227追加
                    # 標準化(Training dataのmeanとstdを使う)
                    for i in range(9):
                        me1 = float(sms_pc_strg_nb[i * 2])
                        si1 = float(sms_pc_strg_nb[1 + i * 2])
                        X[:, i] = standard_n1(X[:, i], me1, si1)
                    
                    # 推定する
                    Predictions = model_pc_strg_nb.predict_on_batch(X)
                    # 最有力なもの
                    Cmax = Predictions[0].argmax()
                    # 候補(3クラス)
                    # Cause_List = ['Leak or Pump failure', 'CCG abnormal or Leak', 'Pumping down or Leak']
                    # 候補(2クラス)
                    Cause_List = ['Leak or Pump (CCG) failure', 'Pumping down or Leak']
                
                    Text_Cause = Max_Record_Name[7:] + ' ' + Cause_List[Cmax]
                    
                    # 20240226追加
                    keras.backend.clear_session()
                    del X, Predictions
                    gc.collect()
            
            # 20240224追加　ローカルリストの削除
            del Abnormal_Result_Strg_List
        
        # レコードが無い場合
        else:
            Text_Cause = 'none'
            Max_Record_Name = 'none'

        # 推定原因リストの作成(Strg) 
        if(Ring_Name == 'LER'):
            self.LER_ABN_Cause_Strg_List[k][0] = Text_Cause
            self.LER_ABN_Cause_Strg_List[k][1] = Max_Record_Name
        if(Ring_Name == 'HER'):
            self.HER_ABN_Cause_Strg_List[k][0] = Text_Cause
            self.HER_ABN_Cause_Strg_List[k][1] = Max_Record_Name
            
    # 推定原因のラベルの初期値   
    if(Ring_Name == 'LER'):
        self.LER_ABN_Cause_Strg_label["text"] = self.LER_ABN_Cause_Strg_List[0][0]
        if(self.LER_ABN_Cause_Strg_List[0][0] != 'none'):
            self.LER_ABN_Cause_Strg_label["bg"] = 'white'
            self.LER_ABN_Cause_Strg_label["fg"] = 'red'
        else:
            self.LER_ABN_Cause_Strg_label["bg"] = 'lightgrey'
            self.LER_ABN_Cause_Strg_label["fg"] = 'black'
    if(Ring_Name == 'HER'):
        self.HER_ABN_Cause_Strg_label["text"] = self.HER_ABN_Cause_Strg_List[0][0]
        if(self.HER_ABN_Cause_Strg_List[0][0] != 'none'):
            self.HER_ABN_Cause_Strg_label["bg"] = 'white'
            self.HER_ABN_Cause_Strg_label["fg"] = 'red'
        else:
            self.HER_ABN_Cause_Strg_label["bg"] = 'lightgrey'
            self.HER_ABN_Cause_Strg_label["fg"] = 'black'
            
    # Tailのレコードがある場合
    Text_Cause = ''
    Max_Record_Name = ''
    
    for k in range(5):
        # Tailのレコードがある場合
        if(Cause_Record_Name_Tail[k][0] != '0.0'):
            Date_Range_DIF = Cause_Record_Name_Tail[k][0]
            Date_Range_STD = Cause_Record_Name_Tail[k][1]
            Max_Record_Name = Cause_Record_Name_Tail[k][2]
        
            # Abnormalのリストを読み込む
            Abnormal_Result_Tail_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Tail_WB.npy'
            Abnormal_Result_Tail_Text_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Tail_WB.txt'
            Abnormal_Result_Tail_List = Check_Tail_Abnormal_Result_File(Abnormal_Result_Tail_File_Name, 
                                                            Abnormal_Result_Tail_Text_File_Name)
                
            # Abnormal_Result_Tail_List = np.array([['Date_Range_DIF', 'Date_Range_STD', 'Abort_Timing', 'Record Name', 'RMSE_cal', 
            #                                 'RMSE_std','Beam_Max', 'MaxRSE_std', 'Max_Pre', 'Avg_Pre', 'Max_Cal', 
            #                                 'Avg_Cal', 'Max_Raw_Pre', 'Avg_Raw_Pre', 'Max_STD_Raw_Pre', 'Avg_STD_Raw_Pre',
            #                                 'W0_std', 'W1_std', 'W2_std', 'W3_std', 'W4_std', 'W0_dif', 'W1_dif', 'W2_dif',
            #                                 'W3_dif', 'W4_dif', 'RMSE_dif', 'MaxRSE_dif', 'Cause_no']])
                
            #逆側から探す
            hit = 0
            # nhit = 0
            for i in reversed(range(Abnormal_Result_Tail_List.shape[0])):
                if (Max_Record_Name == Abnormal_Result_Tail_List[i][3]): # Record_Name = 3　
                    W0_std = Abnormal_Result_Tail_List[i][16]
                    W1_std = Abnormal_Result_Tail_List[i][17]
                    W2_std = Abnormal_Result_Tail_List[i][18]
                    W3_std = Abnormal_Result_Tail_List[i][19]
                    W0_dif = Abnormal_Result_Tail_List[i][21]
                    W1_dif = Abnormal_Result_Tail_List[i][22]
                    W2_dif = Abnormal_Result_Tail_List[i][23]
                    W3_dif = Abnormal_Result_Tail_List[i][24]
                    Max_Pre = Abnormal_Result_Tail_List[i][8]
                    Max_Cal = Abnormal_Result_Tail_List[i][10]
                    hit = 1
                # nhit = nhit +1
                if (hit ==1):
                    break
            if (hit == 0): # 該当するレコードが無かったら
                Text_Cause = 'no old result file'
            else:
                W_STD =[float(W0_std), float(W1_std), float(W2_std)]
                    
                # 入力パラメータは10個)
                X1 = funclog1(float(W0_std))
                X2 = funclog1(float(W1_std))
                X3 = funclog1(float(W2_std))
                X4 = funclog1(float(W3_std))
                X5 = funclog1(float(W0_dif))
                X6 = funclog1(float(W1_dif))
                X7 = funclog1(float(W2_dif))
                X8 = funclog1(float(W3_dif))
                X9 = funclog1(float(Max_Pre))
                X10 = funclog1(float(Max_Cal))

                # 列を加える(入力パラメータは10個)
                X = np.hstack((X1, X2, X3, X4, X5, X6, X7, X8, X9, X10))
                X = X[None,:]
                    
                # 20240227追加
                # 標準化(Training dataのmeanとstdを使う)
                for i in range(10):
                    me1 = float(sms_pc_tail_wb[i * 2])
                    si1 = float(sms_pc_tail_wb[1 + i * 2])
                    X[:, i] = standard_n1(X[:, i], me1, si1)
                
                # 推定する
                Predictions = model_pc_tail_wb.predict_on_batch(X)
                # 最有力なもの
                Cmax = Predictions[0].argmax()
                # 候補
                Cause_List = ['Leak or Pump failure', 'Overheating or Discharge', 'Pressure burst']
                
                Text_Cause = Max_Record_Name[7:] + ' ' + Cause_List[Cmax]
                
                # 20240226追加
                keras.backend.clear_session()
                del X, Predictions
                gc.collect()
            
            # 20240224追加　ローカルリストの削除
            del Abnormal_Result_Tail_List
            
        else:
            Text_Cause = 'none'
            Max_Record_Name = 'none'
    
        # 推定原因リストの作成(Tail)     
        if(Ring_Name == 'LER'):
            self.LER_ABN_Cause_Tail_List[k][0] = Text_Cause
            self.LER_ABN_Cause_Tail_List[k][1] = Max_Record_Name

        if(Ring_Name == 'HER'):
            self.HER_ABN_Cause_Tail_List[k][0] = Text_Cause
            self.HER_ABN_Cause_Tail_List[k][1] = Max_Record_Name
    
    # 推定原因のラベルの初期値   
    if(Ring_Name == 'LER'):
        self.LER_ABN_Cause_Tail_label["text"] = self.LER_ABN_Cause_Tail_List[0][0]
        if(self.LER_ABN_Cause_Tail_List[0][0] != 'none'):
            self.LER_ABN_Cause_Tail_label["bg"] = 'white'
            self.LER_ABN_Cause_Tail_label["fg"] = 'red'
        else:
            self.LER_ABN_Cause_Tail_label["bg"] = 'lightgrey'
            self.LER_ABN_Cause_Tail_label["fg"] = 'black'
    if(Ring_Name == 'HER'):
        self.HER_ABN_Cause_Tail_label["text"] = self.HER_ABN_Cause_Tail_List[0][0]
        if(self.HER_ABN_Cause_Tail_List[0][0] != 'none'):
            self.HER_ABN_Cause_Tail_label["bg"] = 'white'
            self.HER_ABN_Cause_Tail_label["fg"] = 'red'
        else:
            self.HER_ABN_Cause_Tail_label["bg"] = 'lightgrey'
            self.HER_ABN_Cause_Tail_label["fg"] = 'black'
    
    # 20240224追加　ローカル変数の削除
    del model_pc_strg_wb, model_pc_strg_nb, model_pc_tail_wb
    
    #　メモリーの開放
    gc.collect()
    
    return

# ==============================================================================================================
# ==============================================================================================================
        
def Make_LER_Record(LER_List):
    LER_List[0] = ['none', 'VALCCG:D01_L01A:PRES', 'VALCCG:D01_L02:PRES', 'VALCCG:D01_L03:PRES', 'VALCCG:D01_L04:PRES', 
                   'VALCCG:D01_L05:PRES', 'VALCCG:D01_L06:PRES', 'VALCCG:D01_L07:PRES', 'VALCCG:D01_L08:PRES', 
                   'VALCCG:D01_L09:PRES', 'VALCCG:D01_L10:PRES', 'VALCCG:D01_L11:PRES', 'VALCCG:D01_L12:PRES', 
                   'VALCCG:D01_L13:PRES', 'VALCCG:D01_L14:PRES', 'VALCCG:D01_L15:PRES', 'VALCCG:D01_L16:PRES', 
                   'VALCCG:D01_L17:PRES', 'VALCCG:D01_L18:PRES', 'VALCCG:D01_L19:PRES', 'VALCCG:D01_L20:PRES', 
                   'VALCCG:D01_L21:PRES', 'VALCCG:D01_L22:PRES', 'VALCCG:D01_L23:PRES']
    LER_List[1] = ['none', 'VALCCG:D02_L01:PRES', 'VALCCG:D02_L02:PRES', 'VALCCG:D02_L03:PRES', 'VALCCG:D02_L04:PRES', 
                   'VALCCG:D02_L05:PRES', 'VALCCG:D02_L06:PRES', 'VALCCG:D02_L07:PRES', 'VALCCG:D02_L08:PRES', 
                   'VALCCG:D02_L09:PRES', 'VALCCG:D02_L10:PRES', 'VALCCG:D02_L11:PRES', 'VALCCG:D02_L12:PRES', 
                   'VALCCG:D02_L13:PRES', 'VALCCG:D02_L14:PRES', 'VALCCG:D02_L15:PRES', 'VALCCG:D02_L16:PRES', 
                   'VALCCG:D02_L17:PRES', 'VALCCG:D02_L18:PRES', 'VALCCG:D02_L19:PRES', 'VALCCG:D02_L20:PRES', 
                   'VALCCG:D02_L21:PRES', 'VALCCG:D02_L22:PRES', 'VALCCG:D02_L23:PRES', 'VALCCG:D02_L24:PRES',
                   'VALCCG:D02_L25:PRES']
    LER_List[2] = ['none', 'VALCCG:D03_L01:PRES', 'VALCCG:D03_L02:PRES', 'VALCCG:D03_L03:PRES', 'VALCCG:D03_L04:PRES', 
                   'VALCCG:D03_L05:PRES', 'VALCCG:D03_L06:PRES', 'VALCCG:D03_L07:PRES', 'VALCCG:D03_L08:PRES', 
                   'VALCCG:D03_L09:PRES', 'VALCCG:D03_L10:PRES', 'VALCCG:D03_L11:PRES', 'VALCCG:D03_L11A:PRES',
                   'VALCCG:D03_L12:PRES', 'VALCCG:D03_L13:PRES', 'VALCCG:D03_L14:PRES', 'VALCCG:D03_L15:PRES', 
                   'VALCCG:D03_L16:PRES', 'VALCCG:D03_L17:PRES', 'VALCCG:D03_L18:PRES', 'VALCCG:D03_L19:PRES', 
                   'VALCCG:D03_L20:PRES', 'VALCCG:D03_L21:PRES', 'VALCCG:D03_L22:PRES', 'VALCCG:D03_L23:PRES', 
                   'VALCCG:D03_L24:PRES', 'VALCCG:D03_L25:PRES', 'VALCCG:D03_L26:PRES']
    LER_List[3] = ['none', 'VALCCG:D04_L01:PRES', 'VALCCG:D04_L02:PRES', 'VALCCG:D04_L03:PRES', 'VALCCG:D04_L04:PRES', 
                   'VALCCG:D04_L05:PRES', 'VALCCG:D04_L06:PRES', 'VALCCG:D04_L07:PRES', 'VALCCG:D04_L08:PRES', 
                   'VALCCG:D04_L09:PRES', 'VALCCG:D04_L09A:PRES', 'VALCCG:D04_L09B:PRES', 'VALCCG:D04_L09C:PRES', 
                   'VALCCG:D04_L09D:PRES', 'VALCCG:D04_L10:PRES', 'VALCCG:D04_L11:PRES', 'VALCCG:D04_L12:PRES',
                   'VALCCG:D04_L13:PRES', 'VALCCG:D04_L14:PRES', 'VALCCG:D04_L15:PRES', 'VALCCG:D04_L16:PRES', 
                   'VALCCG:D04_L17:PRES', 'VALCCG:D04_L18:PRES', 'VALCCG:D04_L19:PRES', 'VALCCG:D04_L20:PRES', 
                   'VALCCG:D04_L21:PRES', 'VALCCG:D04_L22:PRES', 'VALCCG:D04_L23:PRES', 'VALCCG:D04_L24:PRES']
    LER_List[4] = ['none', 'VALCCG:D05_L01:PRES', 'VALCCG:D05_L02:PRES', 'VALCCG:D05_L03:PRES', 'VALCCG:D05_L04:PRES', 
                   'VALCCG:D05_L05:PRES', 'VALCCG:D05_L06:PRES', 'VALCCG:D05_L07:PRES', 'VALCCG:D05_L08:PRES', 
                   'VALCCG:D05_L09:PRES', 'VALCCG:D05_L10:PRES', 'VALCCG:D05_L11:PRES', 'VALCCG:D05_L12:PRES', 
                   'VALCCG:D05_L13:PRES', 'VALCCG:D05_L14:PRES', 'VALCCG:D05_L15:PRES', 'VALCCG:D05_L16:PRES', 
                   'VALCCG:D05_L17:PRES', 'VALCCG:D05_L18:PRES', 'VALCCG:D05_L20:PRES', 'VALCCG:D05_L21:PRES', 
                   'VALCCG:D05_L22:PRES', 'VALCCG:D05_L22A:PRES', 'VALCCG:D05_L23:PRES', 'VALCCG:D05_L24:PRES', 
                   'VALCCG:D05_L25:PRES', 'VALCCG:D05_L26:PRES']
    LER_List[5] = ['none', 'VALCCG:D06_L01:PRES', 'VALCCG:D06_L02:PRES', 'VALCCG:D06_L03:PRES', 'VALCCG:D06_L04:PRES', 
                   'VALCCG:D06_L05:PRES', 'VALCCG:D06_L06:PRES', 'VALCCG:D06_L07:PRES', 'VALCCG:D06_L08:PRES', 
                   'VALCCG:D06_L09:PRES', 'VALCCG:D06_L10:PRES', 'VALCCG:D06_L11:PRES', 'VALCCG:D06_L12:PRES', 
                   'VALCCG:D06_L13:PRES', 'VALCCG:D06_L14:PRES', 'VALCCG:D06_L15:PRES', 'VALCCG:D06_L16:PRES', 
                   'VALCCG:D06_L17:PRES', 'VALCCG:D06_L18:PRES', 'VALCCG:D06_L19:PRES', 'VALCCG:D06_L20:PRES', 
                   'VALCCG:D06_L21:PRES', 'VALCCG:D06_L22:PRES', 'VALCCG:D06_L23:PRES', 'VALCCG:D06_L24:PRES', 
                   'VALCCG:D06_L25:PRES', 'VALCCG:D06_L26:PRES']
    LER_List[6] = ['none', 'VALCCG:D07_L00:PRES', 'VALCCG:D07_L0:PRES', 'VALCCG:D07_L01:PRES', 'VALCCG:D07_L02:PRES', 
                   'VALCCG:D07_L03:PRES', 'VALCCG:D07_L04:PRES', 'VALCCG:D07_L05:PRES', 'VALCCG:D07_L05A:PRES', 
                   'VALCCG:D07_L06:PRES', 'VALCCG:D07_L07:PRES', 'VALCCG:D07_L08:PRES', 'VALCCG:D07_L09:PRES', 
                   'VALCCG:D07_L10:PRES', 'VALCCG:D07_L11:PRES', 'VALCCG:D07_L12:PRES', 'VALCCG:D07_L13:PRES', 
                   'VALCCG:D07_L14:PRES', 'VALCCG:D07_L15:PRES', 'VALCCG:D07_L16:PRES', 'VALCCG:D07_L17:PRES', 
                   'VALCCG:D07_L18:PRES', 'VALCCG:D07_L19:PRES', 'VALCCG:D07_L20:PRES', 'VALCCG:D07_L21:PRES', 
                   'VALCCG:D07_L22:PRES', 'VALCCG:D07_L23:PRES', 'VALCCG:D07_L24:PRES']
    LER_List[7] = ['none', 'VALCCG:D08_L01:PRES', 'VALCCG:D08_L02:PRES', 'VALCCG:D08_L03:PRES', 'VALCCG:D08_L04:PRES', 
                   'VALCCG:D08_L05:PRES', 'VALCCG:D08_L06:PRES', 'VALCCG:D08_L07:PRES', 'VALCCG:D08_L08:PRES', 
                   'VALCCG:D08_L09:PRES', 'VALCCG:D08_L10:PRES', 'VALCCG:D08_L11:PRES', 'VALCCG:D08_L12:PRES', 
                   'VALCCG:D08_L13:PRES', 'VALCCG:D08_L14:PRES', 'VALCCG:D08_L15:PRES', 'VALCCG:D08_L16:PRES', 
                   'VALCCG:D08_L16X:PRES', 'VALCCG:D08_L17:PRES', 'VALCCG:D08_L18:PRES', 'VALCCG:D08_L0X:PRES']
    LER_List[8] = ['none', 'VALCCG:D09_L01:PRES', 'VALCCG:D09_L02:PRES', 'VALCCG:D09_L03:PRES', 'VALCCG:D09_L04:PRES', 
                   'VALCCG:D09_L05:PRES', 'VALCCG:D09_L06:PRES', 'VALCCG:D09_L07:PRES', 'VALCCG:D09_L08:PRES', 
                   'VALCCG:D09_L09:PRES', 'VALCCG:D09_L10:PRES', 'VALCCG:D09_L11:PRES', 'VALCCG:D09_L12:PRES', 
                   'VALCCG:D09_L13:PRES', 'VALCCG:D09_L14:PRES', 'VALCCG:D09_L15:PRES', 'VALCCG:D09_L16:PRES', 
                   'VALCCG:D09_L17:PRES', 'VALCCG:D09_L18:PRES', 'VALCCG:D09_L19:PRES', 'VALCCG:D09_L20:PRES', 
                   'VALCCG:D09_L21:PRES', 'VALCCG:D09_L22:PRES', 'VALCCG:D09_L23:PRES', 'VALCCG:D09_L24:PRES', 
                   'VALCCG:D09_L25:PRES', 'VALCCG:D09_L26:PRES']
    LER_List[9] = ['none', 'VALCCG:D10_L01:PRES', 'VALCCG:D10_L02:PRES', 'VALCCG:D10_L03:PRES', 'VALCCG:D10_L04:PRES', 
                   'VALCCG:D10_L05:PRES', 'VALCCG:D10_L06:PRES', 'VALCCG:D10_L07:PRES', 'VALCCG:D10_L08:PRES', 
                   'VALCCG:D10_L09:PRES', 'VALCCG:D10_L09A:PRES', 'VALCCG:D10_L09B:PRES', 'VALCCG:D10_L09C:PRES', 
                   'VALCCG:D10_L10:PRES', 'VALCCG:D10_L11:PRES', 'VALCCG:D10_L12:PRES', 'VALCCG:D10_L13:PRES', 
                   'VALCCG:D10_L14:PRES', 'VALCCG:D10_L15:PRES', 'VALCCG:D10_L16:PRES', 'VALCCG:D10_L17:PRES', 
                   'VALCCG:D10_L18:PRES', 'VALCCG:D10_L19:PRES', 'VALCCG:D10_L20:PRES', 'VALCCG:D10_L21:PRES', 
                   'VALCCG:D10_L22:PRES', 'VALCCG:D10_L23:PRES', 'VALCCG:D10_L24:PRES', 'VALCCG:D10_L25:PRES']
    LER_List[10] = ['none', 'VALCCG:D11_L01:PRES', 'VALCCG:D11_L02:PRES', 'VALCCG:D11_L03:PRES', 'VALCCG:D11_L04:PRES', 
                   'VALCCG:D11_L05:PRES', 'VALCCG:D11_L06:PRES', 'VALCCG:D11_L07:PRES', 'VALCCG:D11_L08:PRES', 
                   'VALCCG:D11_L09:PRES', 'VALCCG:D11_L10:PRES', 'VALCCG:D11_L11:PRES', 'VALCCG:D11_L12:PRES', 
                   'VALCCG:D11_L13:PRES', 'VALCCG:D11_L14:PRES', 'VALCCG:D11_L15:PRES', 'VALCCG:D11_L16:PRES', 
                   'VALCCG:D11_L17:PRES', 'VALCCG:D11_L17A:PRES', 'VALCCG:D11_L18:PRES', 'VALCCG:D11_L19:PRES', 
                   'VALCCG:D11_L20:PRES', 'VALCCG:D11_L21:PRES', 'VALCCG:D11_L22:PRES', 'VALCCG:D11_L23:PRES', 
                   'VALCCG:D11_L24:PRES', 'VALCCG:D11_L25:PRES']
    LER_List[11] = ['none', 'VALCCG:D12_L01:PRES', 'VALCCG:D12_L02:PRES', 'VALCCG:D12_L03:PRES', 'VALCCG:D12_L04:PRES', 
                   'VALCCG:D12_L05:PRES', 'VALCCG:D12_L06:PRES', 'VALCCG:D12_L07:PRES', 'VALCCG:D12_L08:PRES', 
                   'VALCCG:D12_L09:PRES', 'VALCCG:D12_L10:PRES', 'VALCCG:D12_L11:PRES', 'VALCCG:D12_L12:PRES', 
                   'VALCCG:D12_L13:PRES', 'VALCCG:D12_L14:PRES', 'VALCCG:D12_L15:PRES', 'VALCCG:D12_L16:PRES', 
                   'VALCCG:D12_L17:PRES', 'VALCCG:D12_L18:PRES', 'VALCCG:D12_L19:PRES', 
                   'VALCCG:D12_L20:PRES', 'VALCCG:D12_L21:PRES', 'VALCCG:D12_L22:PRES', 'VALCCG:D12_L23:PRES', 
                   'VALCCG:D12_L24:PRES', 'VALCCG:D12_L25:PRES', 'VALCCG:D12_L26:PRES']
        
    return LER_List

# ==============================================================================================================
        
def Make_LER_Record_Place(LER_List):
    # D01
    LER_List[0] = ['none', 'QLC1LP (TL12) ZDLM', 'QLC2LP (TL21) GV_D01_L1', 'BLY2LP.1 (TL35)', 'BLY2LP.2 (TL61)', 
                   'QLB2LP (TL74)', 'QLB3LP (TL84) Stopper', 'BLB3LP (TL88) GV_D01_L2', 'BLX2LP.1 (TL98)', 
                   'BLX2LP.2 (TL116)', 'QLA3LP (TL130)', 'QLA5LP (TL140)', 'BLA6LP.2 (TL149)', 
                   'QLA8LP (TL160)', 'B2P.3 (TL166) GV_D01_L4', 'QD1P.1 (TL179)', 'QF2P.2 (TL186)', 
                   'B2P.4 (TL192) Rot SX', 'QEAP.3 (TL205) Rot SX', 'QD5P.3 (TL210)', 'QEAP.4 (TL229)', 
                   'QF4P.3 (TL234) Rot SX', 'B2P.7 (TL242)', 'QD1P.2 (TL255) GV_D01_L5']
    # D02
    LER_List[1] = ['none', 'QF2P.46 (TR248) GV_D02_L1', 'B2P.100 (TR242) Rot SX', 'QEAP.45 (TR229) Rot SX', 'QD5P.45 (TR224)', 
                   'QEAP.46 (TR205)', 'QF4P.39 (TR200) Rot SX', 'B2P.103 (TR191) Rot SX', 'QD1P.24 (TR179)', 
                   'QF2P.48 (TR172)', 'QLA8RP (TR158) GV_D02_L2', 'QLA7RP (TR148)', 'BLA6RP.1 (TR143) Col D02_H1', 
                   'QLA3RP (TR125)', 'BLA2RP (TR121) Col D02_H2', 'QLA2RP (TR116) Col D02_H2', 'BLX2RP.1 (TR113)', 
                   'BLX2RP.2 (TR95)', 'QLB3RP (TR82) Col D02_V1 GV D02_L3', 'QLB3RP (TR76) Stopper', 'QLB2RP (TR67) Col D02_H3', 
                   'BLB2RP (TR65)', 'BLY3RP.1 (TR55)', 'BLY2RP.2 (TR32)', 'QLC2RP (TR18) Col D02_H4',
                   'QLC1RP (TR13) GV_D02_L4']
    # D03
    LER_List[2] = ['none', 'QF2P.40 (OL261) GV_D04_L4', 'B2P.88 (OL268)', 'QEAP.41 (OL281)', 'QD5P.41 (OL286)', 
                   'QEAP.42 (OL304)', 'QF4P.35 (OL309)', 'B2P.91 (OL319)', 'QD1P.21 (OL331)', 
                   'QF2P.42 (OL338)', 'B2P.92 (OL344)', 'QT2OTP.1 (OL352) Col D03_H1', 'QTAOTP.1 (OL357) Col D03_H1', 
                   'QT3OTP.1 (OL362)', 'QT5OTP (TR377)', 'QT4OTP.2 (TR370)', 'QTAOTP.2 (TR358)', 'B2P.95 (TR350)', 
                   'Qf2P.43 (TR338)', 'QD1P.22 (TR331)', 'QF2P.44 (TR324)', 'B2P.96 (TR318)', 
                   'QEAP.43 (TR305)', 'QD5P.43 (TR300) Col D03_V1', 'QEAP.44 (TR281)', 'QF4P.37 (TR276)', 
                   'B2P.99 (TR267)', 'QD1P.23 (TR255) GV_D02_L1']
    # D04
    LER_List[3] = ['none', 'QN7OP (OL9) Wig GV_D04_L1', 'QN8OP (OL18) Wig', 'QDWOP.5 (OL28) Wig', 'QFWOP (OL37) Wig', 
                   'QDWOP.6 (OL47) Wig', 'QW2OLP (OL56) Wig', 'QW3OLP (OL66) Wig', 'QW4OLP (OL75) Wig', 
                   'QW5OLP (OL82) GV_D04_L2', 'QW6OLP (OL90)', 'QW6OLP (OL97)', 'QW7OLP (OL101)', 
                   'QW8OLP (OL106)', 'QSBOP.2 (OL119)', 'QEAP.37 (OL129) GV_D04_L3', 'QD5P.37 (OL134)',
                   'QEAP.38 (OL152)', 'QF4P.31 (OL157)', 'B2P.83 (OL166)', 'QD1P.19 (OL178)', 
                   'QF2P.38 (OL185)', 'B2P.84 (OL192)', 'QEAP.39 (OL205)', 'QD5P.39 (OL210)',
                   'QEAP.40 (OL228)', 'QF4P.33 (OL233)', 'B2P.87 (OL241)', 'QD1P.20 (OL254) GV_D04_L4']
    # D05
    LER_List[4] = ['none', 'QF2P.34 (OR248) GV_D05_L1', 'B2P.74 (OR241)', 'QEAP.33 (OR228)', 'QD5P.33 (OR223)', 
                   'QEAP.34 (OR205)', 'QF4P.29 (OR200)', 'B2P.77 (OR190)', 'QD1P.18 (OR178)', 
                   'QF2P.36 (OR171)', 'B2P.78 (OR165)', 'QEAP.35 (OR152)', 'QD5P.35 (OR147)', 
                   'QEAP.36 (OR129)', 'QS2OP.1 (OR124) GV_D05_L2', 'QS1OP.1 (OR114)', 'QW7ORP (OR101)', 
                   'QW6ORP (OR97)', 'QW5ORP (OR84) GV_D05_L3 HOM', 'QW2ORP (OR56) GV_D05_L4 HOM', 'QN1OP (OR49)', 
                   'QN2OP.1 (OR41)', 'QN3OP (OR34)', 'QN3OP (OR27) Col_D05_V1', 'QN2OP.2 (OR12)', 
                   'QN5OP (OR4)', 'QN6OP (OL0) GV_D04_L1']
    # D06
    LER_List[5] = ['none', 'QF2P.28 (FL261) GV_D07_L6', 'B2P.62 (FL268) EC Test', 'QEAP.29 (FL281) Col_D06_H3', 
                   'QD5P.29 (FL286)', 'QEAP.30 (FL304)', 'QF4P.25 (FL309) Col_D06_H4', 'B2P.65 (FL319)', 
                   'QD1P.15 (FL331) GV_D06_L1', 'QF2P.30 (FL337) GV_D06_L1', 'B2P.66 (FL344)', 'QT2FOP.1 (FL352)', 
                   'QT3FOP.1 (FL362) Col_D06_V1', 'QT5FOP (OR376)', 'QT4FOP.2 (OR369)', 'QTAFOP.2 (OR357)', 
                   'QT1FOP.2 (OR347)', 'B2P.69 (OR343)', 'QD1P.16 (OR331)', 'QF2P.32 (OR324)', 'B2P.70 (OR317)', 
                   'QEAP.31 (OR304)', 'QD5P.31 (OR299)', 'QEAP.32 (OR281)', 'QF4P.27 (OR276) Col_D06_V2', 
                   'B2P.73 (OR266) Col_D06_V2', 'QD1P.17 (OR254) GV_D05_L1']
    # D07
    LER_List[6] = ['none', 'QV3P.1 (FR17) GV_D08_L6 T-FB', 'QV2P.1 (FR12)', 'QV2P.2 (FL12)', 'QV3P.2 (FL17) GV_D07_L1 L-FB', 
                   'QI3P (FL68) DCCT GV_D07_L3', 'QI4P (FL77) Inj-K V-K', 'QI5P (FL83)', 'QKI6P (FL93) Inj', 
                   'Septum-2', 'Septum-1', 'QI7P (FL102) Abt-K', 'QI8P (FL107) Abt-K Inj-K GV_D07_L4', 
                   'QSBFLP (FL119) Dump', 'QEAP.25 (FL129) GV_D07_L5', 'QD5P.25 (FL134) GV_D07_L5', 'QEAP.26 (FL152)', 
                   'QF4P.21 (FL157)', 'B2P.57 (FL165)', 'QD1P.13 (FL178)', 'QF2P.26 (FL185)', 
                   'B2P.58 (FL191)', 'QEAP.27 (FL205)', 'QD5P.27 (FL210)', 'QEAP.28 (FL228)', 
                   'QF4P.23 (FL233)', 'B2P.61 (FL241)', 'QD1P.14 (FL250) GV_D07_L6']
    # D08
    LER_List[7] = ['none', 'QF2P.22 (FR251) GV_D08_L1', 'B2P.46 (FR242)', 'QEAP.21 (FR228)', 'QD5P.21 (FR223)', 
                   'QEAP.22 (FR205)', 'QF4P.19 (FR200)', 'B2P.49 (FR192)', 'QD1P.12 (FR179)', 
                   'QF2P.24 (FR172)', 'B2P.40 (FR165)', 'QEAP.23 (FR152)', 'QD5P.23 (FR147)', 
                   'QEAP.24 (FR129)', 'QS2FRP (FR124) GV_D08_L2', 'QS1FRP (FR114)', 'QR6P (FR107) XRM Ext', 
                   'X-Line_1', 'QR4P (FR94)', 'QR3P (FR86) SRM HOM GV_D08_L4', 'X_Line_2 GV']
    # D09
    LER_List[8] = ['none', 'QF2P.16 (NL258) GV_D10_L4', 'B2P.34 (NL268)', 'QEAP.17 (NL281)', 'QD5P.17 (NL286)', 
                   'QEAP.18 (NL305)', 'QF4P.15 (NL310)', 'B2P.37 (NL318)', 'QD1P.9 (NL331)', 
                   'QF2P.18 (NL338)', 'B2P.38 (NL344)', 'QT2NFP.1 (NL352)', 'QT3NFP.1 (NL363)', 
                   'QT5NFP (FR377)', 'QT4NFP.2 (FR370)', 'QTANFP.2 (FR357)', 'QT1NFP.2 (FR347)', 
                   'B2P.41 (FR344)', 'QD1P.10 (FR331)', 'QF2P.20 (FR324)', 'B2P.42 (FR318)', 
                   'QEAP.19 (FR305)', 'QD5P.19 (FR299)', 'QEAP.20 (FR281)', 'QF4P.17 (FR276)', 
                   'B2P.45 (FR268)', 'QD1P.11 (FR258) GV_D08_L1']
    # D10
    LER_List[9] = ['none', 'QDWNP.4 (NL9) Wig GV_D10_L1', 'QFWNP.4 (NL18) Wig', 'QDWNP.5 (NL28) Wig', 'QFWNP.5 (NL37) Wig', 
                   'QDWNP.6 (NL47) Wig', 'QW2NLP (NL56) Wig', 'QW3NLP (NL66) Wig', 'QW4NLP (NL75) Wig', 
                   'QW5NLP (NL83) GV_D10_L2', 'QW6NLP (NL92)', 'QW6NLP (NL97)', 'QW7NLP (NL102)', 
                   'QW8NLP (NL107)', 'QSBNP.2 (NL119)', 'QEAP.13 (NL129) GV_D10_L3', 'QD5P.13 (NL134)', 
                   'QEAP.14 (NL152)', 'QF4P.11 (NL157)', 'B2P.29 (NL166)', 'QD1P.7 (NL179)', 
                   'QF2P.14 (NL186)', 'B2P.30 (NL192)', 'QEAP.15 (NL205)', 'QD5P.15 (NL210)', 
                   'QEAP.16 (NL228)', 'QF4P.13 (NL234)', 'B2P.33 (NL242)', 'QD1P.8 (NL252) GV_D10_L4']
    # D11
    LER_List[10] = ['none', 'QF2P.10 (NR248) GV_D11_L1', 'B2P.20 (NR236)', 'QEAP.9 (NR228)', 'QD5P.9 (NR223)', 
                   'QEAP.10 (NR205)', 'QF4P.9 (NR200)', 'B2P.23 (NR192)', 'QD1P.6 (NR179)', 
                   'QF2P.12 (NR172)', 'B2P.24 (NR160)', 'QEAP.11 (NR152)', 'QD5P.11 (NR147)', 
                   'QEAP.12 (NR129)', 'QS2NP.1 (NR124) GV_D11_L2', 'QS1NP.1 (NR114)', 'QW7NRP (NR102) Chicane', 
                   'QW5NRP (NR81) Chicane', 'QW4NRP (NR75)', 'QW3NRP (NR71) GV_D11_L3', 'QW3NRP (NR66) Wig GV_D11_L3', 
                   'QW2NRP (NR56) Wig', 'QDWNP.1 (NR47) Wig', 'QFWNP.1 (NR37) Wig', 'QDWNP.2 (NR28) Wig', 
                   'QFWNP.2 (NR18) Wig', 'QDWNP.3 (NR9) Wig GV_D10_l1']
    # D12
    LER_List[11] = ['none', 'QF2P.4 (TL262) GV_D01_L5', 'B2P.8 (TL268) Rot SX', 'QEAP.5 (TL282)', 'QD5P.5 (TL287)', 
                   'QEAP.6 (TL305)', 'QF4P.5 (TL310)', 'B2P.11 (TL319)', 'QD1P.3 (TL331)', 
                   'QF2P.6 (TL338)', 'B2P.12 (TL344)', 'QT2TNP.1 (TL352)', 'QT3TNP.1 (TL363)', 
                   'QT5TNP (NR377)', 'QT4TNP.2 (NR370)', 'QTATNP.2 (NR357)', 'QT1TNP.2 (NR347)', 
                   'B2P.15 (NR343)', 'QD1P.4 (NR331)', 'QF2P.8 (NR324)', 
                   'B2P.16 (NR318)', 'QEAP.7 (NR305)', 'QD5P.7 (NR300)', 'QEAP.8 (NR281)', 
                   'QF4P.7 (NR276)', 'B2P.19 (NR268)', 'QD1P.5 (NR255) GV_D11_L1']
        
    return LER_List
        
# ==============================================================================================================
   
def Make_HER_Record(HER_List):
    HER_List[0] = ['none', 'VAHCCG:D01_H02:PRES', 'VAHCCG:D01_H03:PRES', 'VAHCCG:D01_H04:PRES', 'VAHCCG:D01_H05:PRES', 
                   'VAHCCG:D01_H06:PRES', 'VAHCCG:D01_H07:PRES', 'VAHCCG:D01_H08:PRES', 'VAHCCG:D01_H09:PRES', 
                   'VAHCCG:D01_H10:PRES', 'VAHCCG:D01_H11:PRES', 'VAHCCG:D01_H11A:PRES', 'VAHCCG:D01_H12:PRES', 
                   'VAHCCG:D01_H13:PRES', 'VAHCCG:D01_H13A:PRES', 'VAHCCG:D01_H14:PRES', 'VAHCCG:D01_H15:PRES', 
                   'VAHCCG:D01_H16:PRES', 'VAHCCG:D01_H17:PRES', 'VAHCCG:D01_H18:PRES', 'VAHCCG:D01_H19:PRES',
                   'VAHCCG:D01_H20:PRES', 'VAHCCG:D01_H21:PRES', 'VAHCCG:D01_H22:PRES', 'VAHCCG:D01_H23:PRES',
                   'VAHCCG:D01_H24:PRES', 'VAHCCG:D01_H25:PRES', 'VAHCCG:D01_H26:PRES']
    HER_List[1] = ['none', 'VAHCCG:D02_H01:PRES', 'VAHCCG:D02_H02:PRES', 'VAHCCG:D02_H03:PRES', 'VAHCCG:D02_H04:PRES', 
                   'VAHCCG:D02_H05:PRES', 'VAHCCG:D02_H06:PRES', 'VAHCCG:D02_H07:PRES', 'VAHCCG:D02_H08:PRES',
                   'VAHCCG:D02_H09:PRES', 'VAHCCG:D02_H10:PRES', 'VAHCCG:D02_H11:PRES', 'VAHCCG:D02_H12:PRES',
                   'VAHCCG:D02_H13:PRES', 'VAHCCG:D02_H14:PRES', 'VAHCCG:D02_H15:PRES', 'VAHCCG:D02_H16:PRES',
                   'VAHCCG:D02_H17:PRES', 'VAHCCG:D02_H18:PRES', 'VAHCCG:D02_H19:PRES', 'VAHCCG:D02_H20:PRES',
                   'VAHCCG:D02_H21:PRES', 'VAHCCG:D02_H21A:PRES', 'VAHCCG:D02_H22:PRES', 'VAHCCG:D02_H23:PRES']
    HER_List[2] = ['none', 'VAHCCG:D03_H01:PRES', 'VAHCCG:D03_H02:PRES', 'VAHCCG:D03_H03:PRES', 'VAHCCG:D03_H04:PRES', 
                   'VAHCCG:D03_H05:PRES', 'VAHCCG:D03_H06:PRES', 'VAHCCG:D03_H07:PRES', 'VAHCCG:D03_H08:PRES',
                   'VAHCCG:D03_H09:PRES', 'VAHCCG:D03_H10:PRES', 'VAHCCG:D03_H11:PRES', 'VAHCCG:D03_H12:PRES',
                   'VAHCCG:D03_H13:PRES', 'VAHCCG:D03_H14:PRES', 'VAHCCG:D03_H15:PRES', 'VAHCCG:D03_H16:PRES',
                   'VAHCCG:D03_H17:PRES', 'VAHCCG:D03_H18:PRES', 'VAHCCG:D03_H19:PRES', 'VAHCCG:D03_H20:PRES',
                   'VAHCCG:D03_H21:PRES', 'VAHCCG:D03_H22:PRES', 'VAHCCG:D03_H23:PRES', 'VAHCCG:D03_H24:PRES',
                   'VAHCCG:D03_H25:PRES', 'VAHCCG:D03_H26:PRES']
    HER_List[3] = ['none', 'VAHCCG:D04_H02:PRES', 'VAHCCG:D04_H03:PRES', 'VAHCCG:D04_H04:PRES', 'VAHCCG:D04_H05:PRES', 
                   'VAHCCG:D04_H06:PRES', 'VAHCCG:D04_H06A:PRES', 'VAHCCG:D04_H07:PRES', 'VAHCCG:D04_H08:PRES',
                   'VAHCCG:D04_H08X:PRES', 'VAHCCG:D04_H09:PRES', 'VAHCCG:D04_H10:PRES', 'VAHCCG:D04_H11:PRES', 
                   'VAHCCG:D04_H12:PRES', 'VAHCCG:D04_H13:PRES', 'VAHCCG:D04_H14:PRES', 'VAHCCG:D04_H15:PRES', 
                   'VAHCCG:D04_H16:PRES', 'VAHCCG:D04_H17:PRES', 'VAHCCG:D04_H18:PRES', 'VAHCCG:D04_H19:PRES', 
                   'VAHCCG:D04_H20:PRES', 'VAHCCG:D04_H21:PRES', 'VAHCCG:D04_H22:PRES', 'VAHCCG:D04_H23:PRES',
                   'VAHCCG:D04_H0X:PRES']
    HER_List[4] = ['none', 'VAHCCG:D05_H01:PRES', 'VAHCCG:D05_H02:PRES', 'VAHCCG:D05_H03:PRES', 'VAHCCG:D05_H04:PRES', 
                   'VAHCCG:D05_H05:PRES', 'VAHCCG:D05_H06:PRES', 'VAHCCG:D05_H07:PRES', 'VAHCCG:D05_H08:PRES',
                   'VAHCCG:D05_H09:PRES', 'VAHCCG:D05_H10:PRES', 'VAHCCG:D05_H11:PRES', 'VAHCCG:D05_H12:PRES',
                   'VAHCCG:D05_H13:PRES', 'VAHCCG:D05_H14:PRES', 'VAHCCG:D05_H15:PRES', 'VAHCCG:D05_H15A:PRES',
                   'VAHCCG:D05_H16:PRES', 'VAHCCG:D05_H17:PRES', 'VAHCCG:D05_H18:PRES', 'VAHCCG:D05_H19:PRES',
                   'VAHCCG:D05_H20:PRES', 'VAHCCG:D05_H21:PRES', 'VAHCCG:D05_H22:PRES', 'VAHCCG:D05_H23:PRES',
                   'VAHCCG:D05_H24:PRES', 'VAHCCG:D05_H25:PRES']
    HER_List[5] = ['none', 'VAHCCG:D06_H01:PRES', 'VAHCCG:D06_H02:PRES', 'VAHCCG:D06_H03:PRES', 'VAHCCG:D06_H04:PRES', 
                   'VAHCCG:D06_H05:PRES', 'VAHCCG:D06_H06:PRES', 'VAHCCG:D06_H07:PRES', 'VAHCCG:D06_H08:PRES',
                   'VAHCCG:D06_H09:PRES', 'VAHCCG:D06_H10:PRES', 'VAHCCG:D06_H11:PRES', 'VAHCCG:D06_H12:PRES',
                   'VAHCCG:D06_H13:PRES', 'VAHCCG:D06_H14:PRES', 'VAHCCG:D06_H15:PRES', 'VAHCCG:D06_H16:PRES',
                   'VAHCCG:D06_H17:PRES', 'VAHCCG:D06_H18:PRES', 'VAHCCG:D06_H19:PRES', 'VAHCCG:D06_H20:PRES',
                   'VAHCCG:D06_H21:PRES', 'VAHCCG:D06_H22:PRES', 'VAHCCG:D06_H23:PRES', 'VAHCCG:D06_H24:PRES',
                   'VAHCCG:D06_H25:PRES', 'VAHCCG:D06_H26:PRES']
    HER_List[6] = ['none', 'VAHCCG:D07_H01:PRES', 'VAHCCG:D07_H02:PRES', 'VAHCCG:D07_H03:PRES', 'VAHCCG:D07_H04:PRES', 
                   'VAHCCG:D07_H04A:PRES', 'VAHCCG:D07_H05:PRES', 'VAHCCG:D07_H06:PRES', 'VAHCCG:D07_H07:PRES',
                   'VAHCCG:D07_H08:PRES', 'VAHCCG:D07_H09:PRES', 'VAHCCG:D07_H10:PRES', 'VAHCCG:D07_H11:PRES',
                   'VAHCCG:D07_H12:PRES', 'VAHCCG:D07_H13:PRES', 'VAHCCG:D07_H14:PRES', 'VAHCCG:D07_H15:PRES',
                   'VAHCCG:D07_H16:PRES', 'VAHCCG:D07_H17:PRES', 'VAHCCG:D07_H18:PRES', 'VAHCCG:D07_H19:PRES',
                   'VAHCCG:D07_H20:PRES', 'VAHCCG:D07_H21:PRES']
    HER_List[7] = ['none', 'VAHCCG:D08_H01:PRES', 'VAHCCG:D08_H02:PRES', 'VAHCCG:D08_H03:PRES', 'VAHCCG:D08_H04:PRES', 
                   'VAHCCG:D08_H05:PRES', 'VAHCCG:D08_H06:PRES', 'VAHCCG:D08_H07:PRES', 'VAHCCG:D08_H08:PRES',
                   'VAHCCG:D08_H09:PRES', 'VAHCCG:D08_H10:PRES', 'VAHCCG:D08_H11:PRES', 'VAHCCG:D08_H12:PRES',
                   'VAHCCG:D08_H13:PRES', 'VAHCCG:D08_H14:PRES', 'VAHCCG:D08_H15:PRES', 'VAHCCG:D08_H16:PRES',
                   'VAHCCG:D08_H16A:PRES', 'VAHCCG:D08_H17:PRES', 'VAHCCG:D08_H18:PRES', 'VAHCCG:D08_H19:PRES',
                   'VAHCCG:D08_H20:PRES', 'VAHCCG:D08_H21:PRES', 'VAHCCG:D08_H22:PRES', 'VAHCCG:D08_H23:PRES',
                   'VAHCCG:D08_H24:PRES']
    HER_List[8] = ['none', 'VAHCCG:D09_H01:PRES', 'VAHCCG:D09_H02:PRES', 'VAHCCG:D09_H03:PRES', 'VAHCCG:D09_H04:PRES', 
                   'VAHCCG:D09_H05:PRES', 'VAHCCG:D09_H06:PRES', 'VAHCCG:D09_H07:PRES', 'VAHCCG:D09_H08:PRES',
                   'VAHCCG:D09_H09:PRES', 'VAHCCG:D09_H10:PRES', 'VAHCCG:D09_H11:PRES', 'VAHCCG:D09_H12:PRES',
                   'VAHCCG:D09_H13:PRES', 'VAHCCG:D09_H14:PRES', 'VAHCCG:D09_H15:PRES', 'VAHCCG:D09_H16:PRES',
                   'VAHCCG:D09_H17:PRES', 'VAHCCG:D09_H18:PRES', 'VAHCCG:D09_H19:PRES', 'VAHCCG:D09_H20:PRES',
                   'VAHCCG:D09_H21:PRES', 'VAHCCG:D09_H22:PRES', 'VAHCCG:D09_H23:PRES', 'VAHCCG:D09_H24:PRES',
                   'VAHCCG:D09_H25:PRES', 'VAHCCG:D09_H26:PRES']
    HER_List[9] = ['none', 'VAHCCG:D10_H01:PRES', 'VAHCCG:D10_H02:PRES', 'VAHCCG:D10_H03:PRES', 'VAHCCG:D10_H04:PRES', 
                   'VAHCCG:D10_H05:PRES', 'VAHCCG:D10_H06:PRES', 'VAHCCG:D10_H11:PRES', 'VAHCCG:D10_H12:PRES',
                   'VAHCCG:D10_H13:PRES', 'VAHCCG:D10_H14:PRES', 'VAHCCG:D10_H15:PRES', 'VAHCCG:D10_H15A:PRES',
                   'VAHCCG:D10_H16:PRES', 'VAHCCG:D10_H17:PRES', 'VAHCCG:D10_H18:PRES', 'VAHCCG:D10_H19:PRES',
                   'VAHCCG:D10_H20:PRES', 'VAHCCG:D10_H21:PRES', 'VAHCCG:D10_H22:PRES', 'VAHCCG:D10_H23:PRES',
                   'VAHCCG:D10_H24:PRES', 'VAHCCG:D10_H25:PRES', 'VAHCCG:D10_H26:PRES', 'VAHCCG:D10_H27:PRES',
                   'VAHCCG:D10_H28:PRES']
    HER_List[10] = ['none', 'VAHCCG:D11_H01:PRES', 'VAHCCG:D11_H02:PRES', 'VAHCCG:D11_H03:PRES', 'VAHCCG:D11_H04:PRES', 
                   'VAHCCG:D11_H05:PRES', 'VAHCCG:D11_H06:PRES', 'VAHCCG:D11_H07:PRES', 'VAHCCG:D11_H08:PRES',
                   'VAHCCG:D11_H09:PRES', 'VAHCCG:D11_H10:PRES', 'VAHCCG:D11_H11:PRES', 'VAHCCG:D11_H12:PRES',
                   'VAHCCG:D11_H13:PRES', 'VAHCCG:D11_H14:PRES', 'VAHCCG:D11_H15:PRES', 'VAHCCG:D11_H16:PRES',
                   'VAHCCG:D11_H17:PRES', 'VAHCCG:D11_H18:PRES', 'VAHCCG:D11_H19:PRES']
    HER_List[11] = ['none', 'VAHCCG:D12_H01:PRES', 'VAHCCG:D12_H02:PRES', 'VAHCCG:D12_H03:PRES', 'VAHCCG:D12_H04:PRES', 
                   'VAHCCG:D12_H05:PRES', 'VAHCCG:D12_H06:PRES', 'VAHCCG:D12_H07:PRES', 'VAHCCG:D12_H08:PRES',
                   'VAHCCG:D12_H09:PRES', 'VAHCCG:D12_H10:PRES', 'VAHCCG:D12_H11:PRES', 'VAHCCG:D12_H12:PRES',
                   'VAHCCG:D12_H13:PRES', 'VAHCCG:D12_H14:PRES', 'VAHCCG:D12_H15:PRES', 'VAHCCG:D12_H16:PRES',
                   'VAHCCG:D12_H17:PRES', 'VAHCCG:D12_H18:PRES', 'VAHCCG:D12_H19:PRES', 'VAHCCG:D12_H20:PRES',
                   'VAHCCG:D12_H21:PRES', 'VAHCCG:D12_H22:PRES', 'VAHCCG:D12_H23:PRES', 'VAHCCG:D12_H24:PRES',
                   'VAHCCG:D12_H25:PRES', 'VAHCCG:D12_H26:PRES']
    
    return HER_List

# ==============================================================================================================
   
def Make_HER_Record_Place(HER_List):
    # D01
    HER_List[0] = ['none', 'QLC2LE (TL6)', 'QLC3LE (TL12) GV_D01_H1 Col D01_H5', 'QLC4LE (TL18)', 'QLC5LE (TL22)', 
                   'QLY1LE.1 (TL28) Col D01_H4', 'QLY2LE.2 (TL47) Col D01_H3', 'QLB2LE (TL60) Col D01_V1', 'QLB3LE (TL68)', 
                   'QLB4LE (TL76) Stopper', 'QLX1LE.2 (TL93) GV_D01_H2', 'QLA2LE (TL99)', 'QLA3LE (TL104)', 
                   'QLA5LE (TL118)', 'QLA5LE (TL124)', 'QLA6LE (TL129)', 'QLA9LE (TL151)', 
                   'QLA10LE (TL155) GV_D01_H3', 'B2E.4 (TL167)', 'QF2E.1 (TL174)', 'QD1E.1 (TL179)',
                   'B2E.5 (TL191)', 'QF4E.2 (TL198)', 'QEAE.3 (TL203)', 'QD5E.4 (TL225)',
                   'QEAE.4 (TL231)', 'B2E.8 (TL243)', 'QF2E.3 (TL250) GV_D01_H4']
    # D02
    HER_List[1] = ['none', 'QD1E.23 (TR256) GV_D02_H1', 'B2E.105 (TR244)', 'QF4E.38 (TR237)', 'QEAE.45 (TR233)', 
                   'QD5E.46 (TR210)', 'QD5E.46 (TR210)', 'B2E,108 (TR192)', 'QF2E.47 (TR186)',
                   'QD1E.24 (TR181)', 'B2E.109 (TR169)', 'QF4E.40 (TR162) GV_D02_H2', 'QLA10RE (TR157)',
                   'QLA7RE (TR135)', 'QLA6RE (TR126)', 'QLA5RE (TR122)', 'QLA4RE (TR111) GV_D02_H3',
                   'QLX2RE.1 (TR93) GV_D02_H3 Stopper', 'QLB8RE (TR73)', 'BLB2RE (TR63)', 'QLY2RE.1 (TR54)',
                   'QLY4RE.2 (TR35)', 'QLC7RE (TR27)', 'QLC5RE (TR19) GV_D02_H4', 'QLC3RE (TR13) GV_D02_H4']
    # D03
    HER_List[2] = ['none', 'QD1E.20 (OL256) GV_D04_H5', 'B2E.93 (OL267)', 'QF4E.34 (OL275)', 'QEAE.41 (OL279)', 
                   'QD5E.42 (OL302)', 'QEAE.42 (OL308)', 'B21E.96 (OL320)', 'QF2E.41 (OL326)',
                   'QD1E.21 (OL332)', 'B2E.97 (OL344)', 'QT2OTE.1 (OL351)', 'B2E.98 (OL362)',
                   'QT4OTE.1 (OL369)', 'QTBOTE.2 (TR374)', 'B2E.99 (TR362)', 'QTAOTE.2 (TR355)',
                   'QT1OTE.2 (TR348)', 'QF2E.43 (TR337)', 'QD1E.22 (TR332)', 'B2E.101 (TR321)',
                   'QF4E.36 (TR313)', 'QEAE.43 (TR308)', 'QD5E.44 (TR286)', 'QEAE.44 (TR280)',
                   'B2E.104 (TR268)', 'QF2E.45 (TR262) GV_D02_H1']
    # D04
    HER_List[3] = ['none', 'QFROE.2 (OR19) GV_D05_H4 RF', 'QDROE.3 (OR9)', 'QFROE.3 (OR0)', 'QDROE.4 (OL9)', 
                   'QDROE.5 (OL28) GV_D04_H1', 'QR4OLE (OL75) SRM GV_D04_H3 HOM', 'QR5OLE (OL85)', 'QR6OLE (OL96)',
                   'X-Line_1', 'QS3OLE (OL117) GV_D04_H4', 'QS4OE.2 (OL123)', 'QEAE.37 (OL128)', 
                   'QD5E.38 (OL151)', 'QEAE.38 (OL156)', 'B2E.88 (OL168)', 'QF2E.37 (OL175)', 
                   'QD1E.19 (OL180)', 'B2E.89 (OL193)', 'QF4E.32 (OL199)', 'QEAE.39 (OL204)', 
                   'QD5E.40 (OL226)', 'QEAE.40 (OL232)', 'B2E.92 (OL243)', 'QF2E.39 (OL251) GV_D04_H5', 'X-Line_2 GV']
    # D05
    HER_List[4] = ['none', 'QD1E.17 (OR256) GV_D05_H1', 'B2E.77 (OR243)', 'QF4E.28 (OR237)', 'QEAE.33 (OR232)', 
                   'QD5E.34 (OR210)', 'QEAE.34 (OR204)', 'B2E.80 (OR192)', 'QF2E.35 (OR185)',
                   'QD1E.18 (OR180)', 'B2E.81 (OR168)', 'QF4E.30 (OR161)', 'QEAE.35 (OR156)',
                   'QD5E.36 (OR134)', 'QEAE.36 (OR128) GV_D05_H2', 'QSBOE.1 (OR121)', 'QR7ORE (OR103)',
                   'QR6ORE (OR98) GV_D05_H3', 'QR6ORE (OR91) GV_D05_H3', 'QR5ORE (OR85) Wig', 'QR4ORE (OR75) Wig',
                   'QR3ORE (OR66) Wig', 'QR2ORE (OR57) Wig', 'QDWOE.1 (OR47) Wig', 'QFWOE (OR38) Wig',
                   'QDWOE.2 (OR28) Wig', 'QFROE.2 (OR24) GV_D05_H1 RF']
    # D06
    HER_List[5] = ['none', 'QD1E.14 (FL255) GV_D07_H4', 'B2E.65 (FL268)', 'QF4E.24 (FL274)S', 'QEAE.29 (FL279)', 
                   'QD5E.30 (FL302)', 'QEAE.30 (FL307)', 'B2E.68 (FL320)', 'QF2E.29 (FL326)',
                   'QD1E.15 (FL331)', 'B2E.69 (FL342)', 'QT2FOE.1 (FL351)', 'QT3FOE.1 (FL359)',
                   'QT4FOE.1 (FL368)', 'QTBFOE.2 (OR373)', 'B2E.71 (OR363', 'QTAFOE.2 (OR355)',
                   'B2E.72 (OR343)', 'QD1E.16 (OR331)', 'QD1E.16 (OR331)', 'B2E.73 (OR320)',
                   'QF4E.26 (OR312)', 'QEAE.31 (OR308)', 'QD5E.32 (OR285)', 'QEAE.32 (OR280)',
                   'B2E.76 (OR267)', 'QF2E.33 (OR261) GV_D05_H1']
    # D07
    HER_List[6] = ['none', 'BX1E (FLFL17)', 'QX5LE (FL32) GV_D07_H1 L-FB', 'QX7LE (FL50) L-FB DCCT', 'QM3E (FL67) FB-Mo', 
                   'QM4E (FL76) GV_D07_H2 T-FB', 'QM5E (FL86)', 'VQM7E (FL103) GV_D07_H3', 'QS3FLE (FL116)',
                   'QS4FLE (FL123)', 'QEAE.25 (FL128)', 'QD5E.26 (FL150)', 'QEAE.26 (FL156)',
                   'B2E.60 (FL167)', 'QF2E.25 (FL175)', 'QD1E.13 (FL180)', 'B2E.61 (FL192)',
                   'QF4E.22 (FL199)', 'QEAE.27 (FL203)', 'QD5E.28 (FL226)', 'QEAE.28 (FL232)',
                   'B2E.64 (FL243)', 'QF2E.27 (FL250) GV_D07_H4']
    # D08
    HER_List[7] = ['none', 'QD1E.11 (FR255) GV_D08_H1', 'B2E.49 (FR243)', 'QF4E.18 (FR236)', 'QEAE.21 (FR232)', 
                   'QD5E.22 (FR209) Col D09_V1', 'QEAE.22 (FR203)', 'B2E.52 (FR192)', 'QF2E.23 (FR185)',
                   'QD1E.12 (FR180)', 'B2E.53 (FR167)', 'QF4E.20 (FR161)', 'QEAE.23 (FR156)',
                   'QD5E.24 (FR133)', 'QEAE.24 (FR128)', 'QI7E (FR105) GV_D08_H2 Inj-K', 'QI5E (FR87) V-K Inj-K',
                   'QI4E (FR81) Inj', 'Septum-1', 'Septume-2', 'Septume-3',
                   'Septume-4', 'QI3E (FR68)', 'QI2E (FR59) Inj-K', 'QX7RE (FR50) Dump Abt-K',
                   'QX4RE (FR25) GV_D08_H4 Abt-K']
    # D09
    HER_List[8] = ['none', 'QD1E.8 (NL255) GV_D10_H5', 'B2E.37 (NL267)', 'QF4E.14 (NL274) Col D09_H2', 'QEAE.17 (NL279)', 
                   'QD5E.18 (NL301)', 'QEAE.18 (NL307) Col_D09_H1', 'B2E.40 (NL318)', 'QF2E.17 (NL326)',
                   'QD1E.9 (NL331)', 'B2E.41 (NL343)', 'QT2NFE.1 (NL350)', 'B2E.42 (NL362)',
                   'QT4NFE.1 (NL368)', 'QTBNFE.2 (FR372)', 'B2E.43 (FR363) Col D09_V2', 'QTANFE.2 (FR354) Col D09_V2',
                   'B2E.44 (FR343)', 'QF2E.19 (FR336)', 'QD1E.10 (FR331)', 'B2E.45 (FR319)',
                   'QF4E.16 (FR312)', 'QEAE.19 (FR307)', 'QD5E.20 (FR285)', 'QEAE.20 (FR279)',
                   'B2E.48 (FR277)', 'QF2E.21 (FR260) GV_D08_H1']
    # D10
    HER_List[9] = ['none', 'QDRNE.2 (NR28) GV_D11_H5', 'QFRNE.2 (NR19)', 'QDRNE.3 (NR9)', 'QFRNE.3 (NR0)', 
                   'QDRNE.4 (NL9)', 'QFRNE.4 (NL19) GV_D10_H1 SCC', 'QR3NE.2 (NL66) GV_D10_H2 SCC', 'QR4NE.2 (NL75) GV_D10_H3',
                   'QR5NE.2 (NL85) GV_D10_H3 GV_D10_H4', 'QR6NE.2 (NL98) GV_D10_H4', 'QS3NE.2 (NL116)', 'QS4NE.2 (NL123)',
                   'QEAE.13 (NL128)', 'QD5E.14 (NL150) Col D09_V4', 'QEAE.14 (NL156)', 'B2E.32 (NL168)',
                   'QF2E.13 (NL175)', 'QD1E.7 (NL180)', 'B2E.33 (NL191)', 'QF4E.12 (NL199) Col D09_H4',
                   'QF4E.12 (NL199)', 'QD5E.16 (NL226) Col D09_V3', 'QEAE.16 (NL232) Col D09_H3', 'B2E.36 (NL243)',
                   'QF2E.15 (NL250) GV_D10_H5']
    # D11
    HER_List[10] = ['none', 'QD1E.5 (NR255)', 'B2E.21 (NR243)', 'QF4E.8 (NR236) Col D12_H2', 'QEAE.9 (NR232)', 
                   'QD5E.10 (NR209) Col D12_V2', 'QEAE.10 (NR204) Col D12_H1', 'B2E.24 (NR191)', 'QF2E.11 (NR185) GV_D11_H1',
                   'QD1E.6 (NR180) GV_D11_H1', 'B2E.25 (NR169)', 'QF4E.10 (NR161)', 'QEAE.11 (NR156)',
                   'QD5E.12 (NR134) Col D12_V1', 'QEAE.12 (NR128)', 'QSBNE.1 (NR120)', 'QR7NE.1 (NR103) GV_D11_H2',
                   'VQR6NE.1 (NR98) GV_D11_H2 GV_D11_H3', 'QR5NE.1 (NR85) GV_D11_H3', 'QR4NE.1 (NR75) GV_D11_H4 SCC']
    # D12
    HER_List[11] = ['none', 'QD1E.2 (TL255) GV_D01_H4', 'B2E.9 (TL267)', 'QF4E.4 (TL274)', 'QEAE.5 (TL278)', 
                   'QD5E.6 (TL301)', 'QEAE.6 (TL307)', 'B2E.12 (TL318)', 'QF2E.5 (TL325)',
                   'QD1E.3 (TL331)', 'B2E.13 (TL349)', 'QT2TNE.1 (TL350)', 'B2E.14 (TL361)',
                   'QT4TNE.1 (TL368)', 'QTBTNE.2 (NR372)', 'B2E.15 (NR361) Col D12_V4', 'QTATNE.2 (NR354) Col D12_V4',
                   'B2E.16 (NR344)', 'QF2E.7 (NR336) GV_D12_H1', 'QD1E.4 (NR331) GV_D12_H1', 'B2E.17 (NR320)',
                   'QF4E.6 (NR312) Col D12_H4', 'QEAE.7 (NR307)', 'QD5E.8 (NR285) Col D12_V3', 'QEAE.8 (NR279) Col D12_H3',
                   'B2E.20 (NR267)', 'QF2E.9 (NR260)']
    
    return HER_List

# ==============================================================================================================

def Main_Command_LER(self):
    # エラーフラグリセット
    Error_f = 'none'
    
    # 状態表示変更
    self.Status_change_running()
    self.LER_ABN_canvas["highlightthickness"] = 5
    self.LER_ABN_canvas["highlightbackground"] = 'lightgreen'
    self.LER_ABN_canvas.update()
        
    # リングの名前
    Ring_Name = 'LER'
    List_Para = self.LER_CCG_List
    
    # 各側室のCCGの数+1
    LER_CCG_n = [0] * 12
    for k in range(12):
        LER_CCG_n[k] = len(self.LER_Record_Name_List_box[k])
        # print(k, LER_CCG_n[k])
        
    # abnormal, normalをfileに保存するかどうか (LER, HER共通)
    File_Save_Para = self.Save_var.get()
    
    # Abort Timing
    Abort_Timing = self.LER_Abort_Time_entry.get()
    
    try:
        date_dt = datetime.datetime.strptime(Abort_Timing, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(traceback.format_exc())
        print(str(e))
        Error_f = str(e)
        tk.messagebox.showinfo('Check LER Time format', Error_f)
        self.LER_ABN_canvas["highlightthickness"] = 0
        self.LER_ABN_canvas.update()
        return
    
    # ランモードの確認 auto かmanualか
    n = self.Mode_var.get()
    Run_Mode = self.Mode_rdo_text[n]
        
    # Auto時のモードの確認　Abort triggerかIntervalか
    m = self.Auto_Mode_var.get()
    Auto_Mode = self.Auto_Mode_rdo_text[m]
    
    # constモード(Triggerモード中のconstモード)
    Cns_Mode = self.LER_trg_cns
    
    # 時間間隔
    try:
        self.LER_Interval_h = self.Interval_entry.get()
        Hadv = float(self.LER_Interval_h)
    except Exception as e:
        print(traceback.format_exc())
        print(str(e))
        Error_f = str(e)
        tk.messagebox.showinfo('Check interval time format', Error_f)
        self.LER_ABN_canvas["highlightthickness"] = 0
        self.LER_ABN_canvas.update()
        return
    
    # Referenceデータの最終日
    try:
        self.LER_Last_Ref_d = self.Last_Ref_entry.get()
        Last_Refd = float(self.LER_Last_Ref_d)
    except Exception as e:
        print(traceback.format_exc())
        print(str(e))
        Error_f = str(e)
        tk.messagebox.showinfo('Check Last Ref Day format', Error_f)
        self.LER_ABN_canvas["highlightthickness"] = 0
        self.LER_ABN_canvas.update()
        return
    
    # Referenceデータの期間(日)
    try:
        self.LER_Ref_Period_d = self.Ref_Period_entry.get()
        Ref_Pd = float(self.LER_Ref_Period_d)
    except Exception as e:
        print(traceback.format_exc())
        print(str(e))
        Error_f = str(e)
        tk.messagebox.showinfo('Check Ref Period format', Error_f)
        self.LER_ABN_canvas["highlightthickness"] = 0
        self.LER_ABN_canvas.update()
        return
    
    # Auto_modeで走っているかどうか
    Arun = self.LER_auto_run
    
    # グラフでチェックするレコード名
    # noneならプロットしない
    Check_Record_Name = self.LER_Record_Name_entry.get()
        
    # アボートタイミングから、各モードのDate_Range(データの時間範囲)を決める
    # ビームがあるかどうかも調べる。No_Beam = 1 ならビームなし(シャットダウン中とか)、0 ならビームあり
    No_Beam, Error_f = Define_Date_Range(Abort_Timing, List_Para, Run_Mode, Auto_Mode, Cns_Mode, Arun, Hadv, Last_Refd, Ref_Pd)
    # 20240202変更
    # STDデータ、DIFデータにアボートがあるか
    No_STD_Abort = 0
    # No_DIF_Abort = 0
    # print(8405, Error_f)
    if(Error_f == 'No_STD_Abort'): # STDにアボートが無かったら
        No_STD_Abort = 1
    # elif(Error_f == 'No_DIF_Abort'): # DIFにアボートが無かったら
    #     No_DIF_Abort = 1
    elif(Error_f != 'none'):
        tk.messagebox.showinfo('LER STD Strg Error', Error_f)
        return
    
    if (No_Beam == 0): # ビームがあるなら
        self.LER_Beam_Mode_label["text"] = 'With Beam'
    else:
        self.LER_Beam_Mode_label["text"] = 'No Beam'
    
    # 選別する方法
    Method = 'FNN'  # FNN 回帰モデルの結果を使い自動で選別

    # Plot_Para = 1 で　Method = Manual ならならプロットで確認する。
    Plot_Para = 0

    print(' ')
    print('Please wait for a while')
    
    # ##########################################################################################################
    # ビーム蓄積時(Strg)の解析(STDモード、スタンダードデータ) 
    
    # ビームがあったら
    if(No_Beam == 0): 
        
        # 入射蓄積時(Strg)の解析(STDモード、スタンダードデータ)
        Date_Range_STD, Error_fss = Get_Fit_STD_Strg(List_Para, Method, Ref_Pd, LER_CCG_n)
        
        # エラーがあったら、
        if(Error_fss != 'none'):
            tk.messagebox.showinfo('LER STD Strg Error', Error_fss)
            self.LER_ABN_canvas["highlightthickness"] = 0
            self.LER_ABN_canvas.update()
            return
        # エラーがなかったら
        else:    
            # STD_Strgのプロット(横軸ビーム電流)
            Mode_Para = 'STD_Strg'
            Make_Plot_Strg(Method, Check_Record_Name, Date_Range_STD, Mode_Para, List_Para, Abort_Timing)
            # プロットの保存
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_STD + '.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
        
            # STD_Strgのプロット(横軸時刻)
            Make_Plot_Strg_Time(Method, Check_Record_Name, Date_Range_STD, Mode_Para, List_Para, Abort_Timing)
            # プロットの保存
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_STD + '_Time.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
        
        # STDにアボートがあったら、
        if(No_STD_Abort == 0):
            # ビームアボート後(Tail)の解析(STDモード、スタンダードデータ)
            Date_Range_STD, Error_fst = Get_Fit_STD_Tail(List_Para, Method)
        
            # エラーがあったら、
            if(Error_fst != 'none'):
                tk.messagebox.showinfo('LER STD Tail Error', Error_fst)
                return
            # エラーが無かったら
            else:
                # STD_Tailのプロット
                Mode_Para = 'STD_Tail'
                Make_Plot_Tail(Method, Check_Record_Name, Date_Range_STD, Mode_Para, List_Para, Abort_Timing)
                # プロットの保存
                fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_STD + '.png'
                if(Check_Record_Name != 'none'):
                    plt.savefig(fig_name)
                # plt.show()
                plt.clf()
                plt.close()
        
    else: #ビームが無かったら (No_Beam = 1)
        Date_Range_STD, Error_fss = Get_Fit_STD_Strg_NB(List_Para, Method, Ref_Pd, LER_CCG_n)
        
        # エラーがあったら、
        if(Error_fss != 'none'):
            tk.messagebox.showinfo('LER STD Strg Error', Error_fss)
            tk.messagebox.showinfo('LER STD Strg Error', 'May be no kblog data')
            return
        # エラーが無かったら
        else:
            # STD_Strgのプロット(横軸ビーム電流)
            Mode_Para = 'STD_Strg'
            Make_Plot_Strg_NB(Method, Check_Record_Name, Date_Range_STD, Mode_Para, List_Para, Abort_Timing)
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_STD + '.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
            
            # STD_Strgのプロット(横軸時刻)
            Make_Plot_Strg_Time(Method, Check_Record_Name, Date_Range_STD, Mode_Para, List_Para, Abort_Timing)
            # プロットの保存
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_STD + '_Time.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
    
    # ##########################################################################################################
    # アボート直前の蓄積中、アボート直後の圧力をチェックする(CHKモード)
    # 蓄積時とアボート直後続けてリングの各側室について直前のビーム蓄積中のデータを読み出し、
    # 各レコードについて最小二乗誤差法で回帰曲線を計算する。結果を辞書にして保存する。
    
    # 初期値
    Cn_Beam = 0
    Error_fct = 'none'
    Error_fcs = 'none'
    
    # ビームがあったら
    if(No_Beam == 0): 
        Mode_Para = 'CHK_Strg'
        Date_Range_CHK, Error_fcs = Get_Fit_CHK_Strg(Method, Check_Record_Name, Mode_Para, List_Para, Abort_Timing, LER_CCG_n)
        Date_Range_DIF = Date_Range_CHK
        
        # エラーがあったら、
        if(Error_fcs != 'none'):
            tk.messagebox.showinfo('LER CHK Strg Error', Error_fcs)
            # return
        # エラーが無かったら
        else:
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_CHK + '.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
            
            # CHK_Strgのプロット(横軸時刻)
            Make_Plot_Strg_Time(Method, Check_Record_Name, Date_Range_CHK, Mode_Para, List_Para, Abort_Timing)
            # プロットの保存
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_CHK + '_Time.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
        
        # もしSTDにアボートがあったら
        if (No_STD_Abort == 0):
            Mode_Para = 'CHK_Tail'
            Date_Range_CHK, Error_fct = Get_Fit_CHK_Tail(Method, Check_Record_Name, Mode_Para, List_Para, Abort_Timing)
        
            # エラーがあったら、
            if(Error_fct != 'none') and (Error_fct != 'No Tail Data'):
                tk.messagebox.showinfo('LER CHK Tail Error', Error_fct)
                # return
            # ビームが有って、Tail dataが無かったらかつアボートが無かったら(Cn_Beam = 1)
            elif(Error_fct == 'No Tail Data'):
                Cn_Beam = 1
            # アボートがあったら
            else:
                fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_CHK + '.png'
                if(Check_Record_Name != 'none'):
                    plt.savefig(fig_name)
                # plt.show()
                plt.clf()
                plt.close()
        
    else: # ビームが無かったら
        Mode_Para = 'CHK_Strg'
        Date_Range_CHK, Error_fcs = Get_Fit_CHK_Strg_NB(Method, Check_Record_Name, Mode_Para, List_Para, Abort_Timing, LER_CCG_n)
        Date_Range_DIF = Date_Range_CHK
        
        # エラーがあったら、
        if(Error_fcs != 'none'):
            tk.messagebox.showinfo('LER CHK Strg Error', Error_fcs)
            self.LER_ABN_canvas["highlightthickness"] = 0
            self.LER_ABN_canvas.update()
            # return
        # エラーが無かったら
        else:
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_CHK + '.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
            
            # STD_Strgのプロット(横軸時刻)
            Make_Plot_Strg_Time(Method, Check_Record_Name, Date_Range_CHK, Mode_Para, List_Para, Abort_Timing)
            # プロットの保存
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_CHK + '_Time.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
        
    # ##########################################################################################################
    # アボート直前の蓄積中、アボート直後の圧力を調べ、
    # スタンダードの回帰曲線での予想とズレの大きなレコードをみつける(DIFモード)
    # 蓄積時(Strg)とアボート直後(Tail)を続けて調べる
    
    # クラス分け方法の確認 SDM か Keras か
    n = self.ANN_Method_var.get()
    Class_Method = self.ANN_Method_rdo_text[n]
    
    if(No_Beam == 0): # ビームがあったら
        # 20240131変更
        # ビームがあってTail dataもあったら
        if(Error_fcs == 'none') and (Error_fct == 'none'):
            # リングの各側室についてMSE計算に用いるデータを取り出す
            # Date_Range_DIFをファイルに保存する
    
            Mode_Para = 'DIF_Strg'
            Date_Range_DIF, Error_fds = Get_DIF_Strg(List_Para, LER_CCG_n)

            # エラーがあったら、
            if(Error_fds != 'none'):
                tk.messagebox.showinfo('LER Error', Error_fds)
                # return
            # エラーが無かったら
            else:
                N_Rec_Strg = Find_Abnormal_Strg(Method, Mode_Para, List_Para, File_Save_Para, Plot_Para, Abort_Timing, Class_Method, Check_Record_Name)
                self.LER_N_Rec_Strg_label["text"] = str(N_Rec_Strg)
                
            # もしSTDにアボートがあったら
            if (No_STD_Abort == 0):
                Mode_Para = 'DIF_Tail'
                Date_Range_DIF, Error_fdt = Get_DIF_Tail(List_Para)

                # エラーがあったら、
                if(Error_fdt != 'none'):
                    tk.messagebox.showinfo('LER Error', Error_fdt)
                    # return
                # エラーが無かったら
                else:
                    N_Rec_Tail = Find_Abnormal_Tail(Method, Mode_Para, List_Para, File_Save_Para, Plot_Para, Abort_Timing, Class_Method, Check_Record_Name)
                    self.LER_N_Rec_Tail_label["text"] = str(N_Rec_Tail)
    
        # 20240131変更
        #ビームが有って、かつ(Tail dataが無いか(Cn_Beam = 1)、あるいは、STDデータにアボートが無かったら(No_STD_Abort = 1))
        elif(Error_fcs == 'none') and ((Cn_Beam == 1) or (No_STD_Abort == 1)): 
            # リングの各側室についてMSE計算に用いるデータを取り出す
            # Date_Range_DIFをファイルに保存する
    
            Mode_Para = 'DIF_Strg'
            Date_Range_DIF, Error_fds = Get_DIF_Strg(List_Para, LER_CCG_n)

            # エラーがあったら、
            if(Error_fds != 'none'):
                tk.messagebox.showinfo('LER Error', Error_fds)
                # return
            # エラーが無かったら
            else:
                N_Rec_Strg = Find_Abnormal_Strg(Method, Mode_Para, List_Para, File_Save_Para, Plot_Para, Abort_Timing, Class_Method, Check_Record_Name)
                self.LER_N_Rec_Strg_label["text"] = str(N_Rec_Strg)
                
    else: # ビームが無かったら
        if(Error_fcs == 'none'):
            Mode_Para = 'DIF_Strg'
            Date_Range_DIF, Error_fds = Get_DIF_Strg_NB(List_Para, LER_CCG_n)

            # エラーがあったら、
            if(Error_fds != 'none'):
                tk.messagebox.showinfo('LER Error', Error_fds)
                self.LER_ABN_canvas["highlightthickness"] = 0
                self.LER_ABN_canvas.update()
                return
            # エラーが無かったら
            else:
                N_Rec_Strg = Find_Abnormal_Strg_NB(Method, Mode_Para, List_Para, File_Save_Para, Plot_Para, Abort_Timing, Class_Method, Check_Record_Name)
                self.LER_N_Rec_Strg_label["text"] = str(N_Rec_Strg)
                
                N_Rec_Tail = Find_Abnormal_Tail_NB(Method, Mode_Para, List_Para, File_Save_Para, Plot_Para, Abort_Timing)
                self.LER_N_Rec_Tail_label["text"] = str(N_Rec_Tail)

    # ##########################################################################################################
    # Abnormalになったレコードの頻度のトレンドをプロットする。
    
    # プロットする点数(nplot+nfill前のfillから）# ----------------------------------------------------????????
    Nplot = 12
    # nfill分の計 # -------------------------------------------------------------------------------------????????
    Nfill = 8 
    # プロットするRedordの最大数 # -----------------------------------------------------------------------????????
    Nrecord = 5 
    
    # プロットし、abnormalの数(abmax)でcanvasの枠の色を変える
    abmax, Cause_Record_Name_Strg, Cause_Record_Name_Tail = Record_Freq_plot(self, List_Para, Method, Nplot, Nfill, 
                                                                        Nrecord, No_Beam, Cn_Beam)
    
    Abort_Timing_d = datetime.datetime.strptime(Abort_Timing, '%Y-%m-%d %H:%M:%S')
    Abort_Timing_k = '{:%Y%m%d%H%M%S}'.format(Abort_Timing_d) # 書式変更
    
    if(abmax == -1):
        print('no abnormal or normal list')
        self.LER_ABN_canvas["highlightthickness"] = 0
    else:
        if(abmax > 6 ):
            self.LER_ABN_canvas["highlightthickness"] = 5
            self.LER_ABN_canvas["highlightbackground"] = 'red'
        elif(abmax > 4 and abmax <= 6):
            self.LER_ABN_canvas["highlightthickness"] = 5
            self.LER_ABN_canvas["highlightbackground"] = 'hotpink'
        else:
            self.LER_ABN_canvas["highlightthickness"] = 0
    
        fig_name = Ring_Name + '_Trend_' + Abort_Timing_k + '.png'
        plt.savefig(fig_name, dpi = 400)
        # plt.show()
        plt.clf()
        plt.close()
    
    # ##########################################################################################################
    # プロット、Abort_Timingリストをアップデートする。
    
    # LER_STD_Strg(P vS I)
    # 画像ファイルがあるか
    plot_file_name = 'LER_STD_Strg_' + Check_Record_Name + '_' + Date_Range_STD + '.png'
    path = plot_file_name
    is_file = os.path.isfile(path)
    if(is_file == False): # ファイルがなかったら
        self.LER_STD_Strg_I_pil_image = Image.open("No_data.png")
    else:
        self.LER_STD_Strg_I_pil_image = Image.open(plot_file_name)
    
    self.LER_STD_Strg_I_canvas.delete('p1')
    w = self.LER_STD_Strg_I_pil_image.width
    h = self.LER_STD_Strg_I_pil_image.height
    self.LER_STD_Strg_I_pil_image = self.LER_STD_Strg_I_pil_image.resize((int(w * (self.LER_STD_Strg_I_canvas_width / w)), 
                                                                int(h * (self.LER_STD_Strg_I_canvas_height / h))))
    self.LER_STD_Strg_I_plot = ImageTk.PhotoImage(image = self.LER_STD_Strg_I_pil_image, master = self.LER_STD_frame)
    # 画像を描画
    self.LER_STD_Strg_I_canvas.create_image(5 + self.LER_STD_Strg_I_canvas_width / 2, 5 + self.LER_STD_Strg_I_canvas_height / 2, 
                                       image = self.LER_STD_Strg_I_plot, tag = 'p1')
        
    # ラベル変更
    Date_Range_STD_1 = Convert_Kblogrd_to_Dtime(Date_Range_STD[:14])
    Date_Range_STD_1d = '{:%Y-%m-%d}'.format(Date_Range_STD_1)
    Date_Range_STD_2 = Convert_Kblogrd_to_Dtime(Date_Range_STD[15:])
    Date_Range_STD_2d = '{:%Y-%m-%d}'.format(Date_Range_STD_2)
    self.LER_STD_AT_label["text"] = 'Date range  = ' + Date_Range_STD_1d + ' - ' + Date_Range_STD_2d
    self.LER_STD_RC_label["text"] = 'Record name = ' + Check_Record_Name
        
    # LER_STD_Strg(P vs T)
    # 画像ファイルがあるか
    plot_file_name = 'LER_STD_Strg_' + Check_Record_Name + '_' + Date_Range_STD + '_Time.png'
    path = plot_file_name
    is_file = os.path.isfile(path)
    if(is_file == False): # ファイルがなかったら
        self.LER_STD_Strg_T_pil_image = Image.open("No_data.png")
    else:
        self.LER_STD_Strg_T_pil_image = Image.open(plot_file_name)
        
    self.LER_STD_Strg_T_canvas.delete('p2')
    w = self.LER_STD_Strg_T_pil_image.width
    h = self.LER_STD_Strg_T_pil_image.height
    self.LER_STD_Strg_T_pil_image = self.LER_STD_Strg_T_pil_image.resize((int(w * (self.LER_STD_Strg_T_canvas_width / w)), 
                                                                int(h * (self.LER_STD_Strg_T_canvas_height / h))))
    self.LER_STD_Strg_T_plot = ImageTk.PhotoImage(image = self.LER_STD_Strg_T_pil_image, master = self.LER_STD_frame)
    # 画像を描画
    self.LER_STD_Strg_T_canvas.create_image(5 + self.LER_STD_Strg_T_canvas_width / 2, 5 + self.LER_STD_Strg_T_canvas_height / 2, 
                                       image = self.LER_STD_Strg_T_plot, tag = 'p2')
    
    # LER_STD_Tail
    # 画像ファイルがあるか
    plot_file_name = 'LER_STD_Tail_' + Check_Record_Name + '_' + Date_Range_STD + '.png'
    path = plot_file_name
    is_file = os.path.isfile(path)
    if(is_file == False): # ファイルがなかったら
        self.LER_STD_Tail_pil_image = Image.open("No_data.png")
    else:
        self.LER_STD_Tail_pil_image = Image.open(plot_file_name)
   
    self.LER_STD_Tail_canvas.delete('p3')
    w = self.LER_STD_Tail_pil_image.width
    h = self.LER_STD_Tail_pil_image.height
    self.LER_STD_Tail_pil_image = self.LER_STD_Tail_pil_image.resize((int(w * (self.LER_STD_Tail_canvas_width / w)), 
                                                            int(h * (self.LER_STD_Tail_canvas_height / h))))
    self.LER_STD_Tail_plot = ImageTk.PhotoImage(image = self.LER_STD_Tail_pil_image, master = self.LER_STD_frame)
    # 画像を描画
    self.LER_STD_Tail_canvas.create_image(5 + self.LER_STD_Tail_canvas_width / 2, 5 + self.LER_STD_Tail_canvas_height / 2, 
                                     image = self.LER_STD_Tail_plot, tag = 'p3')    
    
    # LER_CHK_Strg(P vs I)
    # 画像ファイルがあるか
    plot_file_name = 'LER_CHK_Strg_' + Check_Record_Name + '_' + Date_Range_CHK + '.png'
    path = plot_file_name
    is_file = os.path.isfile(path)
    if(is_file == False): # ファイルがなかったら
        self.LER_CHK_Strg_I_pil_image = Image.open("No_data.png")
    else:
        self.LER_CHK_Strg_I_pil_image = Image.open(plot_file_name)
    
    self.LER_CHK_Strg_I_canvas.delete('p1')
    w = self.LER_CHK_Strg_I_pil_image.width
    h = self.LER_CHK_Strg_I_pil_image.height
    self.LER_CHK_Strg_I_pil_image = self.LER_CHK_Strg_I_pil_image.resize((int(w * (self.LER_CHK_Strg_I_canvas_width / w)), 
                                                                int(h * (self.LER_CHK_Strg_I_canvas_height / h))))
    self.LER_CHK_Strg_I_plot = ImageTk.PhotoImage(image = self.LER_CHK_Strg_I_pil_image, master = self.LER_CHK_frame)
    # 画像を描画
    self.LER_CHK_Strg_I_canvas.create_image(5 + self.LER_CHK_Strg_I_canvas_width / 2, 5 + self.LER_CHK_Strg_I_canvas_height / 2, 
                                       image = self.LER_CHK_Strg_I_plot, tag = 'p1')
    
    # ラベル変更
    Date_Range_DIF_1 = Convert_Kblogrd_to_Dtime(Date_Range_DIF[:14])
    Date_Range_DIF_1d = '{:%m-%d %H:%M}'.format(Date_Range_DIF_1)
    Date_Range_DIF_2 = Convert_Kblogrd_to_Dtime(Date_Range_DIF[15:])
    Date_Range_DIF_2d = '{:%m-%d %H:%M}'.format(Date_Range_DIF_2)
    self.LER_CHK_AT_label["text"] = 'Date range  = ' + Date_Range_DIF_1d + ' - ' +  Date_Range_DIF_2d
    self.LER_CHK_RC_label["text"] = 'Record name = ' + Check_Record_Name
        
    # アブノーマルなレコードだったらフレームの色を赤に変更する変更
    if(No_Beam == 0): # ビームがあったら
        Abnormal_Result_Strg_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_WB.npy'
        Abnormal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_WB.txt'
        Abnormal_Result_Strg_List = Check_Strg_Abnormal_Result_File(Abnormal_Result_Strg_File_Name, 
                                                                   Abnormal_Result_Strg_Text_File_Name)
    if(No_Beam == 1): # ビームが無かったら
        Abnormal_Result_Strg_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_NB.npy'
        Abnormal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_NB.txt'
        Abnormal_Result_Strg_List = Check_Strg_Abnormal_Result_File(Abnormal_Result_Strg_File_Name, 
                                                                   Abnormal_Result_Strg_Text_File_Name)
    nab = 0
    for nn in range(len(Abnormal_Result_Strg_List)):
        if(Abnormal_Result_Strg_List[nn, 0] == Date_Range_DIF) and (Abnormal_Result_Strg_List[nn, 3] == Check_Record_Name):
            nab = 1
            break
    if(nab == 1):
        self.LER_CHK_Strg_I_canvas["highlightthickness"] = 3
        self.LER_CHK_Strg_I_canvas["highlightbackground"] = 'red'
    else:
        self.LER_CHK_Strg_I_canvas["highlightthickness"] = 0
        
    # LER_CHK_Strg(P vs T)
    # 画像ファイルがあるか
    plot_file_name = 'LER_CHK_Strg_' + Check_Record_Name + '_' + Date_Range_CHK + '_Time.png'
    path = plot_file_name
    is_file = os.path.isfile(path)
    if(is_file == False): # ファイルがなかったら
        self.LER_CHK_Strg_T_pil_image = Image.open("No_data.png")
    else:
        self.LER_CHK_Strg_T_pil_image = Image.open(plot_file_name)
        
    self.LER_CHK_Strg_T_canvas.delete('p2')
    w = self.LER_CHK_Strg_T_pil_image.width
    h = self.LER_CHK_Strg_T_pil_image.height
    self.LER_CHK_Strg_T_pil_image = self.LER_CHK_Strg_T_pil_image.resize((int(w * (self.LER_CHK_Strg_T_canvas_width / w)), 
                                                                int(h * (self.LER_CHK_Strg_T_canvas_height / h))))
    self.LER_CHK_Strg_T_plot = ImageTk.PhotoImage(image = self.LER_CHK_Strg_T_pil_image, master = self.LER_CHK_frame)
    # 画像を描画
    self.LER_CHK_Strg_T_canvas.create_image(5 + self.LER_CHK_Strg_T_canvas_width / 2, 5 + self.LER_CHK_Strg_T_canvas_height / 2, 
                                       image = self.LER_CHK_Strg_T_plot, tag = 'p2')
    
    # LER_CHK_Tail
    # 画像ファイルがあるか
    plot_file_name = 'LER_CHK_Tail_' + Check_Record_Name + '_' + Date_Range_CHK + '.png'
    path = plot_file_name
    is_file = os.path.isfile(path)
    if(is_file == False): # ファイルがなかったら
        self.LER_CHK_Tail_pil_image = Image.open("No_data.png")
    else:
        self.LER_CHK_Tail_pil_image = Image.open(plot_file_name)
   
    self.LER_CHK_Tail_canvas.delete('p3')
    w = self.LER_CHK_Tail_pil_image.width
    h = self.LER_CHK_Tail_pil_image.height
    self.LER_CHK_Tail_pil_image = self.LER_CHK_Tail_pil_image.resize((int(w * (self.LER_CHK_Tail_canvas_width / w)), 
                                                            int(h * (self.LER_CHK_Tail_canvas_height / h))))
    self.LER_CHK_Tail_plot = ImageTk.PhotoImage(image = self.LER_CHK_Tail_pil_image, master = self.LER_CHK_frame)
    # 画像を描画
    self.LER_CHK_Tail_canvas.create_image(5 + self.LER_CHK_Tail_canvas_width / 2, 5 + self.LER_CHK_Tail_canvas_height / 2, 
                                     image = self.LER_CHK_Tail_plot, tag = 'p3')
    
    # アブノーマルなレコードだったらフレームの色を赤に変更する変更
    Abnormal_Result_Tail_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Tail_WB.npy'
    Abnormal_Result_Tail_Text_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Tail_WB.txt'
    Abnormal_Result_Tail_List = Check_Tail_Abnormal_Result_File(Abnormal_Result_Tail_File_Name, 
                                                               Abnormal_Result_Tail_Text_File_Name)
    nab = 0
    for nn in range(len(Abnormal_Result_Tail_List)):
        if(Abnormal_Result_Tail_List[nn, 0] == Date_Range_DIF) and (Abnormal_Result_Tail_List[nn, 3] == Check_Record_Name):
            nab = 1
            break
    if(nab == 1):
        self.LER_CHK_Tail_canvas["highlightthickness"] = 3
        self.LER_CHK_Tail_canvas["highlightbackground"] = 'red'
    else:
        self.LER_CHK_Tail_canvas["highlightthickness"] = 0
        
    # LER_ABN_Trend
    # 画像ファイルがあるか
    plot_file_name = 'LER_Trend_' + Abort_Timing_k + '.png'
    path = plot_file_name
    is_file = os.path.isfile(path)
    if(is_file == False): # ファイルがなかったら
        self.LER_ABN_pil_image = Image.open("No_data.png")
    else:
        self.LER_ABN_pil_image = Image.open(plot_file_name)
        
    self.LER_ABN_canvas.delete('p4')
    w = self.LER_ABN_pil_image.width
    h = self.LER_ABN_pil_image.height
    self.LER_ABN_pil_image = self.LER_ABN_pil_image.resize((int(w * (self.LER_ABN_canvas_width / w)), 
                                                                int(h * (self.LER_ABN_canvas_height / h))))
    self.LER_ABN_plot = ImageTk.PhotoImage(image = self.LER_ABN_pil_image, master = self.LER_ABN_frame)
    # 画像を描画(中点x, 中点y, 画像)
    self.LER_ABN_canvas.create_image(5 + self.LER_ABN_canvas_width / 2, 5 + self.LER_ABN_canvas_height / 2, 
                                         image = self.LER_ABN_plot, tag = 'p4')
    
    # ビームがない時に、"No Beam"をグラフの右側に表示する。
    if(No_Beam == 1):
        self.LER_ABN_canvas.delete('t4')
        self.LER_ABN_nobeam_image = Image.open("No_beam.png")
        self.LER_ABN_nobeam_image = self.LER_ABN_nobeam_image.resize((200, 200))
        self.LER_ABN_nobeam_plot = ImageTk.PhotoImage(image = self.LER_ABN_nobeam_image, master = self.LER_ABN_frame)
        self.LER_ABN_canvas.create_image(500, 150, image = self.LER_ABN_nobeam_plot, tag = 't4')
        
        # self.LER_ABN_canvas.create_text(500, 150, text = 'No Beam', font=('Arial', 24, 'italic'), fill="gray", tag = 't4')
    
    # Tailがない時に、"No Tail"をグラフの右側に表示する。
    if(Cn_Beam == 1):
        self.LER_ABN_canvas.delete('t4')
        self.LER_ABN_notail_image = Image.open("No_tail.png")
        self.LER_ABN_notail_image = self.LER_ABN_notail_image.resize((200, 200))
        self.LER_ABN_notail_plot = ImageTk.PhotoImage(image = self.LER_ABN_notail_image, master = self.LER_ABN_frame)
        self.LER_ABN_canvas.create_image(500, 150, image = self.LER_ABN_notail_plot, tag = 't4')
        
        # self.LER_ABN_canvas.create_text(500, 150, text = 'No Tail', font=('Arial', 24, 'italic'), fill="gray", tag = 't4')
        
    # LERアボート時刻のリスト 
    path = 'LER_Abort_Time_List.txt'
    is_file = os.path.isfile(path)
    if(is_file == True): # ファイルがあったら
        content =[]
        with open (path, encoding = 'utf8', newline = '') as f:
            csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
            for row in csvreader:
                content = content + row #contentは1行のリストになる
            self.LER_Abort_Time_List_box = content
        self.LER_Abort_Time_listbox.delete(0, tk.END)
        for item in self.LER_Abort_Time_List_box:
            self.LER_Abort_Time_listbox.insert(tk.END, item)

    # ##########################################################################################################
    
    # Abnormalの原因を推定する
    Find_Possible_Cause(self, List_Para, Method, Cause_Record_Name_Strg, Cause_Record_Name_Tail, No_Beam, Cn_Beam)

    # ##########################################################################################################
    # Last_Refd+Ref_Pd+k日前のデータを削除する
    Deletedate_dt0 = datetime.datetime.strptime(Abort_Timing, '%Y-%m-%d %H:%M:%S')
    Day_advance0 = Last_Refd + Ref_Pd
    Last_Refd_i = int(Last_Refd)
    
    # Last_Refd+Ref_Pd日から1~4日 20240315変更
    for k in range(1, 5):
        day_advance =  -(Day_advance0 + k)
        Deletedate_dt = Deletedate_dt0 + datetime.timedelta(days = day_advance)
        Deletedate = '{:%Y%m%d}'.format(Deletedate_dt)
    
        path_list = glob.glob('LER*' + Deletedate + '*.txt')
        if(len(path_list) > 0):
            for n in path_list:
                command_d = 'rm ' + n
                try:
                    subprocess.run(command_d, shell = True)
                except Exception as e:
                    print("subprocess.check_call() failed", command_d)
                    print(str(e))
    
        path_list = glob.glob('LER*' + Deletedate + '*.npy')
        if(len(path_list) > 0):
            for n in path_list:
                command_d = 'rm ' + n
                try:
                    subprocess.run(command_d, shell = True)
                except Exception as e:
                    print("subprocess.check_call() failed", command_d)
                    print(str(e))
    
        path_list = glob.glob('LER*' + Deletedate + '*.npz')
        if(len(path_list) > 0):
            for n in path_list:
                command_d = 'rm ' + n
                try:
                    subprocess.run(command_d, shell = True)
                except Exception as e:
                    print("subprocess.check_call() failed", command_d)
                    print(str(e))
    
        print(Last_Refd + Ref_Pd + k, ' days old data were removed', len(path_list))
    
    # Last_Refd日から1~Last_Refd-2日 20240315変更
    for k in range(1, Last_Refd_i - 1):
        day_advance =  -(Last_Refd_i - k)
        Deletedate_dt = Deletedate_dt0 + datetime.timedelta(days = day_advance)
        Deletedate = '{:%Y%m%d}'.format(Deletedate_dt)
    
        path_list = glob.glob('LER*' + Deletedate + '*.txt')
        if(len(path_list) > 0):
            for n in path_list:
                command_d = 'rm ' + n
                try:
                    subprocess.run(command_d, shell = True)
                except Exception as e:
                    print("subprocess.check_call() failed", command_d)
                    print(str(e))
    
        path_list = glob.glob('LER*' + Deletedate + '*.npy')
        if(len(path_list) > 0):
            for n in path_list:
                command_d = 'rm ' + n
                try:
                    subprocess.run(command_d, shell = True)
                except Exception as e:
                    print("subprocess.check_call() failed", command_d)
                    print(str(e))
    
        path_list = glob.glob('LER*' + Deletedate + '*.npz')
        if(len(path_list) > 0):
            for n in path_list:
                command_d = 'rm ' + n
                try:
                    subprocess.run(command_d, shell = True)
                except Exception as e:
                    print("subprocess.check_call() failed", command_d)
                    print(str(e))
    
        print(Last_Refd_i -k, ' days old data were removed', len(path_list))

    # kekb-co-userに入り、Last_Refd + Ref_Pd + k日前のデータを消す
    Rmv_kekbcouser('LER', Last_Refd_i, Deletedate_dt0, Day_advance0)
    
    return

# ==============================================================================================================
# ==============================================================================================================

def Main_Command_HER(self):
    # 状態表示変更
    self.Status_change_running()
    self.HER_ABN_canvas["highlightthickness"] = 5
    self.HER_ABN_canvas["highlightbackground"] = 'lightgreen'
    self.HER_ABN_canvas.update()
    
    # エラーフラグリセット
    Error_f = 'none'
    
    # リングの名前
    Ring_Name = 'HER'
    List_Para = self.HER_CCG_List
    
    # 各側室のCCGの数+1
    HER_CCG_n = [0] * 12
    for k in range(12):
        HER_CCG_n[k] = len(self.HER_Record_Name_List_box[k])
        # print(k, HER_CCG_n[k])
        
    # abnormal, normalをfileに保存するかどうか LER、HER共通
    File_Save_Para = self.Save_var.get()
    
    # Abort Timing
    Abort_Timing = self.HER_Abort_Time_entry.get()
    
    try:
        date_dt = datetime.datetime.strptime(Abort_Timing, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(traceback.format_exc())
        print(str(e))
        Error_f = str(e)
        tk.messagebox.showinfo('Check HER Time format', Error_f)
        self.HER_ABN_canvas["highlightthickness"] = 0
        self.HER_ABN_canvas.update()
        return
    
    # ランモードの確認 auto かmanualか
    n = self.Mode_var.get()
    Run_Mode = self.Mode_rdo_text[n]
        
    # Auto時のモードの確認　Abort triggerかIntervalか
    m = self.Auto_Mode_var.get()
    Auto_Mode = self.Auto_Mode_rdo_text[m]
    
    # constモード(Triggerモード中のconstモード)
    Cns_Mode = self.HER_trg_cns
    
    # 時間間隔
    try:
        self.HER_Interval_h = self.Interval_entry.get()
        Hadv = float(self.HER_Interval_h)
    except Exception as e:
        print(traceback.format_exc())
        print(str(e))
        Error_f = str(e)
        tk.messagebox.showinfo('Check interval time format', Error_f)
        self.HER_ABN_canvas["highlightthickness"] = 0
        self.HER_ABN_canvas.update()
        return
    
    # Referenceデータの最終日
    try:
        self.HER_Last_Ref_d = self.Last_Ref_entry.get()
        Last_Refd = float(self.HER_Last_Ref_d)
    except Exception as e:
        print(traceback.format_exc())
        print(str(e))
        Error_f = str(e)
        tk.messagebox.showinfo('Check Last Ref Day format', Error_f)
        self.HER_ABN_canvas["highlightthickness"] = 0
        self.HER_ABN_canvas.update()
        return
    
    # Referenceデータの期間(日)
    try:
        self.HER_Ref_Period_d = self.Ref_Period_entry.get()
        Ref_Pd = float(self.HER_Ref_Period_d)
    except Exception as e:
        print(traceback.format_exc())
        print(str(e))
        Error_f = str(e)
        tk.messagebox.showinfo('Check Ref Period format', Error_f)
        self.HER_ABN_canvas["highlightthickness"] = 0
        self.HER_ABN_canvas.update()
        return
    
    # Auto_modeで走っているかどうか
    Arun = self.HER_auto_run
    
    # グラフでチェックするレコード名
    # noneならプロットしない
    Check_Record_Name = self.HER_Record_Name_entry.get()
        
    # アボートタイミングから、各モードのDate_Range(データの時間範囲)を決める
    # ビームがあるかどうかも調べる。1 ならビームなし(シャットダウン中とか)、0 ならビームあり
    No_Beam, Error_f = Define_Date_Range(Abort_Timing, List_Para, Run_Mode, Auto_Mode, Cns_Mode, Arun, Hadv, Last_Refd, Ref_Pd)
    # 20240202変更
    # STDデータ、DIFデータにアボートがあるか
    No_STD_Abort = 0
    # No_DIF_Abort = 0
    if(Error_f == 'No_STD_Abort'): # STDにアボートが無かったら
        No_STD_Abort = 1
    # elif(Error_f == 'No_DIF_Abort'): # DIFにアボートが無かったら
    #     No_DIF_Abort = 1
    elif(Error_f != 'none'):
        tk.messagebox.showinfo('HER STD Strg Error', Error_f)
        return
    
    if (No_Beam == 0): # ビームがあるなら
        self.HER_Beam_Mode_label["text"] = 'With Beam'
    else:
        self.HER_Beam_Mode_label["text"] = 'No Beam'
        
    # 選別する方法
    Method = 'FNN'  # FNN 回帰モデルの結果を使い自動で選別(Strgはi32)

    # Plot_Para = 1 で　Method = Manual ならならプロットで確認する(Manualの時)。
    Plot_Para = 0
    
    print(' ')
    print('Please wait for a while')
    
    # ##########################################################################################################
    # ビーム蓄積時(Strg)の解析(STDモード、スタンダードデータ) 
    
    if(No_Beam == 0): # ビームがあったら
        
        # 入射蓄積時(Strg)の解析(STDモード、スタンダードデータ)
        Date_Range_STD, Error_fss = Get_Fit_STD_Strg(List_Para, Method, Ref_Pd, HER_CCG_n)
    
        # エラーがあったら、
        if(Error_fss != 'none'):
            tk.messagebox.showinfo('HER STD Strg Error', Error_fss)
            self.HER_ABN_canvas["highlightthickness"] = 0
            self.HER_ABN_canvas.update()
            return
        # エラーが無かったら
        else:
            # STD_Strgのプロット(横軸ビーム電流)
            Mode_Para = 'STD_Strg'
            Make_Plot_Strg(Method, Check_Record_Name, Date_Range_STD, Mode_Para, List_Para, Abort_Timing)
            # プロットの保存
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_STD + '.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
            
            # STD_Strgのプロット(横軸時刻)
            Make_Plot_Strg_Time(Method, Check_Record_Name, Date_Range_STD, Mode_Para, List_Para, Abort_Timing)
            # プロットの保存
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_STD + '_Time.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
        
        # STDにアボートがあったら、
        if(No_STD_Abort == 0):
            # ビームアボート後(Tail)の解析(STDモード、スタンダードデータ)
            Date_Range_STD, Error_fst = Get_Fit_STD_Tail(List_Para, Method)
    
            # エラーがあったら、
            if(Error_fst != 'none'):
                tk.messagebox.showinfo('HER Error', Error_fst)
                return

            # エラーが無かったら、かつSTDにアボートがあったら
            else:
                # STD_Tailのプロット
                Mode_Para = 'STD_Tail'
                Make_Plot_Tail(Method, Check_Record_Name, Date_Range_STD, Mode_Para, List_Para, Abort_Timing)
                # プロットの保存
                fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_STD + '.png'
                if(Check_Record_Name != 'none'):
                    plt.savefig(fig_name)
                # plt.show()
                plt.clf()
                plt.close()
        
    else: #ビームが無かったら
        Date_Range_STD, Error_fss = Get_Fit_STD_Strg_NB(List_Para, Method, Ref_Pd, HER_CCG_n)
        
        # エラーがあったら、
        if(Error_fss != 'none'):
            tk.messagebox.showinfo('HER STD Strg Error', Error_fss)
            tk.messagebox.showinfo('HER STD Strg Error', 'May be no kblog data')
            return
        # エラーが無かったら
        else:
            # STD_Strgのプロット(横軸ビーム電流)
            Mode_Para = 'STD_Strg'
            Make_Plot_Strg_NB(Method, Check_Record_Name, Date_Range_STD, Mode_Para, List_Para, Abort_Timing)
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_STD + '.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
        
            # STD_Strgのプロット(横軸時刻)
            Make_Plot_Strg_Time(Method, Check_Record_Name, Date_Range_STD, Mode_Para, List_Para, Abort_Timing)
            # プロットの保存
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_STD + '_Time.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
    
    # ##########################################################################################################
    # アボート直前の蓄積中、アボート直後の圧力をチェックする(CHKモード)
    # 蓄積時とアボート直後続けてリングの各側室について直線のビーム蓄積中のデータを読み出し、
    # 各レコードについて最小二乗誤差法で回帰曲線を計算する。結果を辞書にして保存する。
    
    # 初期値
    Cn_Beam = 0
    Error_fct = 'none'
    Error_fcs = 'none'
    
    # ビームがあったら
    if(No_Beam == 0): 
        Mode_Para = 'CHK_Strg'
        Date_Range_CHK, Error_fcs = Get_Fit_CHK_Strg(Method, Check_Record_Name, Mode_Para, List_Para, Abort_Timing, HER_CCG_n)
        Date_Range_DIF = Date_Range_CHK
        
        # エラーがあったら、
        if(Error_fcs != 'none'):
            tk.messagebox.showinfo('HER CHK Strg Error', Error_fcs)
            # return
        # エラーが無かったら
        else:
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_CHK + '.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
            
            # STD_Strgのプロット(横軸時刻)
            Make_Plot_Strg_Time(Method, Check_Record_Name, Date_Range_CHK, Mode_Para, List_Para, Abort_Timing)
            # プロットの保存
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_CHK + '_Time.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
        
        # もしSTDにアボートがあったら
        if (No_STD_Abort == 0):
            Mode_Para = 'CHK_Tail'
            Date_Range_CHK, Error_fct = Get_Fit_CHK_Tail(Method, Check_Record_Name, Mode_Para, List_Para, Abort_Timing)
        
            # エラーがあったら、
            if(Error_fct != 'none') and (Error_fct != 'No Tail Data'):
                tk.messagebox.showinfo('HER CHK Tail Error', Error_fct)
                # return
            # ビームが有ってTail Dataが無ければ
            elif(Error_fct == 'No Tail Data'):
                Cn_Beam = 1
            # エラーが無かったら
            else:
                fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_CHK + '.png'
                if(Check_Record_Name != 'none'):
                    plt.savefig(fig_name)
                # plt.show()
                plt.clf()
                plt.close()
        
    else: # ビームが無かったら
        Mode_Para = 'CHK_Strg'
        Date_Range_CHK, Error_fcs = Get_Fit_CHK_Strg_NB(Method, Check_Record_Name, Mode_Para, List_Para, Abort_Timing, HER_CCG_n)
        Date_Range_DIF = Date_Range_CHK
        
        # エラーがあったら、
        if(Error_fcs != 'none'):
            tk.messagebox.showinfo('HER CHK Strg Error', Error_fcs)
            self.HER_ABN_canvas["highlightthickness"] = 0
            self.HER_ABN_canvas.update()
            # return
        # エラーが無かったら
        else:
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_CHK + '.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
            
            # STD_Strgのプロット(横軸時刻)
            Make_Plot_Strg_Time(Method, Check_Record_Name, Date_Range_CHK, Mode_Para, List_Para, Abort_Timing)
            # プロットの保存
            fig_name = Ring_Name + '_' + Mode_Para + '_' + Check_Record_Name + '_' + Date_Range_CHK + '_Time.png'
            if(Check_Record_Name != 'none'):
                plt.savefig(fig_name)
            # plt.show()
            plt.clf()
            plt.close()
    
    # ##########################################################################################################
    # アボート直前の蓄積中、アボート直後の圧力を調べ、
    # スタンダードの回帰曲線での予想とズレの大きなレコードをみつける(DIFモード)
    # 蓄積時(Strg)とアボート直後(Tail)を続けて調べる
    
    # クラス分け方法の確認 SDM か Keras か
    n = self.ANN_Method_var.get()
    Class_Method = self.ANN_Method_rdo_text[n]
    
    # ビームがあったら
    if(No_Beam == 0): 
        if(Error_fcs == 'none') and (Error_fct == 'none'):
            # リングの各側室についてMSE計算に用いるデータを取り出す
            # Date_Range_DIFをファイルに保存する
    
            Mode_Para = 'DIF_Strg'
            Date_Range_DIF, Error_fds = Get_DIF_Strg(List_Para, HER_CCG_n)

            # エラーがあったら、
            if(Error_fds != 'none'):
                tk.messagebox.showinfo('HER DIF Strg Error', Error_fds)
                # return
            # エラーが無かったら
            else:
                N_Rec_Strg = Find_Abnormal_Strg(Method, Mode_Para, List_Para, File_Save_Para, Plot_Para, Abort_Timing, Class_Method, Check_Record_Name)
                self.HER_N_Rec_Strg_label["text"] = str(N_Rec_Strg)

            # もしSTDにアボートがあったら
            if (No_STD_Abort == 0):
                Mode_Para = 'DIF_Tail'
                Date_Range_DIF, Error_fdt = Get_DIF_Tail(List_Para)
        
                # エラーがあったら、
                if(Error_fdt != 'none'):
                    tk.messagebox.showinfo('HER DIF Tail Error', Error_fdt)
                    # return
                # エラーが無かったら
                else:
                    N_Rec_Tail = Find_Abnormal_Tail(Method, Mode_Para, List_Para, File_Save_Para, Plot_Para, Abort_Timing, Class_Method, Check_Record_Name)
                    self.HER_N_Rec_Tail_label["text"] = str(N_Rec_Tail)
                    
        # 20240131変更
        #ビームが有って、かつ(Tail dataが無いか(Cn_Beam = 1)、あるいは、STDデータにアボートが無かったら(No_STD_Abort = 1))
        elif(Error_fcs == 'none') and ((Cn_Beam == 1) or (No_STD_Abort == 1)):
            # リングの各側室についてMSE計算に用いるデータを取り出す
            # Date_Range_DIFをファイルに保存する
    
            Mode_Para = 'DIF_Strg'
            Date_Range_DIF, Error_fds = Get_DIF_Strg(List_Para, HER_CCG_n)

            # エラーがあったら、
            if(Error_fds != 'none'):
                tk.messagebox.showinfo('HER DIF Strg Error', Error_fds)
                # return
            # エラーが無かったら
            else:
                N_Rec_Strg = Find_Abnormal_Strg(Method, Mode_Para, List_Para, File_Save_Para, Plot_Para, Abort_Timing, Class_Method, Check_Record_Name)
                self.HER_N_Rec_Strg_label["text"] = str(N_Rec_Strg)

    else: # ビームが無かったら
        if(Error_fcs == 'none'):
            Mode_Para = 'DIF_Strg'
            Date_Range_DIF, Error_fds = Get_DIF_Strg_NB(List_Para, HER_CCG_n)
        
            # エラーがあったら、
            if(Error_fds != 'none'):
                tk.messagebox.showinfo('HER Error', Error_fds)
                self.HER_ABN_canvas["highlightthickness"] = 0
                self.HER_ABN_canvas.update()
                return
            # エラーが無かったら
            else:
                N_Rec_Strg = Find_Abnormal_Strg_NB(Method, Mode_Para, List_Para, File_Save_Para, Plot_Para, Abort_Timing, Class_Method, Check_Record_Name)
                self.HER_N_Rec_Strg_label["text"] = str(N_Rec_Strg)
                
                N_Rec_Tail = Find_Abnormal_Tail_NB(Method, Mode_Para, List_Para, File_Save_Para, Plot_Para, Abort_Timing)
                self.HER_N_Rec_Tail_label["text"] = str(N_Rec_Tail)
    
    # ##########################################################################################################
    # Abnormalになったレコードの頻度のトレンドをプロットする。
    
    # プロットする点数(nplot+nfill前のfillから）# ----------------------------------------------------????????
    Nplot = 12
    # nfill分の計 # -------------------------------------------------------------------------------------????????
    Nfill = 8 
    # プロットするRedordの最大数 # -----------------------------------------------------------------------????????
    Nrecord = 5 
    
    # プロットし、abnormalの数でcanvasの枠の色を変える
    abmax, Cause_Record_Name_Strg, Cause_Record_Name_Tail = Record_Freq_plot(self, List_Para, Method, Nplot, Nfill, 
                                                                         Nrecord, No_Beam, Cn_Beam)
    
    Abort_Timing_d = datetime.datetime.strptime(Abort_Timing, '%Y-%m-%d %H:%M:%S')
    Abort_Timing_k = '{:%Y%m%d%H%M%S}'.format(Abort_Timing_d) # 書式変更
    
    if(abmax == -1):
        print('no abnormal or normal list')
        self.HER_ABN_canvas["highlightthickness"] = 0
    else:
        if(abmax > 6 ):
            self.HER_ABN_canvas["highlightthickness"] = 5
            self.HER_ABN_canvas["highlightbackground"] = 'red'
        elif(abmax > 4 and abmax <= 6):
            self.HER_ABN_canvas["highlightthickness"] = 5
            self.HER_ABN_canvas["highlightbackground"] = 'hotpink'
        else:
            self.HER_ABN_canvas["highlightthickness"] = 0
    
        fig_name = Ring_Name + '_Trend_' + Abort_Timing_k + '.png'
        plt.savefig(fig_name, dpi = 400)
        # plt.show()
        plt.clf()
        plt.close()
    
    # ##########################################################################################################
    # プロット、Abort_Timingリストをアップデートする。
        
    # HER_STD_Strg(P vS I)
    # 画像ファイルがあるか
    plot_file_name = 'HER_STD_Strg_' + Check_Record_Name + '_' + Date_Range_STD + '.png'
    path = plot_file_name
    is_file = os.path.isfile(path)
    if(is_file == False): # ファイルがなかったら
        self.HER_STD_Strg_I_pil_image = Image.open("No_data.png")
    else:
        self.HER_STD_Strg_I_pil_image = Image.open(plot_file_name)
    
    self.HER_STD_Strg_I_canvas.delete('p1')
    w = self.HER_STD_Strg_I_pil_image.width
    h = self.HER_STD_Strg_I_pil_image.height
    self.HER_STD_Strg_I_pil_image = self.HER_STD_Strg_I_pil_image.resize((int(w * (self.HER_STD_Strg_I_canvas_width / w)), 
                                                                int(h * (self.HER_STD_Strg_I_canvas_height / h))))
    self.HER_STD_Strg_I_plot = ImageTk.PhotoImage(image = self.HER_STD_Strg_I_pil_image, master = self.HER_STD_frame)
    # 画像を描画
    self.HER_STD_Strg_I_canvas.create_image(5 + self.HER_STD_Strg_I_canvas_width / 2, 5 + self.HER_STD_Strg_I_canvas_height / 2, 
                                       image = self.HER_STD_Strg_I_plot, tag = 'p1')
        
    # ラベル変更
    Date_Range_STD_1 = Convert_Kblogrd_to_Dtime(Date_Range_STD[:14])
    Date_Range_STD_1d = '{:%Y-%m-%d}'.format(Date_Range_STD_1)
    Date_Range_STD_2 = Convert_Kblogrd_to_Dtime(Date_Range_STD[15:])
    Date_Range_STD_2d = '{:%Y-%m-%d}'.format(Date_Range_STD_2)
    self.HER_STD_AT_label["text"] = 'Date range  = ' + Date_Range_STD_1d + ' - ' + Date_Range_STD_2d
    self.HER_STD_RC_label["text"] = 'Record name = ' + Check_Record_Name
            
    # HER_STD_Strg(P vs T)
    # 画像ファイルがあるか
    plot_file_name = 'HER_STD_Strg_' + Check_Record_Name + '_' + Date_Range_STD + '_Time.png'
    path = plot_file_name
    is_file = os.path.isfile(path)
    if(is_file == False): # ファイルがなかったら
        self.HER_STD_Strg_T_pil_image = Image.open("No_data.png")
    else:
        self.HER_STD_Strg_T_pil_image = Image.open(plot_file_name)
        
    self.HER_STD_Strg_T_canvas.delete('p2')
    w = self.HER_STD_Strg_T_pil_image.width
    h = self.HER_STD_Strg_T_pil_image.height
    self.HER_STD_Strg_T_pil_image = self.HER_STD_Strg_T_pil_image.resize((int(w * (self.HER_STD_Strg_T_canvas_width / w)), 
                                                                int(h * (self.HER_STD_Strg_T_canvas_height / h))))
    self.HER_STD_Strg_T_plot = ImageTk.PhotoImage(image = self.HER_STD_Strg_T_pil_image, master = self.HER_STD_frame)
    # 画像を描画
    self.HER_STD_Strg_T_canvas.create_image(5 + self.HER_STD_Strg_T_canvas_width / 2, 5 + self.HER_STD_Strg_T_canvas_height / 2, 
                                       image = self.HER_STD_Strg_T_plot, tag = 'p2')
    
    # HER_STD_Tail
    # 画像ファイルがあるか
    plot_file_name = 'HER_STD_Tail_' + Check_Record_Name + '_' + Date_Range_STD + '.png'
    path = plot_file_name
    is_file = os.path.isfile(path)
    if(is_file == False): # ファイルがなかったら
        self.HER_STD_Tail_pil_image = Image.open("No_data.png")
    else:
        self.HER_STD_Tail_pil_image = Image.open(plot_file_name)
   
    self.HER_STD_Tail_canvas.delete('p3')
    w = self.HER_STD_Tail_pil_image.width
    h = self.HER_STD_Tail_pil_image.height
    self.HER_STD_Tail_pil_image = self.HER_STD_Tail_pil_image.resize((int(w * (self.HER_STD_Tail_canvas_width / w)), 
                                                            int(h * (self.HER_STD_Tail_canvas_height / h))))
    self.HER_STD_Tail_plot = ImageTk.PhotoImage(image = self.HER_STD_Tail_pil_image, master = self.HER_STD_frame)
    # 画像を描画(中点x, 中点y, 画像)
    self.HER_STD_Tail_canvas.create_image(5 + self.HER_STD_Tail_canvas_width / 2, 5 + self.HER_STD_Tail_canvas_height / 2, 
                                     image = self.HER_STD_Tail_plot, tag = 'p3')    
    
    # HER_CHK_Strg(P vS I)
    # 画像ファイルがあるか
    plot_file_name = 'HER_CHK_Strg_' + Check_Record_Name + '_' + Date_Range_CHK + '.png'
    path = plot_file_name
    is_file = os.path.isfile(path)
    if(is_file == False): # ファイルがなかったら
        self.HER_CHK_Strg_I_pil_image = Image.open("No_data.png")
    else:
        self.HER_CHK_Strg_I_pil_image = Image.open(plot_file_name)
    
    self.HER_CHK_Strg_I_canvas.delete('p1')
    w = self.HER_CHK_Strg_I_pil_image.width
    h = self.HER_CHK_Strg_I_pil_image.height
    self.HER_CHK_Strg_I_pil_image = self.HER_CHK_Strg_I_pil_image.resize((int(w * (self.HER_CHK_Strg_I_canvas_width / w)), 
                                                                int(h * (self.HER_CHK_Strg_I_canvas_height / h))))
    self.HER_CHK_Strg_I_plot = ImageTk.PhotoImage(image = self.HER_CHK_Strg_I_pil_image, master = self.HER_CHK_frame)
    # 画像を描画(中点x, 中点y, 画像)
    self.HER_CHK_Strg_I_canvas.create_image(5 + self.HER_CHK_Strg_I_canvas_width / 2, 5 + self.HER_CHK_Strg_I_canvas_height / 2, 
                                       image = self.HER_CHK_Strg_I_plot, tag = 'p1')
    
    # ラベル変更
    Date_Range_DIF_1 = Convert_Kblogrd_to_Dtime(Date_Range_DIF[:14])
    Date_Range_DIF_1d = '{:%m-%d %H:%M}'.format(Date_Range_DIF_1)
    Date_Range_DIF_2 = Convert_Kblogrd_to_Dtime(Date_Range_DIF[15:])
    Date_Range_DIF_2d = '{:%m-%d %H:%M}'.format(Date_Range_DIF_2)
    self.HER_CHK_AT_label["text"] = 'Date range  = ' + Date_Range_DIF_1d + ' - ' +  Date_Range_DIF_2d
    self.HER_CHK_RC_label["text"] = 'Record name = ' + Check_Record_Name
        
    # アブノーマルなレコードだったらフレームの色を赤に変更する変更
    if(No_Beam == 0): # ビームがあったら
        Abnormal_Result_Strg_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_WB.npy'
        Abnormal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_WB.txt'
        Abnormal_Result_Strg_List = Check_Strg_Abnormal_Result_File(Abnormal_Result_Strg_File_Name, 
                                                                   Abnormal_Result_Strg_Text_File_Name)
    if(No_Beam == 1): # ビームが無かったら
        Abnormal_Result_Strg_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_NB.npy'
        Abnormal_Result_Strg_Text_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Strg_NB.txt'
        Abnormal_Result_Strg_List = Check_Strg_Abnormal_Result_File(Abnormal_Result_Strg_File_Name, 
                                                               Abnormal_Result_Strg_Text_File_Name)
    nab = 0
    for nn in range(len(Abnormal_Result_Strg_List)):
        if(Abnormal_Result_Strg_List[nn, 0] == Date_Range_DIF) and (Abnormal_Result_Strg_List[nn, 3] == Check_Record_Name):
            nab = 1
            break
    if(nab == 1):
        self.HER_CHK_Strg_I_canvas["highlightthickness"] = 3
        self.HER_CHK_Strg_I_canvas["highlightbackground"] = 'red'
    else:
        self.HER_CHK_Strg_I_canvas["highlightthickness"] = 0
        
    # HER_CHK_Strg(P vs T)
    # 画像ファイルがあるか
    plot_file_name = 'HER_CHK_Strg_' + Check_Record_Name + '_' + Date_Range_CHK + '_Time.png'
    path = plot_file_name
    is_file = os.path.isfile(path)
    if(is_file == False): # ファイルがなかったら
        self.HER_CHK_Strg_T_pil_image = Image.open("No_data.png")
    else:
        self.HER_CHK_Strg_T_pil_image = Image.open(plot_file_name)
        
    self.HER_CHK_Strg_T_canvas.delete('p2')
    w = self.HER_CHK_Strg_T_pil_image.width
    h = self.HER_CHK_Strg_T_pil_image.height
    self.HER_CHK_Strg_T_pil_image = self.HER_CHK_Strg_T_pil_image.resize((int(w * (self.HER_CHK_Strg_T_canvas_width / w)), 
                                                                int(h * (self.HER_CHK_Strg_T_canvas_height / h))))
    self.HER_CHK_Strg_T_plot = ImageTk.PhotoImage(image = self.HER_CHK_Strg_T_pil_image, master = self.HER_CHK_frame)
    # 画像を描画(中点x, 中点y, 画像)
    self.HER_CHK_Strg_T_canvas.create_image(5 + self.HER_CHK_Strg_T_canvas_width / 2, 5 + self.HER_CHK_Strg_T_canvas_height / 2, 
                                       image = self.HER_CHK_Strg_T_plot, tag = 'p2')
    
    # HER_CHK_Tail
    # 画像ファイルがあるか
    plot_file_name = 'HER_CHK_Tail_' + Check_Record_Name + '_' + Date_Range_CHK + '.png'
    path = plot_file_name
    is_file = os.path.isfile(path)
    if(is_file == False): # ファイルがなかったら
        self.HER_CHK_Tail_pil_image = Image.open("No_data.png")
    else:
        self.HER_CHK_Tail_pil_image = Image.open(plot_file_name)
   
    self.HER_CHK_Tail_canvas.delete('p3')
    w = self.HER_CHK_Tail_pil_image.width
    h = self.HER_CHK_Tail_pil_image.height
    self.HER_CHK_Tail_pil_image = self.HER_CHK_Tail_pil_image.resize((int(w * (self.HER_CHK_Tail_canvas_width / w)), 
                                                            int(h * (self.HER_CHK_Tail_canvas_height / h))))
    self.HER_CHK_Tail_plot = ImageTk.PhotoImage(image = self.HER_CHK_Tail_pil_image, master = self.HER_CHK_frame)
    # 画像を描画(中点x, 中点y, 画像)
    self.HER_CHK_Tail_canvas.create_image(5 + self.HER_CHK_Tail_canvas_width / 2, 5 + self.HER_CHK_Tail_canvas_height / 2, 
                                     image = self.HER_CHK_Tail_plot, tag = 'p3')
        
    # アブノーマルなレコードだったらフレームの色を赤に変更する変更
    Abnormal_Result_Tail_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Tail_WB.npy'
    Abnormal_Result_Tail_Text_File_Name = Ring_Name + '_' + Method +'_Abnormal_Class2_Result_Tail_WB.txt'
    Abnormal_Result_Tail_List = Check_Tail_Abnormal_Result_File(Abnormal_Result_Tail_File_Name, 
                                                               Abnormal_Result_Tail_Text_File_Name)
    nab = 0
    for nn in range(len(Abnormal_Result_Tail_List)):
        if(Abnormal_Result_Tail_List[nn, 0] == Date_Range_DIF) and (Abnormal_Result_Tail_List[nn, 3] == Check_Record_Name):
            nab = 1
            break
    if(nab == 1):
        self.HER_CHK_Tail_canvas["highlightthickness"] = 3
        self.HER_CHK_Tail_canvas["highlightbackground"] = 'red'
    else:
        self.HER_CHK_Tail_canvas["highlightthickness"] = 0
            
    # HER_ABN_Trend
    # 画像ファイルがあるか
    plot_file_name = 'HER_Trend_' + Abort_Timing_k + '.png'
    path = plot_file_name
    is_file = os.path.isfile(path)
    if(is_file == False): # ファイルがなかったら
        self.HER_ABN_pil_image = Image.open("No_data.png")
    else:
        self.HER_ABN_pil_image = Image.open(plot_file_name)
        
    self.HER_ABN_canvas.delete('p4')
    w = self.HER_ABN_pil_image.width
    h = self.HER_ABN_pil_image.height
    self.HER_ABN_pil_image = self.HER_ABN_pil_image.resize((int(w * (self.HER_ABN_canvas_width / w)), 
                                                                int(h * (self.HER_ABN_canvas_height / h))))
    self.HER_ABN_plot = ImageTk.PhotoImage(image = self.HER_ABN_pil_image, master = self.HER_ABN_frame)
    # 画像を描画(中点x, 中点y, 画像)
    self.HER_ABN_canvas.create_image(5 + self.HER_ABN_canvas_width / 2, 5 + self.HER_ABN_canvas_height / 2, 
                                         image = self.HER_ABN_plot, tag = 'p4')

    # ビームがない時に、"No Beam"をグラフの右側に表示する。
    if(No_Beam == 1):
        self.HER_ABN_canvas.delete('t4')
        self.HER_ABN_nobeam_image = Image.open("No_beam.png")
        self.HER_ABN_nobeam_image = self.HER_ABN_nobeam_image.resize((200, 200))
        self.HER_ABN_nobeam_plot = ImageTk.PhotoImage(image = self.HER_ABN_nobeam_image, master = self.HER_ABN_frame)
        self.HER_ABN_canvas.create_image(500, 150, image = self.HER_ABN_nobeam_plot, tag = 't4')
        
        # self.HER_ABN_canvas.create_text(500, 150, text = 'No Beam', font=('Arial', 24, 'italic'), fill="gray", tag = 't4')
    
    # Tailがない時に、"No Tail"をグラフの右側に表示する。
    if(Cn_Beam == 1):
        self.HER_ABN_canvas.delete('t4')
        self.HER_ABN_notail_image = Image.open("No_tail.png")
        self.HER_ABN_notail_image = self.HER_ABN_notail_image.resize((200, 200))
        self.HER_ABN_notail_plot = ImageTk.PhotoImage(image = self.HER_ABN_notail_image, master = self.HER_ABN_frame)
        self.HER_ABN_canvas.create_image(500, 150, image = self.HER_ABN_notail_plot, tag = 't4')
        
        # self.HER_ABN_canvas.create_text(500, 150, text = 'No Tail', font=('Arial', 24, 'italic'), fill="gray", tag = 't4')
        
    # HERアボート時刻のリスト 
    path = 'HER_Abort_Time_List.txt'
    is_file = os.path.isfile(path)
    if(is_file == True): # ファイルがあったら
        content =[]
        with open (path, encoding = 'utf8', newline = '') as f:
            csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
            for row in csvreader:
                content = content + row #contentは1行のリストになる
            self.HER_Abort_Time_List_box = content
        self.HER_Abort_Time_listbox.delete(0, tk.END)
        for item in self.HER_Abort_Time_List_box:
            self.HER_Abort_Time_listbox.insert(tk.END, item)
    
    # ##########################################################################################################
    
    # Abnormalの原因を推定する
    Find_Possible_Cause(self, List_Para, Method, Cause_Record_Name_Strg, Cause_Record_Name_Tail, No_Beam, Cn_Beam)

    # ##########################################################################################################
    
    # Last_Refd+Ref_Pd+k日前のデータを削除する
    Deletedate_dt0 = datetime.datetime.strptime(Abort_Timing, '%Y-%m-%d %H:%M:%S')
    Day_advance0 = Last_Refd + Ref_Pd
    Last_Refd_i = int(Last_Refd)
    
    # Last_Refd+Ref_Pd日から1~4日 20240315変更
    for k in range(1, 5):
        day_advance =  -(Day_advance0 + k)
        Deletedate_dt = Deletedate_dt0 + datetime.timedelta(days = day_advance)
        Deletedate = '{:%Y%m%d}'.format(Deletedate_dt)
    
        path_list = glob.glob('HER*' + Deletedate + '*.txt')
        if(len(path_list) > 0):
            for n in path_list:
                command_d = 'rm ' + n
                try:
                    subprocess.run(command_d, shell = True)
                except Exception as e:
                    print("subprocess.check_call() failed", command_d)
                    print(str(e))
    
        path_list = glob.glob('HER*' + Deletedate + '*.npy')
        if(len(path_list) > 0):
            for n in path_list:
                command_d = 'rm ' + n
                try:
                    subprocess.run(command_d, shell = True)
                except Exception as e:
                    print("subprocess.check_call() failed", command_d)
                    print(str(e))
    
        path_list = glob.glob('HER*' + Deletedate + '*.npz')
        if(len(path_list) > 0):
            for n in path_list:
                command_d = 'rm ' + n
                try:
                    subprocess.run(command_d, shell = True)
                except Exception as e:
                    print("subprocess.check_call() failed", command_d)
                    print(str(e))
    
        print(Last_Refd + Ref_Pd + k, ' days old data were removed', len(path_list))
    
    # Last_Refd日から1~Last_Refd-2日 20240315変更
    for k in range(1, Last_Refd_i-1):
        day_advance =  -(Last_Refd_i - k)
        Deletedate_dt = Deletedate_dt0 + datetime.timedelta(days = day_advance)
        Deletedate = '{:%Y%m%d}'.format(Deletedate_dt)
    
        path_list = glob.glob('HER*' + Deletedate + '*.txt')
        if(len(path_list) > 0):
            for n in path_list:
                command_d = 'rm ' + n
                try:
                    subprocess.run(command_d, shell = True)
                except Exception as e:
                    print("subprocess.check_call() failed", command_d)
                    print(str(e))
    
        path_list = glob.glob('HER*' + Deletedate + '*.npy')
        if(len(path_list) > 0):
            for n in path_list:
                command_d = 'rm ' + n
                try:
                    subprocess.run(command_d, shell = True)
                except Exception as e:
                    print("subprocess.check_call() failed", command_d)
                    print(str(e))
    
        path_list = glob.glob('HER*' + Deletedate + '*.npz')
        if(len(path_list) > 0):
            for n in path_list:
                command_d = 'rm ' + n
                try:
                    subprocess.run(command_d, shell = True)
                except Exception as e:
                    print("subprocess.check_call() failed", command_d)
                    print(str(e))
    
        print(Last_Refd_i - k, ' days old data were removed', len(path_list))
    
    # kekb-co-userに入り、Last_Refd+ Ref_Pd + k日前のデータを消す
    Rmv_kekbcouser('HER', Last_Refd_i, Deletedate_dt0, Day_advance0)
    
    return

# ==============================================================================================================
###########################################################################################################   ==
##        メインループ             ########################################################################   ==
###########################################################################################################   ==
# ==============================================================================================================

class Application(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
    
        # 各リング、各側室のリストの定義：共通 koblogrdのコマンドで使う
        self.LER_CCG_List = ['LERD01CCG', 'LERD02CCG', 'LERD03CCG', 'LERD04CCG', 'LERD05CCG', 'LERD06CCG', 
                            'LERD07CCG', 'LERD08CCG', 'LERD09CCG', 'LERD10CCG', 'LERD11CCG', 'LERD12CCG']
        self.HER_CCG_List = ['HERD01CCG', 'HERD02CCG', 'HERD03CCG', 'HERD04CCG', 'HERD05CCG', 'HERD06CCG',
                            'HERD07CCG', 'HERD08CCG', 'HERD09CCG', 'HERD10CCG', 'HERD11CCG', 'HERD12CCG']
    
        # LERのCCGレコード名リストを作る
        self.LER_Record_Name_List_box = [[], [], [], [], [], [], [], [], [], [], [], []]
        self.LER_Record_Name_List_box = Make_LER_Record(self.LER_Record_Name_List_box)
        
        # LERのCCGレコード名の場所のリストを作る
        self.LER_Record_Name_Place_List_box = [[], [], [], [], [], [], [], [], [], [], [], []]
        self.LER_Record_Name_Place_List_box = Make_LER_Record_Place(self.LER_Record_Name_Place_List_box)
        
        # HERのCCGレコード名リストを作る
        self.HER_Record_Name_List_box = [[], [], [], [], [], [], [], [], [], [], [], []]
        self.HER_Record_Name_List_box = Make_HER_Record(self.HER_Record_Name_List_box)
        
        # HERのCCGレコード名の場所のリストを作る
        self.HER_Record_Name_Place_List_box = [[], [], [], [], [], [], [], [], [], [], [], []]
        self.HER_Record_Name_Place_List_box = Make_HER_Record_Place(self.HER_Record_Name_Place_List_box)
    
        # LERアボート時刻のリスト
        path = 'LER_Abort_Time_List.txt'
        is_file = os.path.isfile(path)
        if(is_file == True):
            content =[]
            with open (path, encoding = 'utf8', newline = '') as f:
                csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
                for row in csvreader:
                    content = content + row #contentは1行のリストになる
            self.LER_Abort_Time_List_box = content
        else:
            self.LER_Abort_Time_List_box = ['none']
    
        # HERアボート時刻のリスト
        path = 'HER_Abort_Time_List.txt'
        is_file = os.path.isfile(path)
        if(is_file == True):
            content =[]
            with open (path, encoding = 'utf8', newline = '') as f:
                csvreader = csv.reader(f, delimiter = ',') # csvreaderにリストとして読み込む。コンマ区切り。
                for row in csvreader:
                    content = content + row #contentは1行のリストになる
            self.HER_Abort_Time_List_box = content
        else:
            self.HER_Abort_Time_List_box = ['none']
        
        # 20240204追加
        # 初期値
        self.LER_id_first = 0
        self.HER_id_first = 0
        
        # 20240108追加
        # CCGの説明リストボックス初期値
        self.LER_CCG_Strg_List_box = ['none']
        self.LER_CCG_Tail_List_box = ['none']
        self.HER_CCG_Strg_List_box = ['none']
        self.HER_CCG_Tail_List_box = ['none']
        
        # 20240108追加
        # Abnormalの原因のリスト初期値
        self.LER_ABN_Cause_Strg_List = [['none','none'], ['none', 'none'], ['none', 'none'], ['none', 'none'], ['none', 'none']]
        self.LER_ABN_Cause_Tail_List = [['none','none'], ['none', 'none'], ['none', 'none'], ['none', 'none'], ['none', 'none']]
        self.HER_ABN_Cause_Strg_List = [['none','none'], ['none', 'none'], ['none', 'none'], ['none', 'none'], ['none', 'none']]
        self.HER_ABN_Cause_Tail_List = [['none','none'], ['none', 'none'], ['none', 'none'], ['none', 'none'], ['none', 'none']]
        
        # TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK 
        # TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK TK 
    
        # rootクラス作成
        # root = tk.Tk()
        self.master.title("Pressure Anomality Detection Panel")
        self.master.geometry('1740x960')
        self.master.resizable(width=False, height=False)

        # C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
        # C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C

        # Control_Frame作成 
        self.Control_frame = tk.Frame(master, width = 150, height = 910, pady=10, padx=10, relief=tk.RIDGE, bd = 2)
        self.Control_frame.place(x = 10, y = 40 )
        
        # radio buttonのフォントセット
        font_general1 = ('Arial', '9', 'normal')
        font_general2 = ('Arial', '8', 'normal')

        # 時刻のラベル
        Nowtime = datetime.datetime.now()
        self.Nowtime_str = Nowtime.strftime('%Y-%m-%d %H:%M:%S')
        self.time_label = tk.Label(self.Control_frame, text = self.Nowtime_str[2:], font = ('Times new roman', 8), fg = 'green')
        self.time_label.place(x = 2, y = 5)
        
        self.time_label.after(1000 * 10, self.time_update)
        
        # Mode(AutoあるいはManual)を選択するラジオボタン
        # ラベル
        self.Mode_label = tk.Label(self.Control_frame, text = 'Run mode', font = ('Times new roman', 10))
        self.Mode_label.place(x = 3, y = 30)
        # チェック有無変数
        self.Mode_var = tk.IntVar()
        # value = 0のラジオボタンにチェックを入れる
        self.Mode_var.set(1)
        # Modeのリスト
        self.Mode_rdo_text = ['Auto', 'Manual']
        # ラジオボタン作成
        self.Mode_rdo_1 = tk.Radiobutton(self.Control_frame, value = 0, variable = self.Mode_var, text = self.Mode_rdo_text[0],
                                         font = ('Times new roman', 11), fg = 'blue')
        self.Mode_rdo_1.place(x = 1, y = 50)
        self.Mode_rdo_2 = tk.Radiobutton(self.Control_frame, value = 1, variable = self.Mode_var, text = self.Mode_rdo_text[1],
                                         font = ('Times new roman', 11), fg = 'blue')
        self.Mode_rdo_2.place(x = 1, y = 425)

        # 最初はManual
        self.LER_auto_run = 0
        self.HER_auto_run = 0
        
        # Auto時のタイミングを選択するラジオボタン
        # ラベル
        self.Auto_Mode_label = tk.Label(self.Control_frame, text = 'Timing', font = ('Times new roman', 10))
        self.Auto_Mode_label.place(x = 20, y = 75)
        # チェック有無変数
        self.Auto_Mode_var = tk.IntVar()
        # value = 0のラジオボタンにチェックを入れる
        self.Auto_Mode_var.set(0)
        # Ringのリスト
        self.Auto_Mode_rdo_text = ['Abt Trg', 'Cns Intr']
        # ラジオボタン作成
        self.Auto_Mode_rdo_1 = tk.Radiobutton(self.Control_frame, value = 0, variable = self.Auto_Mode_var, 
                                              text = self.Auto_Mode_rdo_text[0], font = font_general1)
        self.Auto_Mode_rdo_1.place(x = 10, y = 95)
        self.Auto_Mode_rdo_2 = tk.Radiobutton(self.Control_frame, value = 1, variable = self.Auto_Mode_var, 
                                              text = self.Auto_Mode_rdo_text[1], font = font_general1)
        self.Auto_Mode_rdo_2.place(x = 10, y = 115)
        
        # Intervalの時間を入力するエントリー
        # ラベル
        self.Interval_entry_label = tk.Label(self.Control_frame, text = 'h', font = ('Times new roman', 10))
        self.Interval_entry_label.place(x = 58, y = 140)
        # entry
        self.Interval_entry = tk.Entry(self.Control_frame, width = 3)
        # entryに値を挿入
        self.Interval_entry.insert(0, 6)
        self.Interval_entry.place(x = 25, y = 140)

        # Auto stopラベル
        self.Auto_stop_label = tk.Label(self.Control_frame, text = 'Auto stop', font = ('Times new roman', 10))
        self.Auto_stop_label.place(x = 20, y = 160)
        
        # Auto stopボタン
        self.Auto_stop_btn = tk.Button(self.Control_frame, text = 'Stop', bg = 'light green', command = self.stop_time) 
        self.Auto_stop_btn.place(x = 15, y = 180)

        # ログのリストボックス
        self.loglist = tk.Listbox(self.Control_frame, height = 15, width = 18, font=("Lucida Console", 7))
        self.loglist.place(x = 15, y = 210)
        
        # Ring(LERあるいはHER)を選択するラジオボタン
        # ラベル
        self.Ring_label = tk.Label(self.Control_frame, text = 'Ring', font = ('Times new roman', 10))
        self.Ring_label.place(x = 20, y = 445)
        # チェック有無変数
        self.Ring_var = tk.IntVar()
        # value = 0のラジオボタンにチェックを入れる
        self.Ring_var.set(0)
        # Ringのリスト
        self.Ring_rdo_text = ['LER', 'HER']
        # ラジオボタン作成
        self.Ring_rdo_1 = tk.Radiobutton(self.Control_frame, value = 0, variable = self.Ring_var, 
                                         text = self.Ring_rdo_text[0], font = font_general1)
        self.Ring_rdo_1.place(x = 10, y = 465)
        self.Ring_rdo_2 = tk.Radiobutton(self.Control_frame, value = 1, variable = self.Ring_var, 
                                         text = self.Ring_rdo_text[1], font = font_general1)
        self.Ring_rdo_2.place(x = 10, y = 485)

        # リストにAbnormal、Normalを保存するかどうかを選択するラジオボタン
        # ラベル
        self.Save_label = tk.Label(self.Control_frame, text = 'Save result', font = ('Times new roman', 10))
        self.Save_label.place(x = 3, y = 513)
        # チェック有無変数
        self.Save_var = tk.IntVar()
        # value = 0のラジオボタンにチェックを入れる
        self.Save_var.set(0)
        # 保存するかどうかのリスト
        self.Save_rdo_text = ['No save', 'Save']
        # ラジオボタン作成
        self.Save_rdo_1 = tk.Radiobutton(self.Control_frame, value = 0, variable = self.Save_var, 
                                         text = self.Save_rdo_text[0], font = font_general1)
        self.Save_rdo_1.place(x = 10, y = 533)
        self.Save_rdo_2 = tk.Radiobutton(self.Control_frame, value = 1, variable = self.Save_var, 
                                         text = self.Save_rdo_text[1], font = font_general1)
        self.Save_rdo_2.place(x = 10, y = 553)

        # Last reference dayの日時を入力するエントリー
        # ラベル
        self.Last_Ref_label = tk.Label(self.Control_frame, text = 'Last ref day', font = ('Times new roman', 10))
        self.Last_Ref_label.place(x = 3, y = 581)
        # ラベル
        self.Last_Ref_entry_label = tk.Label(self.Control_frame, text = 'days ago', font = ('Times new roman', 10))
        self.Last_Ref_entry_label.place(x = 57, y = 601)
        # entry
        self.Last_Ref_entry = tk.Entry(self.Control_frame, width = 3)
        # entryに値を挿入
        self.Last_Ref_entry.insert(0, 5)
        self.Last_Ref_entry.place(x = 25, y = 601)
        
        # Referenceの期間を入力するエントリー
        # ラベル
        self.Ref_Period_label = tk.Label(self.Control_frame, text = 'Ref period', font = ('Times new roman', 10))
        self.Ref_Period_label.place(x = 3, y = 624)
        # ラベル
        self.Ref_Period_entry_label = tk.Label(self.Control_frame, text = 'days', font = ('Times new roman', 10))
        self.Ref_Period_entry_label.place(x = 57, y = 644)
        # entry
        self.Ref_Period_entry = tk.Entry(self.Control_frame, width = 3)
        # entryに値を挿入
        self.Ref_Period_entry.insert(0, 3)
        self.Ref_Period_entry.place(x = 25, y = 644)
        
        # 解析方法を選択するラジオボタン
        # ラベル
        self.ANN_Method_label = tk.Label(self.Control_frame, text = 'Classifying method', font = ('Times new roman', 10))
        self.ANN_Method_label.place(x = 3, y = 675)
        # チェック有無変数
        self.ANN_Method_var = tk.IntVar()
        # value = 0のラジオボタンにチェックを入れる
        self.ANN_Method_var.set(0)
        # Methodのリスト
        self.ANN_Method_rdo_text = ['SDM', 'Keras']
        # ラジオボタン作成
        self.ANN_Method_rdo_1 = tk.Radiobutton(self.Control_frame, value = 0, variable = self.ANN_Method_var, 
                                         text = self.ANN_Method_rdo_text[0], font = ('Times new roman', 10))
        self.ANN_Method_rdo_1.place(x = 5, y = 695)
        self.ANN_Method_rdo_2 = tk.Radiobutton(self.Control_frame, value = 1, variable = self.ANN_Method_var, 
                                         text = self.ANN_Method_rdo_text[1], font = ('Times new roman', 10))
        self.ANN_Method_rdo_2.place(x = 60, y = 695)
        
        # Runスタートラベル
        # self.Run_Start_label= tk.Label(self.Control_frame, text = 'Run Start', fg = 'black', justify = tk.LEFT, 
        #                              font = ('Times new roman', 10))
        # self.Run_Start_label.place(x = 2, y = 710)
        
        # Runスタートボタン
        self.Run_Start_btn = tk.Button(self.Control_frame, text = 'Start', bg = 'lightblue', command = self.Start_btn_click)
        self.Run_Start_btn.place(x = 2, y =730)
        
        # 状態のラベル
        self.Status_label = tk.Label(self.Control_frame, text = 'Waiting', fg = 'blue', justify = tk.LEFT, 
                                     font = ('Times new roman', 10))      
        self.Status_label.place(x = 10, y = 765)
        
        # Auto状態のラベル
        self.Status_LER_Auto_label = tk.Label(self.Control_frame, text = 'LER Auto stop', fg = 'blue', justify = tk.LEFT, 
                                          font = ('Times new roman', 10))      
        self.Status_LER_Auto_label.place(x = 10, y = 800)
        
        self.Status_HER_Auto_label = tk.Label(self.Control_frame, text = 'HER Auto stop', fg = 'blue', justify = tk.LEFT, 
                                          font = ('Times new roman', 10))      
        self.Status_HER_Auto_label.place(x = 10, y = 820)
        
        # ウィンドウを閉じるボタン
        self.Win_Close_btn = tk.Button(self.Control_frame, text = 'Close', bg = 'orange', command = self.Close_btn_click)
        self.Win_Close_btn.place(x = 2, y = 860)
        

        # L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L
        # L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L L

        # LER Abort時刻とRecord Nameのフレーム(LER_Selection)作成 
        self.LER_Selection_frame = tk.Frame(master, width = 170, height = 450, pady=10, padx=10, relief=tk.RIDGE, bd = 2)
        self.LER_Selection_frame.place(x = 860, y = 40 )

        # LERのアボート時刻を選択するエントリー、リストボックス ------------------------------------------------------------
        # ラベル
        self.LER_Abort_Time_label = tk.Label(self.LER_Selection_frame, text = 'LER Check/Abort Time', font = ('Times new roman', 9))
        self.LER_Abort_Time_label.place(x = 3, y = 2)

        # エントリー
        self.LER_Abort_Time_entry = tk.Entry(self.LER_Selection_frame, width = 22, font=("Arial", 7))
        # entryに値を挿入
        Now_dt = datetime.datetime.now()
        Now = Now_dt.strftime('%Y-%m-%d %H:%M:%S')
        self.LER_Abort_Time_entry.insert(0, Now)
        self.LER_Abort_Time_entry.place(x = 3, y = 25)

        # リストボックスの選択肢
        self.LER_Abort_Time_var = tk.StringVar(value = self.LER_Abort_Time_List_box)
        self.LER_Abort_Time_listbox = tk.Listbox(self.LER_Selection_frame, listvariable = self.LER_Abort_Time_var, height = 7,
                                                 font=("Arial", 7))
        # 項目選択時にself.show_LER_selectedを実行
        self.LER_Abort_Time_listbox.bind('<<ListboxSelect>>', self.show_LER_selected)

        # スクロールバーの作成
        self.LER_Abort_Time_scrollbar = ttk.Scrollbar(self.LER_Selection_frame, orient = tk.VERTICAL, 
                                                      command = self.LER_Abort_Time_listbox.yview)
        # スクロールバーをリストボックスに反映
        self.LER_Abort_Time_listbox['yscrollcommand'] = self.LER_Abort_Time_scrollbar.set

        self.LER_Abort_Time_listbox.place(x = 3, y = 48, width = 125, height = 120)
        self.LER_Abort_Time_scrollbar.place(x = 128, y = 48, height = 120)

        # アボートリストのリセットボタン
        self.LER_Abort_Reset_btn = tk.Button(self.LER_Selection_frame, text = 'Abt Time/Data Reset', bg = 'lightgray', 
                                             font = font_general2, command = self.LER_abt_reset) 
        self.LER_Abort_Reset_btn.place(x = 3, y = 172)
        
        # LERのレコード名を選択するエントリー、ラジオボタン、リストボックス ------------------------------------------------------------
        # 基準配置ｘ座標
        LRx = 0
        # ラベル
        self.LER_Record_Name_label = tk.Label(self.LER_Selection_frame, text = 'LER Record Name', font = ('Times new roman', 10))
        self.LER_Record_Name_label.place(x = LRx + 3, y = 210)
        # エントリー
        self.LER_Record_Name_entry = tk.Entry(self.LER_Selection_frame, width = 22, font=("Arial", 7))
        # entryに値を挿入
        self.LER_Record_Name_entry.insert(0, 'none')
        self.LER_Record_Name_entry.place(x = LRx + 3, y = 230)

        # 側室を選ぶラジオボタン
        # チェック有無変数
        self.LER_LC_var = tk.IntVar()
        # value = 0のラジオボタンにチェックを入れる
        self.LER_LC_var.set(0)
        # LCのリスト
        self.LER_LC_rdo_text = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        # ラジオボタン作成
        for i in range(0, 12):
            self.LER_LC_rdo = tk.Radiobutton(self.LER_Selection_frame, value = i, variable = self.LER_LC_var, font = font_general2,
                                                text = self.LER_LC_rdo_text[i], command = self.change_LER_LC_List)
            if(i < 4):
                self.LER_LC_rdo.place(x = LRx - 2 + 37 * i, y = 247)
            elif(i > 3 and i < 8):
                self.LER_LC_rdo.place(x = LRx - 2 + 37 * (i - 4), y = 262)
            else:
                self.LER_LC_rdo.place(x = LRx - 2 + 37 * (i - 8), y = 277)
        
        # アボートタイミングのリストボックスの選択肢
        self.LER_Record_Name_var = tk.StringVar(value = self.LER_Record_Name_List_box[0])
        self.LER_Record_Name_listbox = tk.Listbox(self.LER_Selection_frame, listvariable = self.LER_Record_Name_var, height = 7,
                                                 font=("Arial", 7))
        # 項目選択時にshow_ler_record_selectedを実行
        self.LER_Record_Name_listbox.bind('<<ListboxSelect>>', self.show_ler_record_selected)

        # スクロールバーの作成
        self.LER_Record_Name_scrollbar = ttk.Scrollbar(self.LER_Selection_frame, orient = tk.VERTICAL, 
                                                       command = self.LER_Record_Name_listbox.yview)
        # スクロールバーをリストボックスに反映
        self.LER_Record_Name_listbox['yscrollcommand'] = self.LER_Record_Name_scrollbar.set

        self.LER_Record_Name_listbox.place(x = LRx + 3, y = 300, width = 125, height = 125)
        self.LER_Record_Name_scrollbar.place(x = LRx + 128, y =300, height = 125)

        # LERのSTDプロットを描くフレーム作成(キャンバス)------------------------------------------------------------
        # 基準配置ｘ座標
        LSx = 1390
        # フレーム
        self.LER_STD_frame = tk.Frame(master, width = 340, height = 450, pady=10, padx=10, relief=tk.RIDGE, bd = 2)
        self.LER_STD_frame.place(x = LSx, y = 40 )
        # ラベル
        self.LER_STD_label = tk.Label(self.LER_STD_frame, text = 'LER Reference', font = ('Times new roman', 10))
        self.LER_STD_label.place(x = 5, y = 0)

        # アボートタイミングラベル
        self.LER_STD_AT_label = tk.Label(self.LER_STD_frame, text = 'Date range  = ', fg = 'green', font = ('Arial', 9))
        self.LER_STD_AT_label.place(x = 48, y = 17)
        
        # レコード名ラベル
        self.LER_STD_RC_label = tk.Label(self.LER_STD_frame, text = 'Record name = ', fg = 'green', font = ('Arial', 9))
        self.LER_STD_RC_label.place(x = 48, y = 34)
        
        # Strg(P vs I)
        # ラベル
        self.LER_STD_Strg_label = tk.Label(self.LER_STD_frame, text = 'Storage', font = ('Times new roman', 10))
        self.LER_STD_Strg_label.place(x = 5, y =58)
        
        # キャンバスのサイズ
        self.LER_STD_Strg_I_canvas_width = 145
        self.LER_STD_Strg_I_canvas_height = 150
        # キャンバス
        self.LER_STD_Strg_I_canvas = tk.Canvas(self.LER_STD_frame, width = self.LER_STD_Strg_I_canvas_width, 
                                               height = self.LER_STD_Strg_I_canvas_height, relief='solid', bg = 'white', bd=1)
        self.LER_STD_Strg_I_canvas.place(x = 2, y = 75)
        # 画像を用意
        # 画像をリサイズする
        self.LER_STD_Strg_I_pil_image = Image.open("No_data.png")
        w = self.LER_STD_Strg_I_pil_image.width
        h = self.LER_STD_Strg_I_pil_image.height
        self.LER_STD_Strg_I_pil_image = self.LER_STD_Strg_I_pil_image.resize((int(w * (self.LER_STD_Strg_I_canvas_height / h)), 
                                                                int(h * (self.LER_STD_Strg_I_canvas_height / h))))
        self.LER_STD_Strg_I_plot = ImageTk.PhotoImage(image = self.LER_STD_Strg_I_pil_image, master = self.LER_STD_frame)
        # 画像を描画(中点x, 中点y, 画像)
        self.LER_STD_Strg_I_canvas.create_image(5 + self.LER_STD_Strg_I_canvas_width / 2, 5 + self.LER_STD_Strg_I_canvas_height / 2, 
                                       image = self.LER_STD_Strg_I_plot, tag = 'p1')

        # Strg(P vs T)
        # ラベル
        self.LER_STD_T_label = tk.Label(self.LER_STD_frame, text = 'Trend', font = ('Times new roman', 10))
        self.LER_STD_T_label.place(x = 165, y = 58)
        
        # キャンバスのサイズ
        self.LER_STD_Strg_T_canvas_width = 155
        self.LER_STD_Strg_T_canvas_height = 150
        # キャンバス
        self.LER_STD_Strg_T_canvas = tk.Canvas(self.LER_STD_frame, width = self.LER_STD_Strg_T_canvas_width, 
                                               height = self.LER_STD_Strg_T_canvas_height, relief='solid', bg = 'white', bd=1)
        self.LER_STD_Strg_T_canvas.place(x = 160, y = 75)
        # 画像を用意
        # 画像をリサイズする
        self.LER_STD_Strg_T_pil_image = Image.open("No_data.png")
        w = self.LER_STD_Strg_T_pil_image.width
        h = self.LER_STD_Strg_T_pil_image.height
        self.LER_STD_Strg_T_pil_image = self.LER_STD_Strg_T_pil_image.resize((int(w * (self.LER_STD_Strg_T_canvas_width / w)), 
                                                                int(h * (self.LER_STD_Strg_T_canvas_width / w))))
        self.LER_STD_Strg_T_plot = ImageTk.PhotoImage(image = self.LER_STD_Strg_T_pil_image, master = self.LER_STD_frame)
        # 画像を描画(中点x, 中点y, 画像)
        self.LER_STD_Strg_T_canvas.create_image(5 + self.LER_STD_Strg_T_canvas_width / 2, 5 + self.LER_STD_Strg_T_canvas_height / 2, 
                                       image = self.LER_STD_Strg_T_plot, tag = 'p2')

        # Tail
        # ラベル
        self.LER_STD_Tail_label = tk.Label(self.LER_STD_frame, text = 'Tail', font = ('Times new roman', 10))
        self.LER_STD_Tail_label.place(x = 5, y = 245)
        
        # キャンバスのサイズ
        self.LER_STD_Tail_canvas_width = 310
        self.LER_STD_Tail_canvas_height = 150
        # キャンバス
        self.LER_STD_Tail_canvas = tk.Canvas(self.LER_STD_frame, width = self.LER_STD_Tail_canvas_width, 
                                             height = self.LER_STD_Tail_canvas_height, relief='solid', bg = 'white', bd=1)
        self.LER_STD_Tail_canvas.place(x = 2, y = 265)
        # 画像を用意
        # 画像をリサイズする
        self.LER_STD_Tail_pil_image = Image.open("No_data.png")
        w = self.LER_STD_Tail_pil_image.width
        h = self.LER_STD_Tail_pil_image.height
        self.LER_STD_Tail_pil_image = self.LER_STD_Tail_pil_image.resize((int(w * (self.LER_STD_Tail_canvas_height / h)), 
                                                            int(h * (self.LER_STD_Tail_canvas_height / h))))
        self.LER_STD_Tail_plot = ImageTk.PhotoImage(image = self.LER_STD_Tail_pil_image, master = self.LER_STD_frame)
        # 画像を描画(中点x, 中点y, 画像)
        self.LER_STD_Tail_canvas.create_image(5 + self.LER_STD_Tail_canvas_width / 2, 5 + self.LER_STD_Tail_canvas_height / 2, 
                                     image = self.LER_STD_Tail_plot, tag = 'p3')

        # LERのCHKプロットを描くフレーム作成(キャンバス)------------------------------------------------------------
        # 基準配置ｘ座標
        LSx = 1040
        # フレーム
        self.LER_CHK_frame = tk.Frame(master, width = 340, height = 450, pady=10, padx=10, relief=tk.RIDGE, bd = 2)
        self.LER_CHK_frame.place(x = LSx, y = 40 )
        # ラベル
        self.LER_CHK_label = tk.Label(self.LER_CHK_frame, text = 'LER Check', font = ('Times new roman', 10))
        self.LER_CHK_label.place(x = 5, y = 0)

        # アボートタイミングラベル
        self.LER_CHK_AT_label = tk.Label(self.LER_CHK_frame, text = 'Date range  = ', fg = 'green', font = ('Arial', 9))
        self.LER_CHK_AT_label.place(x = 48, y = 17)
        
        # レコード名ラベル
        self.LER_CHK_RC_label = tk.Label(self.LER_CHK_frame, text = 'Record name = ', fg = 'green', font = ('Arial', 9))
        self.LER_CHK_RC_label.place(x = 48, y = 34)
        
        # Strg(P vs I)
        # ラベル
        self.LER_CHK_Strg_label = tk.Label(self.LER_CHK_frame, text = 'Storage', font = ('Times new roman', 10))
        self.LER_CHK_Strg_label.place(x = 5, y = 53)
        
        # キャンバスのサイズ
        self.LER_CHK_Strg_I_canvas_width = 145
        self.LER_CHK_Strg_I_canvas_height = 143
        # キャンバス
        self.LER_CHK_Strg_I_canvas = tk.Canvas(self.LER_CHK_frame, width = self.LER_CHK_Strg_I_canvas_width, 
                                               height = self.LER_CHK_Strg_I_canvas_height, relief='solid', bg = 'white', bd = 1)
        self.LER_CHK_Strg_I_canvas.place(x = 2, y = 70)
        # 画像を用意
        # 画像をリサイズする
        self.LER_CHK_Strg_I_pil_image = Image.open("No_data.png")
        w = self.LER_CHK_Strg_I_pil_image.width
        h = self.LER_CHK_Strg_I_pil_image.height
        self.LER_CHK_Strg_I_pil_image = self.LER_CHK_Strg_I_pil_image.resize((int(w * (self.LER_CHK_Strg_I_canvas_height / h)), 
                                                                int(h * (self.LER_CHK_Strg_I_canvas_height / h))))
        self.LER_CHK_Strg_I_plot = ImageTk.PhotoImage(image = self.LER_CHK_Strg_I_pil_image, master = self.LER_CHK_frame)
        # 画像を描画(中点x, 中点y, 画像)
        self.LER_CHK_Strg_I_canvas.create_image(5 + self.LER_CHK_Strg_I_canvas_width / 2, 5 + self.LER_CHK_Strg_I_canvas_height / 2, 
                                       image = self.LER_CHK_Strg_I_plot, tag = 'p1')

        # LER Normal or Abnormalとしてrecord、dataをManualのファイルに保存するボタン
        self.LER_Save_Manual_Strg_btn = tk.Button(self.LER_CHK_frame, text = 'Save Strg record as', bg = 'lightgrey', 
                                             font = font_general2, command = self.LER_Save_Manual_Strg) 
        self.LER_Save_Manual_Strg_btn.place(x = 42, y = 220)
        
        # NormalかAbnormalを選択するラジオボタン
        # チェック有無変数
        self.LER_Save_Manual_Strg_Mode_var = tk.IntVar()
        # value = 0のラジオボタンにチェックを入れる
        self.LER_Save_Manual_Strg_Mode_var.set(0)
        # リスト
        self.LER_Save_Manual_Strg_Mode_rdo_text = ['Nor', 'Abn']
        # ラジオボタン作成
        self.LER_Save_Manual_Strg_Mode_rdo_1 = tk.Radiobutton(self.LER_CHK_frame, value = 0, 
                                                              variable = self.LER_Save_Manual_Strg_Mode_var, 
                                                              text = self.LER_Save_Manual_Strg_Mode_rdo_text[0], 
                                                              font = font_general1)
        self.LER_Save_Manual_Strg_Mode_rdo_1.place(x = 183, y = 225)
        self.LER_Save_Manual_Strg_Mode_rdo_2 = tk.Radiobutton(self.LER_CHK_frame, value = 1, 
                                                              variable = self.LER_Save_Manual_Strg_Mode_var, 
                                                              text = self.LER_Save_Manual_Strg_Mode_rdo_text[1], 
                                                              font = font_general1)
        self.LER_Save_Manual_Strg_Mode_rdo_2.place(x = 230, y = 225)
        
        # Strg(P vs T)
        # ラベル
        self.LER_CHK_T_label = tk.Label(self.LER_CHK_frame, text = 'Trend', font = ('Times new roman', 10))
        self.LER_CHK_T_label.place(x = 165, y = 53)
        
        # キャンバスのサイズ
        self.LER_CHK_Strg_T_canvas_width = 155
        self.LER_CHK_Strg_T_canvas_height = 143
        # キャンバス
        self.LER_CHK_Strg_T_canvas = tk.Canvas(self.LER_CHK_frame, width = self.LER_CHK_Strg_T_canvas_width, 
                                               height = self.LER_CHK_Strg_T_canvas_height, relief='solid', bg = 'white', bd=1)
        self.LER_CHK_Strg_T_canvas.place(x = 160, y = 70)
        # 画像を用意
        # 画像をリサイズする
        self.LER_CHK_Strg_T_pil_image = Image.open("No_data.png")
        w = self.LER_CHK_Strg_T_pil_image.width
        h = self.LER_CHK_Strg_T_pil_image.height
        self.LER_CHK_Strg_T_pil_image = self.LER_CHK_Strg_T_pil_image.resize((int(w * (self.LER_CHK_Strg_T_canvas_width / w)), 
                                                                int(h * (self.LER_CHK_Strg_T_canvas_width / w))))
        self.LER_CHK_Strg_T_plot = ImageTk.PhotoImage(image = self.LER_CHK_Strg_T_pil_image, master = self.LER_CHK_frame)
        # 画像を描画(中点x, 中点y, 画像)
        self.LER_CHK_Strg_T_canvas.create_image(5 + self.LER_CHK_Strg_T_canvas_width / 2, 5 + self.LER_CHK_Strg_T_canvas_height / 2, 
                                       image = self.LER_CHK_Strg_T_plot, tag = 'p2')

        # Tail
        # ラベル
        self.LER_CHK_Tail_label = tk.Label(self.LER_CHK_frame, text = 'Tail', font = ('Times new roman', 10))
        self.LER_CHK_Tail_label.place(x = 5, y = 242)
        
        # キャンバスのサイズ
        self.LER_CHK_Tail_canvas_width = 310
        self.LER_CHK_Tail_canvas_height = 140
        # キャンバス
        self.LER_CHK_Tail_canvas = tk.Canvas(self.LER_CHK_frame, width = self.LER_CHK_Tail_canvas_width, 
                                             height = self.LER_CHK_Tail_canvas_height, relief='solid', bg = 'white', bd=1)
        self.LER_CHK_Tail_canvas.place(x = 2, y = 260)
        # 画像を用意
        # 画像をリサイズする
        self.LER_CHK_Tail_pil_image = Image.open("No_data.png")
        w = self.LER_CHK_Tail_pil_image.width
        h = self.LER_CHK_Tail_pil_image.height
        self.LER_CHK_Tail_pil_image = self.LER_CHK_Tail_pil_image.resize((int(w * (self.LER_CHK_Tail_canvas_height / h)), 
                                                            int(h * (self.LER_CHK_Tail_canvas_height / h))))
        self.LER_CHK_Tail_plot = ImageTk.PhotoImage(image = self.LER_CHK_Tail_pil_image, master = self.LER_CHK_frame)
        # 画像を描画(中点x, 中点y, 画像)
        self.LER_CHK_Tail_canvas.create_image(5 + self.LER_CHK_Tail_canvas_width / 2, 5 + self.LER_CHK_Tail_canvas_height / 2, 
                                     image = self.LER_CHK_Tail_plot, tag = 'p3')

        # LER Normal or Abnormalとしてrecord、dataをManualのファイルに保存するボタン(Tail)
        self.LER_Save_Manual_Tail_btn = tk.Button(self.LER_CHK_frame, text = 'Save Tail record as', bg = 'lightgray', 
                                             font = font_general2, command = self.LER_Save_Manual_Tail) 
        self.LER_Save_Manual_Tail_btn.place(x = 42, y = 407)
        
        # NormalかAbnormalを選択するラジオボタン
        # チェック有無変数
        self.LER_Save_Manual_Tail_Mode_var = tk.IntVar()
        # value = 0のラジオボタンにチェックを入れる
        self.LER_Save_Manual_Tail_Mode_var.set(0)
        # リスト
        self.LER_Save_Manual_Tail_Mode_rdo_text = ['Nor', 'Abn']
        # ラジオボタン作成
        self.LER_Save_Manual_Tail_Mode_rdo_1 = tk.Radiobutton(self.LER_CHK_frame, value = 0, 
                                                              variable = self.LER_Save_Manual_Tail_Mode_var, 
                                                              text = self.LER_Save_Manual_Tail_Mode_rdo_text[0], font = font_general1)
        self.LER_Save_Manual_Tail_Mode_rdo_1.place(x = 183, y = 413)
        self.LER_Save_Manual_Tail_Mode_rdo_2 = tk.Radiobutton(self.LER_CHK_frame, value = 1, 
                                                              variable = self.LER_Save_Manual_Tail_Mode_var, 
                                                              text = self.LER_Save_Manual_Tail_Mode_rdo_text[1], font = font_general1)
        self.LER_Save_Manual_Tail_Mode_rdo_2.place(x = 230, y = 413)
        
        # LERのABNトレンドプロットを描くフレーム作成(キャンバス)------------------------------------------------------------
        # 基準配置ｘ座標
        LSx = 170
        # フレーム
        self.LER_ABN_frame = tk.Frame(master, width = 680, height = 450, pady=10, padx=10, relief=tk.RIDGE, bd = 2)
        self.LER_ABN_frame.place(x = LSx, y = 40 )
        
        # ラベル
        self.LER_Beam_Mode_label = tk.Label(self.LER_ABN_frame, text = 'Not defined', font = ('Times new roman', 10), fg = 'green')
        self.LER_Beam_Mode_label.place(x = 580, y = 0)

        # 推定原因のラベル
        self.LER_ABN_Cause_Strg_label = tk.Label(self.LER_ABN_frame, text = 'Possible cause (Strg)', font = ('Arial', 9),
                                                relief = tk.SOLID, bd = 0, width = 43, bg = 'lightgrey')
        self.LER_ABN_Cause_Strg_label.place(x = 40, y = 20)
        self.LER_ABN_Cause_Tail_label = tk.Label(self.LER_ABN_frame, text = 'Possible cause (Tail)', font = ('Arial', 9),
                                                relief = tk.SOLID, bd = 0, width = 43, bg = 'lightgrey')
        self.LER_ABN_Cause_Tail_label.place(x = 350, y = 20)
        
        # トレンドグラフのラベル
        self.LER_ABN_label = tk.Label(self.LER_ABN_frame, text = 'LER Abnormal Record Trend', font = ('Times new roman', 10))
        self.LER_ABN_label.place(x = 5, y = 0)
        # キャンバスのサイズ
        self.LER_ABN_canvas_width = 650
        self.LER_ABN_canvas_height = 300
        # キャンバス
        self.LER_ABN_canvas = tk.Canvas(self.LER_ABN_frame, width = self.LER_ABN_canvas_width, height = self.LER_ABN_canvas_height, 
                               relief='solid', bg = 'white', bd=1)
        self.LER_ABN_canvas.place(x = 3, y = 45)
        # 画像を用意
        # 画像をリサイズする
        self.LER_ABN_pil_image = Image.open("No_data.png")
        w = self.LER_ABN_pil_image.width
        h = self.LER_ABN_pil_image.height
        self.LER_ABN_pil_image = self.LER_ABN_pil_image.resize((int(w * (self.LER_ABN_canvas_width / w)), 
                                                                int(h * (self.LER_ABN_canvas_width / w))))
        self.LER_ABN_plot = ImageTk.PhotoImage(image = self.LER_ABN_pil_image, master = self.LER_ABN_frame)
        # 画像を描画(中点x, 中点y, 画像)
        self.LER_ABN_canvas.create_image(5 + self.LER_ABN_canvas_width / 2, 5 + self.LER_ABN_canvas_height / 2, 
                                         image = self.LER_ABN_plot, tag = 'p4')
        
        # 20240108変更
        # CCGの説明リストボックス
        # self.LER_CCG_Strg_list = tk.Listbox(self.LER_ABN_frame, height = 5, width = 52, font=("Arial", 7))
        # self.LER_CCG_Strg_list.place(x = 5, y =355)
        
        # self.LER_CCG_Tail_list = tk.Listbox(self.LER_ABN_frame, height = 5, width = 52, font=("Arial", 7))
        # self.LER_CCG_Tail_list.place(x = 341, y = 355)

        # CCGの説明リストボックスの選択肢
        self.LER_CCG_Strg_var = tk.StringVar(value = self.LER_CCG_Strg_List_box)
        self.LER_CCG_Strg_listbox = tk.Listbox(self.LER_ABN_frame, listvariable = self.LER_CCG_Strg_var, font=("Arial", 7))
        self.LER_CCG_Strg_listbox.bind('<<ListboxSelect>>', self.change_LER_Strg_Cause)
        self.LER_CCG_Strg_listbox.place(x = 5, y =355, width = 300, height = 74)
        
        self.LER_CCG_Tail_var = tk.StringVar(value = self.LER_CCG_Tail_List_box)
        self.LER_CCG_Tail_listbox = tk.Listbox(self.LER_ABN_frame, listvariable = self.LER_CCG_Tail_var, font=("Arial", 7))
        self.LER_CCG_Tail_listbox.bind('<<ListboxSelect>>', self.change_LER_Tail_Cause)
        self.LER_CCG_Tail_listbox.place(x = 341, y =355, width = 300, height = 74)
        
        # Abormal recordの数のラベル
        self.LER_N_Rec_Strg_label = tk.Label(self.LER_ABN_frame, text = '00', font = ('Arial', 9))
        self.LER_N_Rec_Strg_label.place(x = 307, y = 410)
        
        self.LER_N_Rec_Tail_label = tk.Label(self.LER_ABN_frame, text = '00', font = ('Arial', 9))
        self.LER_N_Rec_Tail_label.place(x = 643, y = 410)
        
        # H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H 
        # H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H H 

        # HER Abort時刻とRecord Nameのフレーム(HER_Selection)作成 
        self.HER_Selection_frame = tk.Frame(master, width = 170, height = 450, pady=10, padx=10, relief=tk.RIDGE, bd = 2)
        self.HER_Selection_frame.place(x = 860, y = 500 )

        # HERのアボート時刻を選択するエントリー、リストボックス ------------------------------------------------------------
        # ラベル
        self.HER_Abort_Time_label = tk.Label(self.HER_Selection_frame, text = 'HER Check/Abort Time', font = ('Times new roman', 9))
        self.HER_Abort_Time_label.place(x = 3, y = 2)

        # エントリー
        self.HER_Abort_Time_entry = tk.Entry(self.HER_Selection_frame, width = 22, font=("Arial", 7))
        # entryに値を挿入
        Now_dt = datetime.datetime.now()
        Now = Now_dt.strftime('%Y-%m-%d %H:%M:%S')
        self.HER_Abort_Time_entry.insert(0, Now)
        self.HER_Abort_Time_entry.place(x = 3, y = 25)

        # アボートタイミングのリストボックスの選択肢
        self.HER_Abort_Time_var = tk.StringVar(value = self.HER_Abort_Time_List_box)
        self.HER_Abort_Time_listbox = tk.Listbox(self.HER_Selection_frame, listvariable = self.HER_Abort_Time_var, height = 7,
                                                font=("Arial", 7))
        # 項目選択時にshow_HER_selectedを実行
        self.HER_Abort_Time_listbox.bind('<<ListboxSelect>>', self.show_HER_selected)

        # スクロールバーの作成
        self.HER_Abort_Time_scrollbar = ttk.Scrollbar(self.HER_Selection_frame, orient = tk.VERTICAL, 
                                                      command = self.HER_Abort_Time_listbox.yview)
        # スクロールバーをリストボックスに反映
        self.HER_Abort_Time_listbox['yscrollcommand'] = self.HER_Abort_Time_scrollbar.set

        self.HER_Abort_Time_listbox.place(x = 3, y = 48, width = 125, height = 120)
        self.HER_Abort_Time_scrollbar.place(x = 128, y = 48, height = 120)

        # アボートリストのリセットボタン
        self.HER_Abort_Reset_btn = tk.Button(self.HER_Selection_frame, text = 'Abt Time/Data Reset', bg = 'lightgray', 
                                             font = font_general2, command = self.HER_abt_reset) 
        self.HER_Abort_Reset_btn.place(x = 3, y = 172)
        
        # HERのレコード名を選択するエントリー、リストボックス ------------------------------------------------------------
        # 基準配置ｘ座標
        LRx = 0
        # ラベル
        self.HER_Record_Name_label = tk.Label(self.HER_Selection_frame, text = 'HER Record Name', font = ('Times new roman', 10))
        self.HER_Record_Name_label.place(x = LRx + 3, y = 210)

        # エントリー
        self.HER_Record_Name_entry = tk.Entry(self.HER_Selection_frame, width = 22, font=("Arial", 7))
        # entryに値を挿入
        self.HER_Record_Name_entry.insert(0, 'none')
        self.HER_Record_Name_entry.place(x = LRx + 3, y = 230)

        # 側室を選ぶラジオボタン
        # チェック有無変数
        self.HER_LC_var = tk.IntVar()
        # value = 0のラジオボタンにチェックを入れる
        self.HER_LC_var.set(0)
        # LCのリスト
        self.HER_LC_rdo_text = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        # ラジオボタン作成
        for i in range(0, 12):
            self.HER_LC_rdo = tk.Radiobutton(self.HER_Selection_frame, value = i, variable = self.HER_LC_var, font = font_general2,
                                                text = self.HER_LC_rdo_text[i], command = self.change_HER_LC_List)
            if(i < 4):
                self.HER_LC_rdo.place(x = LRx - 2 + 37 * i, y = 247)
            elif(i > 3 and i < 8):
                self.HER_LC_rdo.place(x = LRx - 2 + 37 * (i - 4), y = 262)
            else:
                self.HER_LC_rdo.place(x = LRx - 2 + 37 * (i - 8), y = 277)
                
        # リストボックスの選択肢
        self.HER_Record_Name_var = tk.StringVar(value = self.HER_Record_Name_List_box[0])
        self.HER_Record_Name_listbox = tk.Listbox(self.HER_Selection_frame, listvariable = self.HER_Record_Name_var, height = 7,
                                                 font=("Arial", 7))
        # 項目選択時にshow_HER_record_selectedを実行
        self.HER_Record_Name_listbox.bind('<<ListboxSelect>>', self.show_her_record_selected)

        # スクロールバーの作成
        self.HER_Record_Name_scrollbar = ttk.Scrollbar(self.HER_Selection_frame, orient = tk.VERTICAL, 
                                                       command = self.HER_Record_Name_listbox.yview)
        # スクロールバーをリストボックスに反映
        self.HER_Record_Name_listbox['yscrollcommand'] = self.HER_Record_Name_scrollbar.set

        self.HER_Record_Name_listbox.place(x = LRx + 3, y = 300, width = 125, height = 125)
        self.HER_Record_Name_scrollbar.place(x = LRx + 128, y = 300, height = 125)

        # HERのSTDプロットを描くフレーム作成(キャンバス)------------------------------------------------------------
        # 基準配置ｘ座標
        LSx = 1390
        # フレーム
        self.HER_STD_frame = tk.Frame(master, width = 340, height = 450, pady=10, padx=10, relief=tk.RIDGE, bd = 2)
        self.HER_STD_frame.place(x = LSx, y = 500 )
        # ラベル
        self.HER_STD_label = tk.Label(self.HER_STD_frame, text = 'HER Reference', font = ('Times new roman', 10))
        self.HER_STD_label.place(x = 5, y = 0)

        # アボートタイミングラベル
        self.HER_STD_AT_label = tk.Label(self.HER_STD_frame, text = 'Date range  = ', fg = 'green', font = ('Arial', 9))
        self.HER_STD_AT_label.place(x = 48, y = 17)
        
        # レコード名ラベル
        self.HER_STD_RC_label = tk.Label(self.HER_STD_frame, text = 'Record name = ', fg = 'green', font = ('Arial', 9))
        self.HER_STD_RC_label.place(x = 48, y = 34)
        
        # Strg(P vs I)
        # ラベル
        self.HER_STD_Strg_label = tk.Label(self.HER_STD_frame, text = 'Storage', font = ('Times new roman', 10))
        self.HER_STD_Strg_label.place(x = 5, y = 58)
        
        # キャンバスのサイズ
        self.HER_STD_Strg_I_canvas_width = 145
        self.HER_STD_Strg_I_canvas_height = 150
        # キャンバス
        self.HER_STD_Strg_I_canvas = tk.Canvas(self.HER_STD_frame, width = self.HER_STD_Strg_I_canvas_width, 
                                               height = self.HER_STD_Strg_I_canvas_height, relief='solid',bg = 'white', bd=1)
        self.HER_STD_Strg_I_canvas.place(x = 2, y = 75)
        # 画像を用意
        # 画像をリサイズする
        self.HER_STD_Strg_I_pil_image = Image.open("No_data.png")
        w = self.HER_STD_Strg_I_pil_image.width
        h = self.HER_STD_Strg_I_pil_image.height
        self.HER_STD_Strg_I_pil_image = self.HER_STD_Strg_I_pil_image.resize((int(w * (self.HER_STD_Strg_I_canvas_height / h)), 
                                                                int(h * (self.HER_STD_Strg_I_canvas_height / h))))
        self.HER_STD_Strg_I_plot = ImageTk.PhotoImage(image = self.HER_STD_Strg_I_pil_image, master = self.HER_STD_frame)
        # 画像を描画(中点x, 中点y, 画像)
        self.HER_STD_Strg_I_canvas.create_image(5 + self.HER_STD_Strg_I_canvas_width / 2, 5 + self.HER_STD_Strg_I_canvas_height / 2, 
                                       image = self.HER_STD_Strg_I_plot, tag = 'p1')

        # Strg(P vs T)
        # ラベル
        self.HER_STD_T_label = tk.Label(self.HER_STD_frame, text = 'Trend', font = ('Times new roman', 10))
        self.HER_STD_T_label.place(x = 165, y = 58)
        
        # キャンバスのサイズ
        self.HER_STD_Strg_T_canvas_width = 155
        self.HER_STD_Strg_T_canvas_height = 150
        # キャンバス
        self.HER_STD_Strg_T_canvas = tk.Canvas(self.HER_STD_frame, width = self.HER_STD_Strg_T_canvas_width, 
                                               height = self.HER_STD_Strg_T_canvas_height, relief='solid', bg = 'white', bd=1)
        self.HER_STD_Strg_T_canvas.place(x = 160, y = 75)
        # 画像を用意
        # 画像をリサイズする
        self.HER_STD_Strg_T_pil_image = Image.open("No_data.png")
        w = self.HER_STD_Strg_T_pil_image.width
        h = self.HER_STD_Strg_T_pil_image.height
        self.HER_STD_Strg_T_pil_image = self.HER_STD_Strg_T_pil_image.resize((int(w * (self.HER_STD_Strg_T_canvas_width / w)), 
                                                                int(h * (self.HER_STD_Strg_T_canvas_width / w))))
        self.HER_STD_Strg_T_plot = ImageTk.PhotoImage(image = self.HER_STD_Strg_T_pil_image, master = self.HER_STD_frame)
        # 画像を描画(中点x, 中点y, 画像)
        self.HER_STD_Strg_T_canvas.create_image(5 + self.HER_STD_Strg_T_canvas_width / 2, 5 + self.HER_STD_Strg_T_canvas_height / 2, 
                                       image = self.HER_STD_Strg_T_plot, tag = 'p2')

        # Tail
        # ラベル
        self.HER_STD_Tail_label = tk.Label(self.HER_STD_frame, text = 'Tail', font = ('Times new roman', 10))
        self.HER_STD_Tail_label.place(x = 5, y = 245)
        
        # キャンバスのサイズ
        self.HER_STD_Tail_canvas_width = 310
        self.HER_STD_Tail_canvas_height = 150
        # キャンバス
        self.HER_STD_Tail_canvas = tk.Canvas(self.HER_STD_frame, width = self.HER_STD_Tail_canvas_width, 
                                             height = self.HER_STD_Tail_canvas_height, relief='solid', bg = 'white', bd=1)
        self.HER_STD_Tail_canvas.place(x = 2, y = 265)
        # 画像を用意
        # 画像をリサイズする
        self.HER_STD_Tail_pil_image = Image.open("No_data.png")
        w = self.HER_STD_Tail_pil_image.width
        h = self.HER_STD_Tail_pil_image.height
        self.HER_STD_Tail_pil_image = self.HER_STD_Tail_pil_image.resize((int(w * (self.HER_STD_Tail_canvas_height / h)), 
                                                            int(h * (self.HER_STD_Tail_canvas_height / h))))
        self.HER_STD_Tail_plot = ImageTk.PhotoImage(image = self.HER_STD_Tail_pil_image, master = self.HER_STD_frame)
        # 画像を描画(中点x, 中点y, 画像)
        self.HER_STD_Tail_canvas.create_image(5 + self.HER_STD_Tail_canvas_width / 2, 5 + self.HER_STD_Tail_canvas_height / 2, 
                                     image = self.HER_STD_Tail_plot, tag = 'p3')

        # HERのCHKプロットを描くフレーム作成(キャンバス)------------------------------------------------------------
        # 基準配置ｘ座標
        LSx = 1040
        # フレーム
        self.HER_CHK_frame = tk.Frame(master, width = 340, height = 450, pady=10, padx=10, relief=tk.RIDGE, bd = 2)
        self.HER_CHK_frame.place(x = LSx, y = 500 )
        # ラベル
        self.HER_CHK_label = tk.Label(self.HER_CHK_frame, text = 'HER Check', font = ('Times new roman', 10))
        self.HER_CHK_label.place(x = 5, y = 0)
   
        # アボートタイミングラベル
        self.HER_CHK_AT_label = tk.Label(self.HER_CHK_frame, text = 'Date range  = ', fg = 'green', font = ('Arial', 9))
        self.HER_CHK_AT_label.place(x = 48, y = 17)
        
        # レコード名ラベル
        self.HER_CHK_RC_label = tk.Label(self.HER_CHK_frame, text = 'Record name = ', fg = 'green', font = ('Arial', 9))
        self.HER_CHK_RC_label.place(x = 48, y = 34)
     
        # Strg(P vs I)
        # ラベル
        self.HER_CHK_Strg_label = tk.Label(self.HER_CHK_frame, text = 'Storage', font = ('Times new roman', 10))
        self.HER_CHK_Strg_label.place(x = 5, y = 53)
        
        # キャンバスのサイズ
        self.HER_CHK_Strg_I_canvas_width = 145
        self.HER_CHK_Strg_I_canvas_height = 143
        # キャンバス
        self.HER_CHK_Strg_I_canvas = tk.Canvas(self.HER_CHK_frame, width = self.HER_CHK_Strg_I_canvas_width, 
                                               height = self.HER_CHK_Strg_I_canvas_height, relief='solid', bg = 'white', bd=1)
        self.HER_CHK_Strg_I_canvas.place(x = 2, y = 70)
        # 画像を用意
        # 画像をリサイズする
        self.HER_CHK_Strg_I_pil_image = Image.open("No_data.png")
        w = self.HER_CHK_Strg_I_pil_image.width
        h = self.HER_CHK_Strg_I_pil_image.height
        self.HER_CHK_Strg_I_pil_image = self.HER_CHK_Strg_I_pil_image.resize((int(w * (self.HER_CHK_Strg_I_canvas_height / h)), 
                                                                int(h * (self.HER_CHK_Strg_I_canvas_height / h))))
        self.HER_CHK_Strg_I_plot = ImageTk.PhotoImage(image = self.HER_CHK_Strg_I_pil_image, master = self.HER_CHK_frame)
        # 画像を描画(中点x, 中点y, 画像)
        self.HER_CHK_Strg_I_canvas.create_image(5 + self.HER_CHK_Strg_I_canvas_width / 2, 5 + self.HER_CHK_Strg_I_canvas_height / 2, 
                                       image = self.HER_CHK_Strg_I_plot, tag = 'p1')

        # HER Normal or Abnormalとしてrecord、dataをManualのファイルに保存するボタン
        self.HER_Save_Manual_Strg_btn = tk.Button(self.HER_CHK_frame, text = 'Save Strg record as', bg = 'lightgray', 
                                             font = font_general2, command = self.HER_Save_Manual_Strg) 
        self.HER_Save_Manual_Strg_btn.place(x = 42, y = 220)
        
        # NormalかAbnormalを選択するラジオボタン
        # チェック有無変数
        self.HER_Save_Manual_Strg_Mode_var = tk.IntVar()
        # value = 0のラジオボタンにチェックを入れる
        self.HER_Save_Manual_Strg_Mode_var.set(0)
        # リスト
        self.HER_Save_Manual_Strg_Mode_rdo_text = ['Nor', 'Abn']
        # ラジオボタン作成
        self.HER_Save_Manual_Strg_Mode_rdo_1 = tk.Radiobutton(self.HER_CHK_frame, value = 0, 
                                                              variable = self.HER_Save_Manual_Strg_Mode_var, 
                                                              text = self.HER_Save_Manual_Strg_Mode_rdo_text[0], 
                                                              font = font_general1)
        self.HER_Save_Manual_Strg_Mode_rdo_1.place(x = 183, y = 225)
        self.HER_Save_Manual_Strg_Mode_rdo_2 = tk.Radiobutton(self.HER_CHK_frame, value = 1, 
                                                              variable = self.HER_Save_Manual_Strg_Mode_var, 
                                                              text = self.HER_Save_Manual_Strg_Mode_rdo_text[1], 
                                                              font = font_general1)
        self.HER_Save_Manual_Strg_Mode_rdo_2.place(x = 230, y = 225)
        
        # Strg(P vs T)
        # ラベル
        self.HER_CHK_T_label = tk.Label(self.HER_CHK_frame, text = 'Trend', font = ('Times new roman', 10))
        self.HER_CHK_T_label.place(x = 165, y = 53)
        
        # キャンバスのサイズ
        self.HER_CHK_Strg_T_canvas_width = 155
        self.HER_CHK_Strg_T_canvas_height = 143
        # キャンバス
        self.HER_CHK_Strg_T_canvas = tk.Canvas(self.HER_CHK_frame, width = self.HER_CHK_Strg_T_canvas_width, 
                                               height = self.HER_CHK_Strg_T_canvas_height, relief='solid', bg = 'white', bd=1)
        self.HER_CHK_Strg_T_canvas.place(x = 160, y = 70)
        
        # 画像を用意
        # 画像をリサイズする
        self.HER_CHK_Strg_T_pil_image = Image.open("No_data.png")
        w = self.HER_CHK_Strg_T_pil_image.width
        h = self.HER_CHK_Strg_T_pil_image.height
        self.HER_CHK_Strg_T_pil_image = self.HER_CHK_Strg_T_pil_image.resize((int(w * (self.HER_CHK_Strg_T_canvas_width / w)), 
                                                                int(h * (self.HER_CHK_Strg_T_canvas_width / w))))
        self.HER_CHK_Strg_T_plot = ImageTk.PhotoImage(image = self.HER_CHK_Strg_T_pil_image, master = self.HER_CHK_frame)
        # 画像を描画(中点x, 中点y, 画像)
        self.HER_CHK_Strg_T_canvas.create_image(5 + self.HER_CHK_Strg_T_canvas_width / 2, 5 + self.HER_CHK_Strg_T_canvas_height / 2, 
                                       image = self.HER_CHK_Strg_T_plot, tag = 'p2')

        # Tail
        # ラベル
        self.HER_CHK_Tail_label = tk.Label(self.HER_CHK_frame, text = 'Tail', font = ('Times new roman', 10))
        self.HER_CHK_Tail_label.place(x = 5, y = 242)
        
        # キャンバスのサイズ
        self.HER_CHK_Tail_canvas_width = 310
        self.HER_CHK_Tail_canvas_height = 140
        # キャンバス
        self.HER_CHK_Tail_canvas = tk.Canvas(self.HER_CHK_frame, width = self.HER_CHK_Tail_canvas_width, 
                                             height = self.HER_CHK_Tail_canvas_height, relief='solid', bg = 'white', bd=1)
        self.HER_CHK_Tail_canvas.place(x = 2, y = 260)
        # 画像を用意
        # 画像をリサイズする
        self.HER_CHK_Tail_pil_image = Image.open("No_data.png")
        w = self.HER_CHK_Tail_pil_image.width
        h = self.HER_CHK_Tail_pil_image.height
        self.HER_CHK_Tail_pil_image = self.HER_CHK_Tail_pil_image.resize((int(w * (self.HER_CHK_Tail_canvas_height / h)), 
                                                            int(h * (self.HER_CHK_Tail_canvas_height / h))))
        self.HER_CHK_Tail_plot = ImageTk.PhotoImage(image = self.HER_CHK_Tail_pil_image, master = self.HER_CHK_frame)
        # 画像を描画(中点x, 中点y, 画像)
        self.HER_CHK_Tail_canvas.create_image(5 + self.HER_CHK_Tail_canvas_width / 2, 5 + self.HER_CHK_Tail_canvas_height / 2, 
                                     image = self.HER_CHK_Tail_plot, tag = 'p3')

        # HER Normal or Abnormalとしてrecord、dataをManualのファイルに保存するボタン(Tail)
        self.HER_Save_Manual_Tail_btn = tk.Button(self.HER_CHK_frame, text = 'Save Tail record as', bg = 'lightgray', 
                                             font = font_general2, command = self.HER_Save_Manual_Tail) 
        self.HER_Save_Manual_Tail_btn.place(x = 42, y = 407)
        
        # NormalかAbnormalを選択するラジオボタン
        # チェック有無変数
        self.HER_Save_Manual_Tail_Mode_var = tk.IntVar()
        # value = 0のラジオボタンにチェックを入れる
        self.HER_Save_Manual_Tail_Mode_var.set(0)
        # リスト
        self.HER_Save_Manual_Tail_Mode_rdo_text = ['Nor', 'Abn']
        # ラジオボタン作成
        self.HER_Save_Manual_Tail_Mode_rdo_1 = tk.Radiobutton(self.HER_CHK_frame, value = 0, 
                                                              variable = self.HER_Save_Manual_Tail_Mode_var, 
                                                              text = self.HER_Save_Manual_Tail_Mode_rdo_text[0], 
                                                              font = font_general1)
        self.HER_Save_Manual_Tail_Mode_rdo_1.place(x = 183, y = 413)
        self.HER_Save_Manual_Tail_Mode_rdo_2 = tk.Radiobutton(self.HER_CHK_frame, value = 1, 
                                                              variable = self.HER_Save_Manual_Tail_Mode_var, 
                                                              text = self.HER_Save_Manual_Tail_Mode_rdo_text[1], 
                                                              font = font_general1)
        self.HER_Save_Manual_Tail_Mode_rdo_2.place(x = 230, y = 413)
        
        # HERのABNトレンドプロットを描くフレーム作成(キャンバス)------------------------------------------------------------
        # 基準配置ｘ座標
        LSx = 170
        # フレーム
        self.HER_ABN_frame = tk.Frame(master, width = 680, height = 450, pady=10, padx=10, relief=tk.RIDGE, bd = 2)
        self.HER_ABN_frame.place(x = LSx, y = 500)
        
        # ラベル
        self.HER_Beam_Mode_label = tk.Label(self.HER_ABN_frame, text = 'Not defined', font = ('Times new roman', 10), fg = 'green')
        self.HER_Beam_Mode_label.place(x = 580, y = 0)

        # 推定原因のラベル
        self.HER_ABN_Cause_Strg_label = tk.Label(self.HER_ABN_frame, text = 'Possible cause (Strg)', font = ('Arial', 9),
                                                relief = tk.SOLID, bd = 0, width = 43, bg = 'lightgrey')
        self.HER_ABN_Cause_Strg_label.place(x = 40, y = 20)
        self.HER_ABN_Cause_Tail_label = tk.Label(self.HER_ABN_frame, text = 'Possible cause (Tail)', font = ('Arial', 9),
                                                relief = tk.SOLID, bd = 0, width = 43, bg = 'lightgrey')
        self.HER_ABN_Cause_Tail_label.place(x = 350, y = 20)
        
        # トレンドグラフのラベル
        self.HER_ABN_label = tk.Label(self.HER_ABN_frame, text = 'HER Abnormal Record Trend', font = ('Times new roman', 10))
        self.HER_ABN_label.place(x = 5, y = 0)
        # キャンバスのサイズ
        self.HER_ABN_canvas_width = 650
        self.HER_ABN_canvas_height = 300
        # キャンバス
        self.HER_ABN_canvas = tk.Canvas(self.HER_ABN_frame, width = self.HER_ABN_canvas_width, height = self.HER_ABN_canvas_height, 
                               relief='solid', bg = 'white', bd=1)
        self.HER_ABN_canvas.place(x = 3, y = 45)
        # 画像を用意
        # 画像をリサイズする
        self.HER_ABN_pil_image = Image.open("No_data.png")
        w = self.HER_ABN_pil_image.width
        h = self.HER_ABN_pil_image.height
        self.HER_ABN_pil_image = self.HER_ABN_pil_image.resize((int(w * (self.HER_ABN_canvas_width / w)), 
                                                                int(h * (self.HER_ABN_canvas_width / w))))
        self.HER_ABN_plot = ImageTk.PhotoImage(image = self.HER_ABN_pil_image, master = self.HER_ABN_frame)
        # 画像を描画(中点x, 中点y, 画像)
        self.HER_ABN_canvas.create_image(5 + self.HER_ABN_canvas_width / 2, 5 + self.HER_ABN_canvas_height / 2, 
                                         image = self.HER_ABN_plot, tag = 'p4')
        
        # 20240108変更
        # CCGの説明リストボックス
        # self.HER_CCG_Strg_list = tk.Listbox(self.HER_ABN_frame, height = 5, width = 52, font=("Arial", 7))
        # self.HER_CCG_Strg_list.place(x = 5, y = 355)
        
        # self.HER_CCG_Tail_list = tk.Listbox(self.HER_ABN_frame, height = 5, width = 52, font=("Arial", 7))
        # self.HER_CCG_Tail_list.place(x = 341, y = 355)
        
        # CCGの説明リストボックスの選択肢
        self.HER_CCG_Strg_var = tk.StringVar(value = self.HER_CCG_Strg_List_box)
        self.HER_CCG_Strg_listbox = tk.Listbox(self.HER_ABN_frame, listvariable = self.HER_CCG_Strg_var, font=("Arial", 7))
        self.HER_CCG_Strg_listbox.bind('<<ListboxSelect>>', self.change_HER_Strg_Cause)
        self.HER_CCG_Strg_listbox.place(x = 5, y =355, width = 300, height = 74)
        
        self.HER_CCG_Tail_var = tk.StringVar(value = self.HER_CCG_Tail_List_box)
        self.HER_CCG_Tail_listbox = tk.Listbox(self.HER_ABN_frame, listvariable = self.HER_CCG_Tail_var, font=("Arial", 7))
        self.HER_CCG_Tail_listbox.bind('<<ListboxSelect>>', self.change_HER_Tail_Cause)
        self.HER_CCG_Tail_listbox.place(x = 341, y =355, width = 300, height = 74)
        
        # Abormal recordの数のラベル
        self.HER_N_Rec_Strg_label = tk.Label(self.HER_ABN_frame, text = '00', font = ('Arial', 9))
        self.HER_N_Rec_Strg_label.place(x = 307, y = 410)
        
        self.HER_N_Rec_Tail_label = tk.Label(self.HER_ABN_frame, text = '00', font = ('Arial', 9))
        self.HER_N_Rec_Tail_label.place(x = 643, y = 410)
        
        # Trigger設定 TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
        self.LER_tc = 0
        # LER create the channel
        # self.LER_pv = 'VALCCG:D01_L13:ONOF:OUTr'
        self.LER_pv = 'CGLSAFE:MR:ABORT'
        self.LER_chid = ca.create_channel(self.LER_pv)
        # self.LER_state = ca.connect_channel(self.LER_chid)

        # LER subscribe to events giving a callback function
        self.LER_eventID = ca.create_subscription(self.LER_chid, callback = self.LERtriggerEvent)

        self.HER_tc = 0
        # HER create the channel
        # self.HER_pv = 'VALCCG:D01_L23:ONOF:OUTr'
        self.HER_pv = 'CGHSAFE:MR:ABORT'
        self.HER_chid = ca.create_channel(self.HER_pv)
        # self.HER_state = ca.connect_channel(self.HER_chid)

        # HER subscribe to events giving a callback function
        self.HER_eventID = ca.create_subscription(self.HER_chid, callback = self.HERtriggerEvent)
        
        # 20240206変更
        # self.LastRunTime = datetime.datetime.now()
        self.LER_LastRunTime = datetime.datetime(year = 2000, month = 1, day = 1, hour = 0)
        self.HER_LastRunTime = datetime.datetime(year = 2000, month = 1, day = 1, hour = 0)
        
    def time_update(self):
        Nowtime = datetime.datetime.now()
        self.Nowtime_str = Nowtime.strftime('%Y-%m-%d %H:%M:%S')
        self.time_label["text"] = self.Nowtime_str[2:]
        
        self.time_label.after(1000 * 10, self.time_update)
        
    def change_LER_LC_List(self):
        n = self.LER_LC_var.get()
        self.LER_Record_Name_listbox.delete(0, tk.END)
        for item in self.LER_Record_Name_List_box[n]:
            self.LER_Record_Name_listbox.insert(tk.END, item)

    def change_HER_LC_List(self):
        n = self.HER_LC_var.get()
        self.HER_Record_Name_listbox.delete(0, tk.END)
        for item in self.HER_Record_Name_List_box[n]:
            self.HER_Record_Name_listbox.insert(tk.END, item)
            
    def show_LER_selected(self, event):
        indices = self.LER_Abort_Time_listbox.curselection()
        if len(indices) != 1:
            return
        n = indices[0]
        self.LER_Abort_Time_entry.delete(0, tk.END)
        self.LER_Abort_Time_entry.insert(0, self.LER_Abort_Time_listbox.get(n))

    def show_ler_record_selected(self, event):
        indices = self.LER_Record_Name_listbox.curselection()
        if len(indices) != 1:
            return
        n = indices[0]
        self.LER_Record_Name_entry.delete(0, tk.END)
        self.LER_Record_Name_entry.insert(0, self.LER_Record_Name_listbox.get(n))

    def show_HER_selected(self, event):
        indices = self.HER_Abort_Time_listbox.curselection()
        if len(indices) != 1:
            return
        n = indices[0]
        self.HER_Abort_Time_entry.delete(0, tk.END)
        self.HER_Abort_Time_entry.insert(0, self.HER_Abort_Time_listbox.get(n))

    def show_her_record_selected(self, event):
        indices = self.HER_Record_Name_listbox.curselection()
        if len(indices) != 1:
            return
        n = indices[0]
        self.HER_Record_Name_entry.delete(0, tk.END)
        self.HER_Record_Name_entry.insert(0, self.HER_Record_Name_listbox.get(n))
        
    # 20240108追記
    def change_LER_Strg_Cause(self, event):
        indices = self.LER_CCG_Strg_listbox.curselection()
        if len(indices) != 1:
            return
        n = indices[0]
        cl = ['red', 'orange', 'green', 'deepskyblue', 'blue']
        # 推定原因のラベル表示の変更
        self.LER_ABN_Cause_Strg_label["text"] = self.LER_ABN_Cause_Strg_List[n][0]
        if(self.LER_ABN_Cause_Strg_List[n][0] != 'none'):
            self.LER_ABN_Cause_Strg_label["bg"] = 'white'
            self.LER_ABN_Cause_Strg_label["fg"] = cl[n]
        else:
            self.LER_ABN_Cause_Strg_label["bg"] = 'lightgrey'
            self.LER_ABN_Cause_Strg_label["fg"] = 'black'
        # レコード名変更
        self.LER_Record_Name_entry.delete(0, tk.END)
        self.LER_Record_Name_entry.insert(0, self.LER_ABN_Cause_Strg_List[n][1])
                
    # 20240108追記
    def change_LER_Tail_Cause(self, event):
        indices = self.LER_CCG_Tail_listbox.curselection()
        if len(indices) != 1:
            return
        n = indices[0]
        cl = ['red', 'orange', 'green', 'deepskyblue', 'blue']
        # 推定原因のラベル表示の変更
        self.LER_ABN_Cause_Tail_label["text"] = self.LER_ABN_Cause_Tail_List[n][0]
        if(self.LER_ABN_Cause_Tail_List[n][0] != 'none'):
            self.LER_ABN_Cause_Tail_label["bg"] = 'white'
            self.LER_ABN_Cause_Tail_label["fg"] = cl[n]
        else:
            self.LER_ABN_Cause_Tail_label["bg"] = 'lightgrey'
            self.LER_ABN_Cause_Tail_label["fg"] = 'black'
        # レコド名変更
        self.LER_Record_Name_entry.delete(0, tk.END)
        self.LER_Record_Name_entry.insert(0, self.LER_ABN_Cause_Tail_List[n][1])
    
    # 20240108追記
    def change_HER_Strg_Cause(self, event):
        indices = self.HER_CCG_Strg_listbox.curselection()
        if len(indices) != 1:
            return
        n = indices[0]
        cl = ['red', 'orange', 'green', 'deepskyblue', 'blue']
        # 推定原因のラベル表示の変更
        self.HER_ABN_Cause_Strg_label["text"] = self.HER_ABN_Cause_Strg_List[n][0]
        if(self.HER_ABN_Cause_Strg_List[n][0] != 'none'):
            self.HER_ABN_Cause_Strg_label["bg"] = 'white'
            self.HER_ABN_Cause_Strg_label["fg"] = cl[n]
        else:
            self.HER_ABN_Cause_Strg_label["bg"] = 'lightgrey'
            self.HER_ABN_Cause_Strg_label["fg"] = 'black'
        # レコード名変更
        self.HER_Record_Name_entry.delete(0, tk.END)
        self.HER_Record_Name_entry.insert(0, self.HER_ABN_Cause_Strg_List[n][1])
                
    # 20240108追記
    def change_HER_Tail_Cause(self, event):
        indices = self.HER_CCG_Tail_listbox.curselection()
        if len(indices) != 1:
            return
        n = indices[0]
        cl = ['red', 'orange', 'green', 'deepskyblue', 'blue']
        # 推定原因のラベル表示の変更
        self.HER_ABN_Cause_Tail_label["text"] = self.HER_ABN_Cause_Tail_List[n][0]
        if(self.HER_ABN_Cause_Tail_List[n][0] != 'none'):
            self.HER_ABN_Cause_Tail_label["bg"] = 'white'
            self.HER_ABN_Cause_Tail_label["fg"] = cl[n]
        else:
            self.HER_ABN_Cause_Tail_label["bg"] = 'lightgrey'
            self.HER_ABN_Cause_Tail_label["fg"] = 'black'
        # レコード名変更
        self.HER_Record_Name_entry.delete(0, tk.END)
        self.HER_Record_Name_entry.insert(0, self.HER_ABN_Cause_Tail_List[n][1])
            
    def Status_change_running(self):
        # 状態表示変更
        self.Status_label["text"] = "Running\nPlease wait"
        self.Status_label["fg"] = "red"
        self.Status_label.update()
    
    def Status_change_waiting(self):
        # 状態表示変更
        self.Status_label["text"] = "Waiting"
        self.Status_label["fg"] = "blue"
        
    # 実行タイマー起動用関数(LER)
    def LER_timeEvent(self):
        # モードの再確認 auto かmanualか
        n = self.Mode_var.get()
        Run_Mode = self.Mode_rdo_text[n]
        
        # モードがManualだったらautoをやめる
        if(Run_Mode == 'Manual'):
            self.stop_time()
            return
        
        # Auto時のモードの確認　Abort triggerかConst Inervalか
        m = self.Auto_Mode_var.get()
        Auto_Mode = self.Auto_Mode_rdo_text[m]
        
        # Autoモードで定期実行なら
        if(Run_Mode == 'Auto') and (Auto_Mode == 'Cns Intr'):
            Now = datetime.datetime.now()
            # 現在時刻の3分前をAbort timingとする
            Now_3b = Now + datetime.timedelta(minutes = -3)
            Abort_Timing = Now_3b.strftime('%Y-%m-%d %H:%M:%S')
            self.LER_Abort_Time_entry.delete(0, tk.END)
            self.LER_Abort_Time_entry.insert(0, Abort_Timing)
            
            # 現在時刻と次の実行時刻
            self.LER_Last_Date = Now.strftime('%Y-%m-%d %H:%M:%S')
            self.LER_Next_Date = Now + datetime.timedelta(minutes = int(float(self.LER_Interval_h) * 60))
            self.LER_Next_Date = self.LER_Next_Date.strftime('%Y-%m-%d %H:%M:%S')
        
            # 再帰的に関数を呼び出す
            delaytime = int(float(self.LER_Interval_h) * 60 * 60 * 1000)
            self.LER_id = self.after(delaytime, self.LER_timeEvent)
            #20240204追加
            self.LER_id_first = 1
            
            self.loglist.insert(tk.END, "Const Last:")
            self.loglist.itemconfigure(tk.END, foreground = 'red')
            self.loglist.insert(tk.END, '  ' + self.LER_Last_Date[5:])
            self.loglist.insert(tk.END, "Const Next:")
            self.loglist.itemconfigure(tk.END, foreground = 'red')
            self.loglist.insert(tk.END, '  ' + self.LER_Next_Date[5:])
            self.loglist.see(tk.END)
            
            # 20240206追加　実行した時間
            self.LER_LastRunTime = datetime.datetime.now()
        
            # 実行するスレッドインスタンス生成
            # triggerモードの中のconstモード
            self.LER_trg_cns = 1
            self.LER_update()
            
            # LER_th = threading.Thread(target = self.LER_update)
            # 実行スレッドスタート
            # LER_th.start()
        
        if(Run_Mode == 'Auto')and(Auto_Mode == 'Abt Trg'):
            # 現在時刻
            Now = datetime.datetime.now()
            
            # 最後のAbort time (実行したのは3分後)
            LER_Last_abtime = self.LER_Abort_Time_entry.get()
            LER_Last_abtime_dt = datetime.datetime.strptime(LER_Last_abtime, '%Y-%m-%d %H:%M:%S')
            # 最後の実行時刻
            LER_Last_extime_dt = LER_Last_abtime_dt + datetime.timedelta(minutes = 3)
            
            # 現在時刻と最後の実行時刻との差
            LER_td = Now - LER_Last_extime_dt
            
            # 指定時間以上経ってたら
            if(LER_td.seconds >= float(self.LER_Interval_h) * 3600):
                # 現在時刻と次回実行時刻
                self.LER_Last_Date = Now.strftime('%Y-%m-%d %H:%M:%S')
                self.LER_Next_Date = Now + datetime.timedelta(minutes = int(float(self.LER_Interval_h) * 60))
                self.LER_Next_Date = self.LER_Next_Date.strftime('%Y-%m-%d %H:%M:%S')
                
                # 現在時刻の3分前をAbort timingとする
                Now_3b = Now + datetime.timedelta(minutes = -3)
                Abort_Timing = Now_3b.strftime('%Y-%m-%d %H:%M:%S')
                self.LER_Abort_Time_entry.delete(0, tk.END)
                self.LER_Abort_Time_entry.insert(0, Abort_Timing)
                
                # 実行しているスレッドインスタンスを止める。
                self.LER_stop_time()
                # Auto中
                self.LER_auto_run = 1
                
                # 再帰的に関数を呼び出す
                delaytime = int(float(self.LER_Interval_h) * 60 * 60 * 1000)
                self.LER_id = self.after(delaytime, self.LER_timeEvent)
                #20240204追加
                self.LER_id_first = 1
                
                # Loglistに書き込む
                self.loglist.insert(tk.END, "Trigger Last:")
                self.loglist.itemconfigure(tk.END, foreground = 'red')
                self.loglist.insert(tk.END, '  ' + self.LER_Last_Date[5:])
                self.loglist.insert(tk.END, "Trigger Next:")
                self.loglist.itemconfigure(tk.END, foreground = 'red')
                self.loglist.insert(tk.END, '  ' + self.LER_Next_Date[5:])
                self.loglist.see(tk.END)
                
                # 20240206追加　実行した時間
                self.LER_LastRunTime = datetime.datetime.now()
        
                # 実行するスレッドインスタンス生成
                # triggerモードの中のconstモード
                self.LER_trg_cns = 1
                self.LER_update()
                
                # LER_th = threading.Thread(target = self.LER_update)
                # 実行スレッドスタート
                # LER_th.start()
            
            # 一定時間未満なら
            else:
                # 次回実行時刻
                self.LER_Next_Date = Now + datetime.timedelta(seconds = int(float(self.LER_Interval_h) * 60 * 60 - LER_td.seconds))
                # 20240206変更
                # self.LER_Next_Date = Now + datetime.timedelta(seconds = int(float(self.LER_Interval_h) * 60 * 60))

                self.LER_Next_Date = self.LER_Next_Date.strftime('%Y-%m-%d %H:%M:%S')
                self.loglist.insert(tk.END, "Trigger_r Next:")
                self.loglist.itemconfigure(tk.END, foreground = 'red')
                self.loglist.insert(tk.END, '  ' + self.LER_Next_Date[5:])
                self.loglist.see(tk.END)
                
                # 実行しているスレッドインスタンスを止める。
                self.LER_stop_time()
                # Auto中
                self.LER_auto_run = 1
                
                # 再帰的に関数を呼び出す
                delaytime = int(float(self.LER_Interval_h) * 60 * 60 * 1000 - LER_td.seconds * 1000)
                # 20240206変更
                # delaytime = int(float(self.LER_Interval_h) * 60 * 60 * 1000)
                self.LER_id = self.after(delaytime, self.LER_timeEvent)
                #20240204追加
                self.LER_id_first = 1
            
    # 実行タイマー起動用関数(HER)
    def HER_timeEvent(self):
        # モードの再確認 auto かmanualか
        n = self.Mode_var.get()
        Run_Mode = self.Mode_rdo_text[n]
        
        # モードがManualだったらautoをやめる
        if(Run_Mode == 'Manual'):
            self.stop_time()
            return
        
        # Auto時のモードの確認　Abort triggerかInervalか
        m = self.Auto_Mode_var.get()
        Auto_Mode = self.Auto_Mode_rdo_text[m]
        
        # Autoモードで定期実行なら
        if(Run_Mode == 'Auto') and (Auto_Mode == 'Cns Intr'):
            Now = datetime.datetime.now()
            # 現在時刻の3分前をAbort timingとする
            Now_3b = Now + datetime.timedelta(minutes = -3)
            Abort_Timing = Now_3b.strftime('%Y-%m-%d %H:%M:%S')
            self.HER_Abort_Time_entry.delete(0, tk.END)
            self.HER_Abort_Time_entry.insert(0, Abort_Timing)
            
            # 現在時刻と次の実行時刻
            self.HER_Last_Date = Now.strftime('%Y-%m-%d %H:%M:%S')
            self.HER_Next_Date = Now + datetime.timedelta(minutes = int(float(self.HER_Interval_h) * 60))
            self.HER_Next_Date = self.HER_Next_Date.strftime('%Y-%m-%d %H:%M:%S')

            # 再帰的に関数を呼び出す
            delaytime = int(float(self.HER_Interval_h) * 60 * 60 * 1000)
            self.HER_id = self.after(delaytime, self.HER_timeEvent)
            #20240204追加
            self.HER_id_first = 1
            
            self.loglist.insert(tk.END, "Const Last:")
            self.loglist.itemconfigure(tk.END, foreground = 'blue')
            self.loglist.insert(tk.END, '  ' + self.HER_Last_Date[5:])
            self.loglist.insert(tk.END, "Const Next:")
            self.loglist.itemconfigure(tk.END, foreground = 'blue')
            self.loglist.insert(tk.END, '  ' + self.HER_Next_Date[5:])
            self.loglist.see(tk.END)
            
            # 20240206追加　実行した時間
            self.HER_LastRunTime = datetime.datetime.now()
        
            # 実行するスレッドインスタンス生成
            # triggerモードの中のconstモード
            self.HER_trg_cns = 1
            self.HER_update()
            
            # HER_th = threading.Thread(target = self.HER_update)
            # 実行スレッドスタート
            # HER_th.start()
        
        if(Run_Mode == 'Auto')and(Auto_Mode == 'Abt Trg'):
            # 現在時刻
            Now = datetime.datetime.now()
            
            # 最後のAbort time (実行したのは3分後)
            HER_Last_abtime = self.HER_Abort_Time_entry.get()
            HER_Last_abtime_dt = datetime.datetime.strptime(HER_Last_abtime, '%Y-%m-%d %H:%M:%S')
            # 最後の実行時刻
            HER_Last_extime_dt = HER_Last_abtime_dt + datetime.timedelta(minutes = 3)
            
            # 現在時刻と最後のアボート時刻との差
            HER_td = Now - HER_Last_extime_dt
            
            # 指定時間以上経ってたら
            if(HER_td.seconds >= float(self.HER_Interval_h) * 3600):
                # 現在時刻と次回実行時刻
                self.HER_Last_Date = Now.strftime('%Y-%m-%d %H:%M:%S')
                self.HER_Next_Date = Now + datetime.timedelta(minutes = int(float(self.HER_Interval_h) * 60))
                self.HER_Next_Date = self.HER_Next_Date.strftime('%Y-%m-%d %H:%M:%S')
                
                # 現在時刻の3分前をAbort timingとする
                Now_3b = Now + datetime.timedelta(minutes = -3)
                Abort_Timing = Now_3b.strftime('%Y-%m-%d %H:%M:%S')
                self.HER_Abort_Time_entry.delete(0, tk.END)
                self.HER_Abort_Time_entry.insert(0, Abort_Timing)
                
                # 実行しているスレッドインスタンスを止める。
                self.HER_stop_time()
                # Auto中
                self.HER_auto_run = 1
                
                # 再帰的に関数を呼び出す
                delaytime = int(float(self.HER_Interval_h) * 60 * 60 * 1000)
                self.HER_id = self.after(delaytime, self.HER_timeEvent)
                #20240204追加
                self.HER_id_first = 1
                
                # Loglistに書き込む
                self.loglist.insert(tk.END, "Trigger Last:")
                self.loglist.itemconfigure(tk.END, foreground = 'blue')
                self.loglist.insert(tk.END, '  ' + self.HER_Last_Date[5:])
                self.loglist.insert(tk.END, "Trigger Next:")
                self.loglist.itemconfigure(tk.END, foreground = 'blue')
                self.loglist.insert(tk.END, '  ' + self.HER_Next_Date[5:])
                self.loglist.see(tk.END)
            
                # 20240206追加　実行した時間
                self.HER_LastRunTime = datetime.datetime.now()
            
                # 実行するスレッドインスタンス生成
                # triggerモードの中のconstモード
                self.HER_trg_cns = 1
                self.HER_update()
                
                # HER_th = threading.Thread(target = self.HER_update)
                # 実行スレッドスタート
                # HER_th.start()
            
            # 一定時間未満なら
            else:
                # 次回実行時刻
                self.HER_Next_Date = Now + datetime.timedelta(seconds = int(float(self.HER_Interval_h) * 60 * 60 - HER_td.seconds))
                # 20240206変更
                # self.HER_Next_Date = Now + datetime.timedelta(seconds = int(float(self.HER_Interval_h) * 60 * 60))
                self.HER_Next_Date = self.HER_Next_Date.strftime('%Y-%m-%d %H:%M:%S')
                self.loglist.insert(tk.END, "Trigger_r Next:")
                self.loglist.itemconfigure(tk.END, foreground = 'blue')
                self.loglist.insert(tk.END, '  ' + self.HER_Next_Date[5:])
                self.loglist.see(tk.END)
                
                # 実行しているスレッドインスタンスを止める。
                self.HER_stop_time()
                # Auto中
                self.HER_auto_run = 1
            
                # 再帰的に関数を呼び出す
                delaytime = int(float(self.HER_Interval_h) * 60 * 60 * 1000 - HER_td.seconds * 1000)
                # 20240206変更
                # delaytime = int(float(self.HER_Interval_h) * 60 * 60 * 1000)
                self.HER_id = self.after(delaytime, self.HER_timeEvent)
                #20240204追加
                self.HER_id_first = 1
            
    # スレッドの処理(LER)
    def LER_update(self):
        # スレッドを定義
        thread_LER = threading.Thread(target = Main_Command_LER(self))
        
        # スタート
        thread_LER.start()
        self.Status_change_waiting()

    # スレッドの処理(HER)
    def HER_update(self):
        # スレッドを定義
        thread_HER = threading.Thread(target = Main_Command_HER(self))
        
        # スタート
        thread_HER.start()
        self.Status_change_waiting()
        
    # スレッドキャンセル(HER、LER)
    def stop_time(self):    
        try:
            self.after_cancel(self.LERid)
        except Exception as e:
            print(traceback.format_exc())
            print(str(e))
            Error_f = str(e)
            tk.messagebox.showinfo('Check mode', 'Check Auto mode or Manual mode')
            
        try:
            self.after_cancel(self.HERid)
        except Exception as e:
            print(traceback.format_exc())
            print(str(e))
            Error_f = str(e)
            tk.messagebox.showinfo('Check mode', 'Check Auto mode or Manual mode')
        
        if (self.LER_id_first == 1):
            try:
                self.after_cancel(self.LER_id)
            except Exception as e:
                print(traceback.format_exc())
                print(str(e))
                Error_f = str(e)
                tk.messagebox.showinfo('Check mode', 'Check Auto mode or Manual mode')
            
        if (self.HER_id_first == 1):
            try:
                self.after_cancel(self.HER_id)
            except Exception as e:
                print(traceback.format_exc())
                print(str(e))
                Error_f = str(e)
                tk.messagebox.showinfo('Check mode', 'Check Auto mode or Manual mode')
            
        # 状態表示変更
        # self.Status_change_waiting()
        self.Status_LER_Auto_label["text"] = "LER Auto Stop"
        self.Status_LER_Auto_label["fg"] = "blue"
        self.Status_LER_Auto_label.update()
        
        self.Status_HER_Auto_label["text"] = "HER Auto Stop"
        self.Status_HER_Auto_label["fg"] = "blue"
        self.Status_HER_Auto_label.update()
        
        # スタートボタンを有効化
        self.Run_Start_btn["state"] = "active"
        
        # Autoの状況クリア
        self.LER_auto_run = 0
        self.HER_auto_run = 0
    
    # スレッドキャンセル(LER const)
    def LER_stop_time(self):
        if (self.LER_id_first == 1):
            try:
                self.after_cancel(self.LER_id)
            except Exception as e:
                print(traceback.format_exc())
                print(str(e))
                Error_f = str(e)
        
        self.LER_auto_run = 0
    
    # スレッドキャンセル(HER const)
    def HER_stop_time(self):
        if (self.HER_id_first == 1):
            try:
                self.after_cancel(self.HER_id)
            except Exception as e:
                print(traceback.format_exc())
                print(str(e))
                Error_f = str(e)
            
        self.HER_auto_run = 0
            
    def Start_btn_click(self):
        
        # モードの確認 auto かmanualか
        n = self.Mode_var.get()
        Run_Mode = self.Mode_rdo_text[n]
        
        # Auto時のモードの確認　Abort triggerかConst Inervalか
        m = self.Auto_Mode_var.get()
        Auto_Mode = self.Auto_Mode_rdo_text[m]
        
        # Manualモードだったら
        if(Run_Mode == 'Manual'):
            # LERかHERか
            n = self.Ring_var.get()
            Ring_Name = self.Ring_rdo_text[n]
            
            # Abort timeの確認
            # 現在時刻
            Now = datetime.datetime.now()
            
            # エントリーのAbort time
            if(Ring_Name =='LER'):
                Last_abtime = self.LER_Abort_Time_entry.get()
            else:
                Last_abtime = self.HER_Abort_Time_entry.get()
            
            try:
                Last_abtime_dt = datetime.datetime.strptime(Last_abtime, '%Y-%m-%d %H:%M:%S')
            except Exception as e:
                print(traceback.format_exc())
                print(str(e))
                Error_f = str(e)
                tk.messagebox.showinfo('Check Time format', Error_f)
                if(Ring_Name =='LER'):
                    self.LER_ABN_canvas["highlightthickness"] = 0
                    self.LER_ABN_canvas.update()
                if(Ring_Name =='HER'):
                    self.HER_ABN_canvas["highlightthickness"] = 0
                    self.HER_ABN_canvas.update()
                return
    
            # 現在時刻との差
            Ab_td = Now - Last_abtime_dt

            #3分未満だったら
            if(Ab_td.seconds < float(3 * 60)):
                print('Input abort time 3min before at least')
                tk.messagebox.showinfo('Wrong abort time', 'Input abort time 3min before at least ' + Ring_Name)
                return
            
            # 状態表示変更
            self.Status_change_running()
            # loglistに記録
            self.Now_n = datetime.datetime.now()
            self.Now_t = self.Now_n.strftime('%Y-%m-%d %H:%M:%S')
            self.loglist.insert(tk.END, 'Manual start:')
            if(Ring_Name == 'LER'):
                self.loglist.itemconfigure(tk.END, foreground = 'red')
            else:
                self.loglist.itemconfigure(tk.END, foreground = 'blue')
            self.loglist.insert(tk.END, '  ' + self.Now_t[5:])
            self.loglist.see(tk.END)
            
            if(Ring_Name == 'LER'):
                self.LER_trg_cns = 0
                Main_Command_LER(self)
            elif(Ring_Name == 'HER'):
                self.HER_trg_cns = 0
                Main_Command_HER(self)

            # 状態表示変更
            self.Status_change_waiting()
            # loglistに記録
            self.Now_n = datetime.datetime.now()
            self.Now_t = self.Now_n.strftime('%Y-%m-%d %H:%M:%S')
            self.loglist.insert(tk.END, 'Manual end:')
            if(Ring_Name == 'LER'):
                self.loglist.itemconfigure(tk.END, foreground = 'red')
            else:
                self.loglist.itemconfigure(tk.END, foreground = 'blue')
            self.loglist.insert(tk.END, '  ' + self.Now_t[5:])
            self.loglist.see(tk.END)
        
        # Autoモードだったら
        elif(Run_Mode == 'Auto'):
            # 既にAuto modeで走っているならメッセージを出す
            if(self.Status_LER_Auto_label["text"] == "LER Auto Running") or (self.Status_HER_Auto_label["text"] == "HER Auto Running"):
                print('Running in Auto mode')
                tk.messagebox.showinfo('Error', 'Stop Auto mode or Try Manual mode.')
                return
            
            # constモード、triggerモード、両方走る
            # loglistに記録
            self.Now_n = datetime.datetime.now()
            self.Now_t = self.Now_n.strftime('%Y-%m-%d %H:%M:%S')
            self.loglist.insert(tk.END, 'Auto start:')
            self.loglist.insert(tk.END, '  ' + self.Now_t[5:])
            self.loglist.see(tk.END)
            
            # 状態表示変更
            self.Status_change_waiting()
            
            #　インターバルの時間 [hour]
            self.LER_Interval_h = self.Interval_entry.get()
            self.HER_Interval_h = self.Interval_entry.get()
            
            # formatチェック
            try:
                Hi = float(self.LER_Interval_h)
            except Exception as e:
                print(traceback.format_exc())
                print(str(e))
                Error_f = str(e)
                tk.messagebox.showinfo('Check interval time format', Error_f)
                self.loglist.insert(tk.END, 'Auto end:')
                self.loglist.insert(tk.END, '  ' + self.Now_t[5:])
                self.loglist.see(tk.END)
                return

            # スタート時刻をAbort timeエントリーに書き込む (LER)
            Now = datetime.datetime.now()
            if(Auto_Mode == 'Abt Trg'):
                # Abort timeを3分前にする
                Now_3b = Now + datetime.timedelta(minutes = -3)
                self.LER_Last_Date = Now_3b.strftime('%Y-%m-%d %H:%M:%S')
                self.LER_Abort_Time_entry.delete(0, tk.END)
                self.LER_Abort_Time_entry.insert(0, self.LER_Last_Date)
            elif(Auto_Mode == 'Cns Intr'):
                self.LER_Last_Date = Now.strftime('%Y-%m-%d %H:%M:%S')
                self.LER_Abort_Time_entry.delete(0, tk.END)
                self.LER_Abort_Time_entry.insert(0, self.LER_Last_Date)
            
            # Autoモードラベル変更 (LER)
            self.Status_LER_Auto_label["text"] = "LER Auto Running"
            self.Status_LER_Auto_label["fg"] = "red"
            self.Status_LER_Auto_label.update()
            
            # 実行タイマー起動 (LER)
            self.LER_timeEvent()
            self.LER_auto_run = 1
            
            # スタート時刻をAbort timeエントリーに書き込む (HER)
            Now = datetime.datetime.now()
            if(Auto_Mode == 'Abt Trg'):
                # Abort timeを3分前にする
                Now_3b = Now + datetime.timedelta(minutes = -3)
                self.HER_Last_Date = Now_3b.strftime('%Y-%m-%d %H:%M:%S')
                self.HER_Abort_Time_entry.delete(0, tk.END)
                self.HER_Abort_Time_entry.insert(0, self.HER_Last_Date)
            elif(Auto_Mode == 'Cns Intr'):
                self.HER_Last_Date = Now.strftime('%Y-%m-%d %H:%M:%S')
                self.HER_Abort_Time_entry.delete(0, tk.END)
                self.HER_Abort_Time_entry.insert(0, self.HER_Last_Date)
            
            # Autoモードラベル変更 (HER)
            self.Status_HER_Auto_label["text"] = "HER Auto Running"
            self.Status_HER_Auto_label["fg"] = "red"
            self.Status_HER_Auto_label.update()
            
            # 実行タイマー起動 (HER)
            self.HER_timeEvent()
            self.HER_auto_run = 1
    
    def LER_Save_Manual_Strg(self):
        # エラーフラグリセット
        Error_f = 'none'
    
        # Recordnameがnoneなら何もしない
        Check_Record_Name = self.LER_Record_Name_entry.get()
        if (Check_Record_Name == 'none'):
            tk.messagebox.showinfo('No record name', 'LER No record name')
            return
        
        # リングの名前
        Ring_Name = 'LER'
        List_Para = self.LER_CCG_List
        
        # Abort Timing
        Abort_Timing = self.LER_Abort_Time_entry.get()
    
        # モードの確認 autoかmanualか
        n = self.Mode_var.get()
        Run_Mode = self.Mode_rdo_text[n]
        
        # Auto時のモードの確認 Abort triggerかIntervalか
        m = self.Auto_Mode_var.get()
        Auto_Mode = self.Auto_Mode_rdo_text[m]
        
        # constモード(Triggerモード中のconstモード)
        Cns_Mode = self.LER_trg_cns
        
        # Auto_modeで走っているかどうか
        Arun = self.LER_auto_run
        
        # normalとしてsaveするか、abnormalとしてsaveするか
        n = self.LER_Save_Manual_Strg_Mode_var.get()
        Save_Class = self.LER_Save_Manual_Strg_Mode_rdo_text[n]
            
        # 時間間隔
        self.LER_Interval_h = self.Interval_entry.get()
        Hadv = float(self.LER_Interval_h)
    
        # Referenceデータの最終日
        self.LER_Last_Ref_d = self.Last_Ref_entry.get()
        Last_Refd = float(self.LER_Last_Ref_d)
    
        # Referenceデータの期間(日)
        self.LER_Ref_Period_d = self.Ref_Period_entry.get()
        Ref_Pd = float(self.LER_Ref_Period_d)
        
        # アボートタイミングから、各モードのDate_Range(データの時間範囲)を決める
        # ビームがあるかどうかも調べる。No_Beam = 1 ならビームなし(シャットダウン中とか)、0 ならビームあり
        No_Beam, Error_f = Define_Date_Range(Abort_Timing, List_Para, Run_Mode, Auto_Mode, Cns_Mode, Arun, Hadv, 
                                             Last_Refd, Ref_Pd)
        if(Error_f != 'none'):
            tk.messagebox.showinfo('LER Strg Error', Error_f)
            return
    
        # 選別する方法
        Method = 'FNN'  # FNN 回帰モデルの結果を使い自動で選別(Strgはi32)
        
        # 各側室のCCGの数+1
        LER_CCG_n = [0] * 12
        for k in range(12):
            LER_CCG_n[k] = len(self.LER_Record_Name_List_box[k])

        # アボート直前の蓄積中の圧力を調べ、
        # スタンダードの回帰曲線での予想とズレの大きなレコードをみつける(DIFモード)
                                  
        if(No_Beam == 0): # ビームがあったら
        
            # リングの各側室についてMSE計算に用いるデータを取り出す
            # Date_Range_DIFをファイルに保存する
            Mode_Para = 'DIF_Strg'
            Date_Range_DIF, Error_fds = Get_DIF_Strg(List_Para, LER_CCG_n)

            # エラーがあったら、
            if(Error_fds != 'none'):
                tk.messagebox.showinfo('LER Error (With Beam)', Error_fds)

            # エラーが無かったら
            else:
                # 確認メッセージ
                Messagebox = tk.messagebox.askquestion('Confirmation', 'Save LER Strg data as ' + Save_Class +'. OK ?', 
                                                       icon='warning')
                if Messagebox == 'no':
                    return
            
                Save_Manual_Strg(self, Method, Mode_Para, List_Para, Check_Record_Name, Save_Class, Abort_Timing)

                # 終了メッセージ
                tk.messagebox.showinfo('Information', 'LER Strg data (With beam) were saved as ' + Save_Class)
        
        else: # ビームが無かったら
            
            # リングの各側室についてMSE計算に用いるデータを取り出す
            # Date_Range_DIFをファイルに保存する
            Mode_Para = 'DIF_Strg'
            Date_Range_DIF, Error_fds = Get_DIF_Strg_NB(List_Para, LER_CCG_n)

            # エラーがあったら、
            if(Error_fds != 'none'):
                tk.messagebox.showinfo('LER Error (No Beam)', Error_fds)

            # エラーが無かったら
            else:
                # 確認メッセージ
                Messagebox = tk.messagebox.askquestion('Confirmation', 'Save LER Strg data as ' + Save_Class +'. OK ?', 
                                                       icon='warning')
                if Messagebox == 'no':
                    return
            
                Save_Manual_Strg_NB(self, Method, Mode_Para, List_Para, Check_Record_Name, Save_Class, Abort_Timing)
            
                # 終了メッセージ
                tk.messagebox.showinfo('Information', 'LER Strg data (No beam) were saved as ' + Save_Class)
        
        return
    
    def LER_Save_Manual_Tail(self):
        # エラーフラグリセット
        Error_f = 'none'
    
        # Recordnameがnoneなら何もしない
        Check_Record_Name = self.LER_Record_Name_entry.get()
        if (Check_Record_Name == 'none'):
            tk.messagebox.showinfo('No record name', 'LER No record name')
            return
        
        # リングの名前
        Ring_Name = 'LER'
        List_Para = self.LER_CCG_List
        
        # Abort Timing
        Abort_Timing = self.LER_Abort_Time_entry.get()
    
        # モードの確認 autoかmanualか
        n = self.Mode_var.get()
        Run_Mode = self.Mode_rdo_text[n]
        
        # Auto時のモードの確認 Abort triggerかIntervalか
        m = self.Auto_Mode_var.get()
        Auto_Mode = self.Auto_Mode_rdo_text[m]
        
        # constモード(Triggerモード中のconstモード)
        Cns_Mode = self.LER_trg_cns
        
        # Auto_modeで走っているかどうか
        Arun = self.LER_auto_run
        
        # normalとしてsaveするか、abnormalとしてsaveするか
        n = self.LER_Save_Manual_Tail_Mode_var.get()
        Save_Class = self.LER_Save_Manual_Tail_Mode_rdo_text[n]
            
        # 時間間隔
        self.LER_Interval_h = self.Interval_entry.get()
        Hadv = float(self.LER_Interval_h)
    
        # Referenceデータの最終日
        self.LER_Last_Ref_d = self.Last_Ref_entry.get()
        Last_Refd = float(self.LER_Last_Ref_d)
    
        # Referenceデータの期間(日)
        self.LER_Ref_Period_d = self.Ref_Period_entry.get()
        Ref_Pd = float(self.LER_Ref_Period_d)
        
        # アボートタイミングから、各モードのDate_Range(データの時間範囲)を決める
        # ビームがあるかどうかも調べる。No_Beam = 1 ならビームなし(シャットダウン中とか)、0 ならビームあり
        No_Beam, Error_f = Define_Date_Range(Abort_Timing, List_Para, Run_Mode, Auto_Mode, Cns_Mode, Arun, Hadv, 
                                             Last_Refd, Ref_Pd)
        if(Error_f != 'none'):
            tk.messagebox.showinfo('LER Strg Error', Error_f)
            return
    
        # 選別する方法
        Method = 'FNN'  # FNN 回帰モデルの結果を使い自動で選別(Strgはi32)

        # アボート直前の蓄積中の圧力を調べ、
        # スタンダードの回帰曲線での予想とズレの大きなレコードをみつける(DIFモード)
        
        # 初期値 (Cn_Beam == 1ならビームがあってTailもあり)
        Cn_Beam = 0
    
        if(No_Beam == 0): # ビームがあったら
            Mode_Para = 'CHK_Tail'
            Date_Range_CHK, Error_fct = Get_Fit_CHK_Tail(Method, Check_Record_Name, Mode_Para, List_Para, Abort_Timing)
            
            # 20240207
            plt.clf()
            plt.close()
            
            # エラーがあったら、
            if(Error_fct != 'none') and (Error_fct != 'No Tail Data'):
                tk.messagebox.showinfo('LER CHK Tail Error', Error_fct)
                return
            
            # ビームが有って、Tail dataが無かったら
            elif(Error_fct == 'No Tail Data'):
                Cn_Beam = 1
               
        if(No_Beam == 0) and (Cn_Beam == 0): # BeamがあってTail があったら
            
            # リングの各側室についてMSE計算に用いるデータを取り出す
            # Date_Range_DIFをファイルに保存する
            Mode_Para = 'DIF_Tail'
            Date_Range_DIF, Error_fdt = Get_DIF_Tail(List_Para)

            # エラーがあったら、
            if(Error_fdt != 'none'):
                tk.messagebox.showinfo('LER Error (With Beam)', Error_fdt)

            # エラーが無かったら
            else:        
                # 確認メッセージ
                Messagebox = tk.messagebox.askquestion('Confirmation', 'Save LER Tail data as ' + Save_Class +'. OK ?', 
                                                       icon='warning')
                if Messagebox == 'no':
                    return
            
                Save_Manual_Tail(self, Method, Mode_Para, List_Para, Check_Record_Name, Save_Class, Abort_Timing)

                # 終了メッセージ
                tk.messagebox.showinfo('Information', 'LER Tail data (With beam) were saved as ' + Save_Class)
        else:
            # Tailデータがない
            tk.messagebox.showinfo('No Tail data!', 'LER No Tail data!')

        return
    
    def HER_Save_Manual_Strg(self):
        # エラーフラグリセット
        Error_f = 'none'
    
        # Recordnameがnoneなら何もしない
        Check_Record_Name = self.HER_Record_Name_entry.get()
        if (Check_Record_Name == 'none'):
            tk.messagebox.showinfo('No record name', 'HER No record name')
            return
        
        # リングの名前
        Ring_Name = 'HER'
        List_Para = self.HER_CCG_List
        
        # Abort Timing
        Abort_Timing = self.HER_Abort_Time_entry.get()
    
        # モードの確認 autoかmanualか
        n = self.Mode_var.get()
        Run_Mode = self.Mode_rdo_text[n]
        
        # Auto時のモードの確認 Abort triggerかIntervalか
        m = self.Auto_Mode_var.get()
        Auto_Mode = self.Auto_Mode_rdo_text[m]
        
        # constモード(Triggerモード中のconstモード)
        Cns_Mode = self.HER_trg_cns
        
        # Auto_modeで走っているかどうか
        Arun = self.HER_auto_run
        
        # normalとしてsaveするか、abnormalとしてsaveするか
        n = self.HER_Save_Manual_Strg_Mode_var.get()
        Save_Class = self.HER_Save_Manual_Strg_Mode_rdo_text[n]
            
        # 時間間隔
        self.HER_Interval_h = self.Interval_entry.get()
        Hadv = float(self.HER_Interval_h)
    
        # Referenceデータの最終日
        self.HER_Last_Ref_d = self.Last_Ref_entry.get()
        Last_Refd = float(self.HER_Last_Ref_d)
    
        # Referenceデータの期間(日)
        self.HER_Ref_Period_d = self.Ref_Period_entry.get()
        Ref_Pd = float(self.HER_Ref_Period_d)
        
        # アボートタイミングから、各モードのDate_Range(データの時間範囲)を決める
        # ビームがあるかどうかも調べる。No_Beam = 1 ならビームなし(シャットダウン中とか)、0 ならビームあり
        No_Beam, Error_f = Define_Date_Range(Abort_Timing, List_Para, Run_Mode, Auto_Mode, Cns_Mode, Arun, Hadv, 
                                             Last_Refd, Ref_Pd)
        if(Error_f != 'none'):
            tk.messagebox.showinfo('HER Strg Error', Error_f)
            return
    
        # 選別する方法
        Method = 'FNN'  # FNN 回帰モデルの結果を使い自動で選別(Strgはi32)

        # 各側室のCCGの数+1
        HER_CCG_n = [0] * 12
        for k in range(12):
            HER_CCG_n[k] = len(self.HER_Record_Name_List_box[k])
            
        # アボート直前の蓄積中の圧力を調べ、
        # スタンダードの回帰曲線での予想とズレの大きなレコードをみつける(DIFモード)
                                  
        if(No_Beam == 0): # ビームがあったら
        
            # リングの各側室についてMSE計算に用いるデータを取り出す
            # Date_Range_DIFをファイルに保存する
            Mode_Para = 'DIF_Strg'
            Date_Range_DIF, Error_fds = Get_DIF_Strg(List_Para, HER_CCG_n)

            # エラーがあったら、
            if(Error_fds != 'none'):
                tk.messagebox.showinfo('HER Error (With Beam)', Error_fds)

            # エラーが無かったら
            else:
                # 確認メッセージ
                Messagebox = tk.messagebox.askquestion('Confirmation', 'HER Save Strg data as ' + Save_Class +'. OK ?', 
                                                       icon='warning')
                if Messagebox == 'no':
                    return
            
                Save_Manual_Strg(self, Method, Mode_Para, List_Para, Check_Record_Name, Save_Class, Abort_Timing)

                # 終了メッセージ
                tk.messagebox.showinfo('Information', 'HER Strg data (With beam) were saved as ' + Save_Class)
        
        else: # ビームが無かったら
            
            # リングの各側室についてMSE計算に用いるデータを取り出す
            # Date_Range_DIFをファイルに保存する
            Mode_Para = 'DIF_Strg'
            Date_Range_DIF, Error_fds = Get_DIF_Strg_NB(List_Para, HER_CCG_n)

            # エラーがあったら、
            if(Error_fds != 'none'):
                tk.messagebox.showinfo('HER Error (No Beam)', Error_fds)

            # エラーが無かったら
            else:
                # 確認メッセージ
                Messagebox = tk.messagebox.askquestion('Confirmation', 'HER Save Strg data as ' + Save_Class +'. OK ?', 
                                                       icon='warning')
                if Messagebox == 'no':
                    return
            
                Save_Manual_Strg_NB(self, Method, Mode_Para, List_Para, Check_Record_Name, Save_Class, Abort_Timing)
            
                # 終了メッセージ
                tk.messagebox.showinfo('Information', 'HER Strg data (No beam) were saved as ' + Save_Class)
        
        return
    
    def HER_Save_Manual_Tail(self):
        # エラーフラグリセット
        Error_f = 'none'
    
        # Record nameがnoneなら何もしない
        Check_Record_Name = self.HER_Record_Name_entry.get()
        if (Check_Record_Name == 'none'):
            tk.messagebox.showinfo('No record name', 'HER No record name')
            return
        
        # リングの名前
        Ring_Name = 'HER'
        List_Para = self.HER_CCG_List
        
        # Abort Timing
        Abort_Timing = self.HER_Abort_Time_entry.get()
    
        # モードの確認 autoかmanualか
        n = self.Mode_var.get()
        Run_Mode = self.Mode_rdo_text[n]
        
        # Auto時のモードの確認 Abort triggerかIntervalか
        m = self.Auto_Mode_var.get()
        Auto_Mode = self.Auto_Mode_rdo_text[m]
        
        # constモード(Triggerモード中のconstモード)
        Cns_Mode = self.HER_trg_cns
        
        # Auto_modeで走っているかどうか
        Arun = self.HER_auto_run
        
        # normalとしてsaveするか、abnormalとしてsaveするか
        n = self.HER_Save_Manual_Tail_Mode_var.get()
        Save_Class = self.HER_Save_Manual_Tail_Mode_rdo_text[n]
            
        # 時間間隔
        self.HER_Interval_h = self.Interval_entry.get()
        Hadv = float(self.HER_Interval_h)
    
        # Referenceデータの最終日
        self.HER_Last_Ref_d = self.Last_Ref_entry.get()
        Last_Refd = float(self.HER_Last_Ref_d)
    
        # Referenceデータの期間(日)
        self.HER_Ref_Period_d = self.Ref_Period_entry.get()
        Ref_Pd = float(self.HER_Ref_Period_d)
        
        # アボートタイミングから、各モードのDate_Range(データの時間範囲)を決める
        # ビームがあるかどうかも調べる。No_Beam = 1 ならビームなし(シャットダウン中とか)、0 ならビームあり
        No_Beam, Error_f = Define_Date_Range(Abort_Timing, List_Para, Run_Mode, Auto_Mode, Cns_Mode, Arun, Hadv, 
                                             Last_Refd, Ref_Pd)
        if(Error_f != 'none'):
            tk.messagebox.showinfo('HER Strg Error', Error_f)
            return
    
        # 選別する方法
        Method = 'FNN'  # FNN 回帰モデルの結果を使い自動で選別(Strgはi32)

        # アボート直前の蓄積中の圧力を調べ、
        # スタンダードの回帰曲線での予想とズレの大きなレコードをみつける(DIFモード)
        
        # 初期値 (Cn_Beam == 1ならビームがあってTailもあり)
        Cn_Beam = 0
    
        if(No_Beam == 0): # ビームがあったら
            Mode_Para = 'CHK_Tail'
            Date_Range_CHK, Error_fct = Get_Fit_CHK_Tail(Method, Check_Record_Name, Mode_Para, List_Para, Abort_Timing)
        
            # 20240207
            plt.clf()
            plt.close()
            
            # エラーがあったら、
            if(Error_fct != 'none') and (Error_fct != 'No Tail Data'):
                tk.messagebox.showinfo('HER CHK Tail Error', Error_fct)
                return
            
            # ビームが有って、Tail dataが無かったら
            elif(Error_fct == 'No Tail Data'):
                Cn_Beam = 1
               
        if(No_Beam == 0) and (Cn_Beam == 0): # BeamがあってTail があったら
            
            # リングの各側室についてMSE計算に用いるデータを取り出す
            # Date_Range_DIFをファイルに保存する
            Mode_Para = 'DIF_Tail'
            Date_Range_DIF, Error_fdt = Get_DIF_Tail(List_Para)

            # エラーがあったら、
            if(Error_fdt != 'none'):
                tk.messagebox.showinfo('HER Error (With Beam)', Error_fdt)

            # エラーが無かったら
            else:        
                # 確認メッセージ
                Messagebox = tk.messagebox.askquestion('Confirmation', 'Save HER Tail data as ' + Save_Class +'. OK ?', 
                                                       icon='warning')
                if Messagebox == 'no':
                    return
            
                Save_Manual_Tail(self, Method, Mode_Para, List_Para, Check_Record_Name, Save_Class, Abort_Timing)

                # 終了メッセージ
                tk.messagebox.showinfo('Information', 'HER Tail data (With beam) were saved as ' + Save_Class)
        else:
            # Tailデータがない
            tk.messagebox.showinfo('No Tail data!', 'HER No Tail data!')

        return
    
    # LERのアボートトリガー
    def LERtriggerEvent(self, pvname = None, value = None, **kw):
        # 最初のコールは無視
        if(self.LER_tc == 0):
            self.LER_tc = 1
            return
        
        # 値がゼロになったのならなにもしない
        if(value == 0):
            self.loglist.insert(tk.END, 'Trig received: ' + str(value))
            self.loglist.itemconfigure(tk.END, foreground = 'red')
            self.loglist.see(tk.END)
            return
        
        # モードの確認 auto かmanualか
        n = self.Mode_var.get()
        Run_Mode = self.Mode_rdo_text[n]
        
        # Auto時のモードの確認　Abort triggerかIntervalか
        m = self.Auto_Mode_var.get()
        Auto_Mode = self.Auto_Mode_rdo_text[m]
        
        # Autoモードで、Triggerモードだったら
        if(Run_Mode == 'Auto') and (Auto_Mode == 'Abt Trg'):
            
            # 20240131追加
            # 前回実行から30分未満なら実行しない。
            self.PresentTime = datetime.datetime.now()
            td = self.PresentTime - self.LER_LastRunTime
            # 20240206変更
            if(td.seconds < 30 * 60) and (self.LER_trg_cns == 0):
                # 20240207追加
                self.Now_n = datetime.datetime.now()
                self.Now_t = self.Now_n.strftime('%Y-%m-%d %H:%M:%S')
                self.loglist.insert(tk.END, 'Trigger recieved:')
                self.loglist.itemconfigure(tk.END, foreground = 'red')
                self.loglist.insert(tk.END, '  ' + self.Now_t[5:])
                # self.loglist.see(tk.END)
                # print(11708, 'LER Last run is within 30 minuits.')
                self.loglist.insert(tk.END, 'But Trig. < 30 min.')
                self.loglist.itemconfigure(tk.END, foreground = 'red')
                self.loglist.see(tk.END)
                
                return
        
            # 3分後に実行する(LER)
            delaytime = int(3 * 60 * 1000)
            self.LERid = self.after(delaytime, self.LERtriggerUpdate)
            
            # 現在時刻をAbort timingとする (LER)
            LER_Now = datetime.datetime.now()
            self.LER_Last_Date = LER_Now.strftime('%Y-%m-%d %H:%M:%S')
            self.LER_Abort_Time_entry.delete(0, tk.END)
            self.LER_Abort_Time_entry.insert(0, self.LER_Last_Date)
            
            # レコード名を指定 (LER)
            if(self.LER_Record_Name_entry.get() == 'none'):        
                Abnormal_Result_Strg_File_Name = 'LER_FNN_Abnormal_Class2_Result_Strg_WB.npy'
                Abnormal_Result_Strg_Text_File_Name = 'LER_FNN_Abnormal_Class2_Result_Strg_WB.txt'
                Abnormal_Result_Strg_List = Check_Strg_Abnormal_Result_File(Abnormal_Result_Strg_File_Name, 
                                                                   Abnormal_Result_Strg_Text_File_Name)
                if(Abnormal_Result_Strg_List[-1, 3] != 'Record Name'):
                    self.LER_Record_Name_entry.delete(0, tk.END)
                    self.LER_Record_Name_entry.insert(0, Abnormal_Result_Strg_List[-1, 3])
                    print(12052, Abnormal_Result_Strg_List[-1, 3])
        
            # loglistに記録
            self.loglist.insert(tk.END, 'Trig received: ' + str(value))
            self.loglist.itemconfigure(tk.END, foreground = 'red')
            self.loglist.insert(tk.END, '  ' + self.LER_Last_Date[5:])
            self.loglist.insert(tk.END, 'Start in 3 min')
            self.loglist.itemconfigure(tk.END, foreground = 'red')
            self.loglist.see(tk.END)
            
            # Autoモードラベル変更
            self.Status_LER_Auto_label["text"] = "LER Auto Running"
            self.Status_LER_Auto_label["fg"] = "red"
            self.Status_LER_Auto_label.update()
            
            # スタートボタンを無効化
            self.Run_Start_btn["state"] = "disable"
            
            # 状態表示変更
            self.LER_ABN_canvas["highlightthickness"] = 5
            self.LER_ABN_canvas["highlightbackground"] = 'lightgreen'
            self.LER_ABN_canvas.update()
            
        # Manualモードだったら
        if(Run_Mode == 'Manual'):
            # loglistに記録して何もしない
            self.loglist.insert(tk.END, 'Trig received: ' + str(value))
            self.loglist.itemconfigure(tk.END, foreground = 'red')
            self.loglist.see(tk.END)
            return
        
        # Autoモードで、Cons Intrモードだったら
        if(Run_Mode == 'Auto') and (Auto_Mode == 'Cns Intr'):
            # loglistに記録して何もしない
            self.loglist.insert(tk.END, 'Trig received: ' + str(value))
            self.loglist.itemconfigure(tk.END, foreground = 'red')
            self.loglist.see(tk.END)
            return          
            
        self.Status_change_waiting()
    
    # HERのアボートトリガー
    def HERtriggerEvent(self, pvname = None, value = None, **kw):
        # 最初のコールは無視
        if(self.HER_tc == 0):
            self.HER_tc = 1
            return
        
        # 値がゼロになったのならなにもしない
        if(value == 0):
            self.loglist.insert(tk.END, 'Trig received: ' + str(value))
            self.loglist.itemconfigure(tk.END, foreground = 'blue')
            self.loglist.see(tk.END)
            return
        
        # モードの確認 auto かmanualか
        n = self.Mode_var.get()
        Run_Mode = self.Mode_rdo_text[n]
        
        # Auto時のモードの確認　Abort triggerかIntervalか
        m = self.Auto_Mode_var.get()
        Auto_Mode = self.Auto_Mode_rdo_text[m]
        
        # Autoモードで、Triggerモードだったら
        if(Run_Mode == 'Auto') and (Auto_Mode == 'Abt Trg'):
            
            # 20240131追加
            # 前回実行から30分未満なら実行しない。
            self.PresentTime = datetime.datetime.now()
            td = self.PresentTime - self.HER_LastRunTime
            # 20240206変更
            if(td.seconds < 30 * 60) and (self.HER_trg_cns == 0):
                self.Now_n = datetime.datetime.now()
                self.Now_t = self.Now_n.strftime('%Y-%m-%d %H:%M:%S')
                self.loglist.insert(tk.END, 'Trigger recieved:')
                self.loglist.itemconfigure(tk.END, foreground = 'blue')
                self.loglist.insert(tk.END, '  ' + self.Now_t[5:])
                # self.loglist.see(tk.END)
                # print(11708, 'HER Last run is within 30 minuits.')
                self.loglist.insert(tk.END, 'But Trig. < 30 min.')
                self.loglist.itemconfigure(tk.END, foreground = 'blue')
                self.loglist.see(tk.END)
                # 20240207追加
                
                return
                                                 
            # 3分後に実行する(HER)
            delaytime = int(3 * 60 * 1000)
            self.HERid = self.after(delaytime, self.HERtriggerUpdate)
            
            # 現在時刻をAbort timingとする (HER)
            HER_Now = datetime.datetime.now()
            self.HER_Last_Date = HER_Now.strftime('%Y-%m-%d %H:%M:%S')
            self.HER_Abort_Time_entry.delete(0, tk.END)
            self.HER_Abort_Time_entry.insert(0, self.HER_Last_Date)
            
            # レコード名を指定 (HER)
            if(self.HER_Record_Name_entry.get() == 'none'):
                Abnormal_Result_Strg_File_Name = 'HER_FNN_Abnormal_Class2_Result_Strg_WB.npy'
                Abnormal_Result_Strg_Text_File_Name = 'HER_FNN_Abnormal_Class2_Result_Strg_WB.txt'
                Abnormal_Result_Strg_List = Check_Strg_Abnormal_Result_File(Abnormal_Result_Strg_File_Name, 
                                                                   Abnormal_Result_Strg_Text_File_Name)
                if(Abnormal_Result_Strg_List[-1, 3] != 'Record Name'):
                    self.HER_Record_Name_entry.delete(0, tk.END)
                    self.HER_Record_Name_entry.insert(0, Abnormal_Result_Strg_List[-1, 3])         
                    print(12149, Abnormal_Result_Strg_List[-1, 3])
                
            # loglistに記録
            self.loglist.insert(tk.END, 'Trig received: ' + str(value))
            self.loglist.itemconfigure(tk.END, foreground = 'blue')
            self.loglist.insert(tk.END, '  ' + self.HER_Last_Date[5:])
            self.loglist.insert(tk.END, 'Start in 3 min')
            self.loglist.itemconfigure(tk.END, foreground = 'blue')
            self.loglist.see(tk.END)
            
            # Autoモードラベル変更
            self.Status_HER_Auto_label["text"] = "HER Auto Running"
            self.Status_HER_Auto_label["fg"] = "red"
            self.Status_HER_Auto_label.update()
            
            # スタートボタンを無効化
            self.Run_Start_btn["state"] = "disable"
            
            # 状態表示変更
            self.HER_ABN_canvas["highlightthickness"] = 5
            self.HER_ABN_canvas["highlightbackground"] = 'lightgreen'
            self.HER_ABN_canvas.update()
            
        # Manualモードだったら
        if(Run_Mode == 'Manual'):
            # loglistに記録して何もしない
            self.loglist.insert(tk.END, 'Trig received: ' + str(value))
            self.loglist.itemconfigure(tk.END, foreground = 'blue')
            self.loglist.see(tk.END)
            return
        
        # Autoモードで、Cons Intrモードだったら
        if(Run_Mode == 'Auto') and (Auto_Mode == 'Cns Intr'):
            # loglistに記録して何もしない
            self.loglist.insert(tk.END, 'Trig received: ' + str(value))
            self.loglist.itemconfigure(tk.END, foreground = 'blue')
            self.loglist.see(tk.END)
            return          
            
        self.Status_change_waiting()
    
    # LER Triggerのスレッドの処理
    def LERtriggerUpdate(self):
        # ただのtriggerモード
        self.LER_trg_cns = 0
        thread_LER = threading.Thread(target = Main_Command_LER(self))
        
        # LERスタート
        thread_LER.start()
        
        # 20240206追加　実行した時間
        self.LER_LastRunTime = datetime.datetime.now()
        
        # loglistに記録
        self.Now_n = datetime.datetime.now()
        self.Now_t = self.Now_n.strftime('%Y-%m-%d %H:%M:%S')
        self.loglist.insert(tk.END, 'Trigger start:')
        self.loglist.itemconfigure(tk.END, foreground = 'red')
        self.loglist.insert(tk.END, '  ' + self.Now_t[5:])
        self.loglist.see(tk.END)
        
        # loglist書き込み
        self.Now_n = datetime.datetime.now()
        self.LER_Next_Date_n = self.Now_n + datetime.timedelta(minutes = int(float(self.LER_Interval_h) * 60))
        self.LER_Next_Date_t = self.LER_Next_Date_n.strftime('%Y-%m-%d %H:%M:%S')
        self.loglist.insert(tk.END, "Trigger Next:")
        self.loglist.itemconfigure(tk.END, foreground = 'red')
        self.loglist.insert(tk.END, '  ' + self.LER_Next_Date_t[5:])
        self.loglist.see(tk.END)
        
        self.Status_LER_Auto_label["text"] = "LER Auto Running"
        self.Status_LER_Auto_label["fg"] = "red"
        self.Status_LER_Auto_label.update()
        
        # 表示変更
        self.Status_change_waiting()
        # スタートボタンを有効化
        self.Run_Start_btn["state"] = "active"
    
    # HER Triggerのスレッドの処理
    def HERtriggerUpdate(self):
        # ただのtriggerモード
        self.HER_trg_cns = 0
        thread_HER = threading.Thread(target = Main_Command_HER(self))
        
        # HERスタート
        thread_HER.start()
        
        # 20240206追加　実行した時間
        self.HER_LastRunTime = datetime.datetime.now()
        
        # loglistに記録
        self.Now_n = datetime.datetime.now()
        self.Now_t = self.Now_n.strftime('%Y-%m-%d %H:%M:%S')
        self.loglist.insert(tk.END, 'Trigger start:')
        self.loglist.itemconfigure(tk.END, foreground = 'blue')
        self.loglist.insert(tk.END, '  ' + self.Now_t[5:])
        self.loglist.see(tk.END)
        
        # loglist書き込み
        self.Now_n = datetime.datetime.now()
        self.HER_Next_Date_n = self.Now_n + datetime.timedelta(minutes = int(float(self.HER_Interval_h) * 60))
        self.HER_Next_Date_t = self.HER_Next_Date_n.strftime('%Y-%m-%d %H:%M:%S')
        self.loglist.insert(tk.END, "Trigger Next:")
        self.loglist.itemconfigure(tk.END, foreground = 'blue')
        self.loglist.insert(tk.END, '  ' + self.HER_Next_Date_t[5:])
        self.loglist.see(tk.END)
        
        self.Status_HER_Auto_label["text"] = "HER Auto Running"
        self.Status_HER_Auto_label["fg"] = "red"
        self.Status_HER_Auto_label.update()
            
        # 表示変更
        self.Status_change_waiting()
        # スタートボタンを有効化
        self.Run_Start_btn["state"] = "active"
        
    # LERアボートタイムリストのリセット
    def LER_abt_reset(self):
        Messagebox = tk.messagebox.askquestion('Reset LER abort list','     Really OK ?     ', icon='warning')
        if Messagebox == 'yes': 
            self.LER_Abort_Time_List_box = ['none']
    
            # リストボックスクリア
            self.LER_Abort_Time_listbox.delete(0, tk.END)
            for item in self.LER_Abort_Time_List_box:
                self.LER_Abort_Time_listbox.insert(tk.END, item)
        
            # アボートタイミングリストを保存する
            Array_to_Save = self.LER_Abort_Time_List_box
            # arrayで保存するファイル名
            Array_File_Name_to_Save = 'LER_Abort_Time_List.npy'
            # textで保存するファイル名
            Text_File_Name_to_Save = 'LER_Abort_Time_List.txt'
            # 保存する関数
            Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
            
            del Array_to_Save
            
            # FNNで分類したAbnormal/Normalの結果を削除する。
            path_list = glob.glob('LER_FNN_*ormal_Class2_Result_Strg_NB.*')
                
            if(len(path_list) > 0):
                for n in path_list:
                    command_d = 'rm ' + n
                    try:
                        subprocess.run(command_d, shell = True)
                    except Exception as e:
                        print("subprocess.check_call() failed", command_d)
                        print(str(e))
    
            path_list = glob.glob('LER_FNN_*ormal_Class2_Result_Tail*.*')
            
            if(len(path_list) > 0):
                for n in path_list:
                    command_d = 'rm ' + n
                    try:
                        subprocess.run(command_d, shell = True)
                    except Exception as e:
                        print("subprocess.check_call() failed", command_d)
                        print(str(e))
    
            print('LER Abort timing list and FNN abnormal/normal NB list were deleted.', 2 * len(path_list))
        
            path_list = glob.glob('LER_FNN_*ormal_Class2_Result_Strg_WB.*')
                
            if(len(path_list) > 0):
                for n in path_list:
                    command_d = 'rm ' + n
                    try:
                        subprocess.run(command_d, shell = True)
                    except Exception as e:
                        print("subprocess.check_call() failed", command_d)
                        print(str(e))
    
            print('LER Abort timing list and FNN abnormal/normal WB list were deleted.', 2 * len(path_list))
        
            return
        else:
            return  
    
    # HERアボートタイムリストのリセット
    def HER_abt_reset(self):
        Messagebox = tk.messagebox.askquestion('Reset HER abort list','    Really OK ?    ', icon='warning')
        if Messagebox == 'yes': 
            self.HER_Abort_Time_List_box = ['none']
    
            # リストボックスクリア
            self.HER_Abort_Time_listbox.delete(0, tk.END)
            for item in self.HER_Abort_Time_List_box:
                self.HER_Abort_Time_listbox.insert(tk.END, item)
        
            # アボートタイミングリストを保存する
            Array_to_Save = self.HER_Abort_Time_List_box
            # arrayで保存するファイル名
            Array_File_Name_to_Save = 'HER_Abort_Time_List.npy'
            # textで保存するファイル名
            Text_File_Name_to_Save = 'HER_Abort_Time_List.txt'
            # 保存する関数
            Result_File_Save(Array_File_Name_to_Save, Text_File_Name_to_Save, Array_to_Save)
            
            del Array_to_Save
            
            # FNNで分類したAbnormal/Normalの結果を削除する。
            path_list = glob.glob('HER_FNN_*ormal_Class2_Result_Strg_NB.*')
            
            if(len(path_list) > 0):
                for n in path_list:
                    command_d = 'rm ' + n
                    try:
                        subprocess.run(command_d, shell = True)
                    except Exception as e:
                        print("subprocess.check_call() failed", command_d)
                        print(str(e))
    
            path_list = glob.glob('HER_FNN_*ormal_Class2_Result_Tail*.*')
            
            if(len(path_list) > 0):
                for n in path_list:
                    command_d = 'rm ' + n
                    try:
                        subprocess.run(command_d, shell = True)
                    except Exception as e:
                        print("subprocess.check_call() failed", command_d)
                        print(str(e))
    
            print('HER Abort timing list and FNN abnormal/normal list were deleted.', 2 * len(path_list))
        
            path_list = glob.glob('HER_FNN_*ormal_Class2_Result_Strg_WB.*')
                
            if(len(path_list) > 0):
                for n in path_list:
                    command_d = 'rm ' + n
                    try:
                        subprocess.run(command_d, shell = True)
                    except Exception as e:
                        print("subprocess.check_call() failed", command_d)
                        print(str(e))
    
            print('HER Abort timing list and FNN abnormal/normal list were deleted.', 2 * len(path_list))
        
            return
        else:
            return
    
    # ウィンドウを閉じる    
    def Close_btn_click(self):
        self.master.destroy()
    
    # サブWindowを取得する関数
    def getSubWindow(self, Cause_List, wtitle):
        # メインWindowに紐づくサブWindowを作成する。
        subWindow = tk.Toplevel()
        subWindow.grab_set() # 親ウィンドウの操作をできなくする

        # サブWindowへタイトルをつける。
        subWindow.title(wtitle)

        # Window(サブWindow)の画面サイズを設定する。
        subWindow.geometry("450x200+500+300")
        
        self.Cau_label = tk.Label(subWindow, text = 'Select possible cause', font = ('Times new roman', 10))
        self.Cau_label.place(x = 30, y = 20)
        
        # 原因のチェック有無変数
        self.Cau_var = tk.IntVar()
        # value = 0のラジオボタンにチェックを入れる
        self.Cau_var.set(0)
        # テキストのリスト
        self.Cau_rdo_text = Cause_List
        # ラジオボタン作成
        self.Cau_rdo_1 = tk.Radiobutton(subWindow, value = 0, variable = self.Cau_var,
                                         text = self.Cau_rdo_text[0], font = ('Times new roman', 10), fg = 'red')
        self.Cau_rdo_1.place(x = 60, y = 40)
        self.Cau_rdo_2 = tk.Radiobutton(subWindow, value = 1, variable = self.Cau_var,
                                         text = self.Cau_rdo_text[1], font = ('Times new roman', 10), fg = 'red')
        self.Cau_rdo_2.place(x = 60, y = 60)
        self.Cau_rdo_3 = tk.Radiobutton(subWindow, value = 2, variable = self.Cau_var,
                                         text = self.Cau_rdo_text[2], font = ('Times new roman', 10), fg = 'red')
        self.Cau_rdo_3.place(x = 60, y = 80)
        
        # Ok and clseボタン
        self.Man_Ok_btn = tk.Button(subWindow, text = 'Ok and Close', bg = 'lightblue', command = subWindow.destroy)
        self.Man_Ok_btn.place(x = 120, y = 160 )

        return subWindow
        
    def main():
        root = tk.Tk()
        app = Application(master = root)
        app.mainloop()

if __name__ == "__main__":
    # main()
    root = tk.Tk()
    app = Application(master = root)
    app.mainloop()
    
    


# ### 

# In[ ]:





# In[ ]:




