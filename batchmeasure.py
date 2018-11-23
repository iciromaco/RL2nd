import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
from PIL import Image
import math
import pandas as pd

from sympy import *
from sympy.abc import a,b,c
#init_session()
px,py =var('px:4'),var('py:4')
t = symbols('t')
from sympy import var
from rdlib2 import *

import datetime
import time

UNIT = 256

CONTOURS_APPROX = 0.005 # 輪郭近似精度
HARRIS_PARA = 1.0 # ハリスコーナー検出で、コーナーとみなすコーナーらしさの指標  1.0 なら最大値のみ
CONTOURS_APPROX = 0.0001 # 輪郭近似精度
SHRINK = 0.8 # 0.75 # 収縮膨張で形状を整える時のパラメータ
GAUSSIAN_RATE1= 0.2 # 先端位置を決める際に使うガウスぼかしの程度を決める係数
GAUSSIAN_RATE2 = 0.1 # 仕上げに形状を整えるためのガウスぼかしの程度を決める係数

# バッチ司令ファイルの読み込み
df = pd.read_excel('自動計測データ.xlsx')
# df = pd.read_csv('画像リストUTF8.csv', sep=',')
   
# ベジエ曲線あてはめ、仮中心線の抽出
def preGetLRdata(img,tlevel = 10, blevel=90,bracket=1):
    # ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #  輪郭を抽出
    _img,contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # バウンダリ矩形を得る
    x0,y0,w,h = cv2.boundingRect(img)
    cnt = contours[0] # 白領域は１つしかないという前提なので、０番の輪郭が大根の輪郭である。
    
    # 閉じた輪郭線の上下を削り、左右２本の輪郭に分割する
    canvas = np.zeros_like(img)  # 描画キャンバスの準備
    canvas = cv2.drawContours(canvas, contours, -1, 255, thickness=1)     # 輪郭線の描画
    # 上下端それぞれ10％をカットする。上下は歪みが大きいのでノイズとなるので 削除するとともに、それにより輪郭を左右分割する。
    cutHead= y0+int(tlevel*h/100) # シルエッ上端から指定％の高さ
    cutBottom = y0+int(blevel*h/100) # シルエッ下端から指定％の高さ
    canvas[0:cutHead,:]=np.zeros((cutHead,img.shape[1])) # 上5%をマスク
    canvas[cutBottom+1:,:]=np.zeros((img.shape[0]-(cutBottom+1),img.shape[1]))  # 下5%をマスク
    cntl,cntr = segmentLR0(canvas,bracket=bracket)
    return cntl,cntr,cnt
    
# 左右セグメントを含む画像から左右の輪郭をえる
def segmentLR0(img,bracket=2):
    # bracket    2 : cv2 の輪郭データそのまま（２重カッコ）、 1: カッコを１つ外したリストを返す
    # 輪郭検出すれば２つの輪郭が見つかるはず。
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
          
    # 線図形の輪郭は中間で折り返しになっている
    cnt0 = contours[0][:int(len(contours[0])/2+1)]
    cnt1 = contours[1][:int(len(contours[1])/2+1)]
    # 中程の点を比べて左にある方を左と判定する。
    c0 = cnt0[int(len(cnt0)/2)][0][0]
    c1 = cnt1[int(len(cnt1)/2)][0][0]
    if  c0 > c1: 
        cntl,cntr = cnt1,cnt0
    else:
        cntr,cntl = cnt1,cnt0
        
    def bracket2to1(cnt):    
        cnt = np.array([[x,y] for [[x,y]] in cnt])
        return cnt
    
    if bracket == 2:
        return cntl,cntr
    else:
        return bracket2to1(cntl),bracket2to1(cntr)

# 左右の輪郭点をベジエ近似する
def cntPair2bez(cntl,cntr,N=3,n_samples=20,precPara=0.01, samplemode = 0, openmode=False, debugmode=False):

    # 輪郭点を（チェインの並び順に）等間隔に n_samples 個サンプリングする。
    cntL = cntl[np.array(list(map(int,np.linspace(0, len(cntl)-1,n_samples))))]
    cntR = cntr[np.array(list(map(int,np.linspace(0, len(cntr)-1,n_samples))))]
    
    if samplemode == 2: # 予想最大径位置より下はサンプリング間隔を２倍にするモード
        dlist = np.array([np.sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)) for [x0,y0],[x1,y1] in zip(cntL,cntR)])
        dmax_index = np.argmax(dlist) # 左右で一番離れている場所付近
        original_indexL = np.array(list(map(int,np.linspace(0, len(cntl)-1,n_samples))))[dmax_index] # その右輪郭での順位
        original_indexR = np.array(list(map(int,np.linspace(0, len(cntr)-1,n_samples))))[dmax_index] # その左輪郭での順位
        cntL = np.r_[cntL[0:dmax_index],cntl[np.array(list(map(int,np.linspace(original_indexL, len(cntl)-1,2*(n_samples-dmax_index)))))]]
        cntR = np.r_[cntR[0:dmax_index],cntr[np.array(list(map(int,np.linspace(original_indexR, len(cntr)-1,2*(n_samples-dmax_index)))))]]
    
    # 左右をそれぞれベジエ 曲線で近似し、その平均として中心軸を仮決定
    datal = cpxl,cpyl,bezXl,bezYl,tpl = fitBezierCurveN(cntL,precPara=precPara,N=N,openmode=openmode,debugmode=debugmode)
    datar = cpxr,cpyr,bezXr,bezYr,tpr = fitBezierCurveN(cntR,precPara=precPara,N=N,openmode=openmode,debugmode=debugmode)
    bezXc,bezYc = (bezXl+bezXr)/2,(bezYl+bezYr)/2
    cpl,cpr,cpc = (cpxl,cpyl),(cpxr,cpyr),((cpxl+cpxr)/2,(cpyl+cpyr)/2)
    bezL,bezR,bezC = (bezXl,bezYl),(bezXr,bezYr),(bezXc,bezYc)
    return cpl,cpr,cpc, bezL,bezR,bezC,cntL,cntR
    
# 結果の描画
def drawBez2(savepath,img,bezL=None,bezR=None,bezC=None,cpl=None,cpr=None,cpc=None, 
             cntL=[],cntR=[],cntC=None, ladder=None,PosL=[],PosR=[],PosC=[],n_samples=20,saveImage=False):
    bezXl,bezYl = bezL if bezL != None else ([],[])
    bezXr,bezYr = bezR if bezR != None else ([],[])
    bezXc,bezYc = bezC if bezC != None else ([],[])
    cpxl,cpyl = cpl if cpl != None else ([],[])
    cpxr,cpyr = cpr if cpr != None else ([],[])
    cpxc,cpyc = cpc if cpc != None else ([],[])
    tplins50 = np.linspace(0, 1, 50)
    tplinsSP = np.linspace(0, 1, n_samples)
    
    plt.figure(figsize=(6,6),dpi=100)
    plt.gca().invert_yaxis() 
    plt.gca().set_aspect('equal', adjustable='box') # アスペクト比を１：１に
    plt.imshow(192+(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)/4).astype(np.uint8))
    # 左輪郭の描画
    if bezL != None:
        plotx = [bezXl.subs(t,tp) for tp in tplins50 ]
        ploty = [bezYl.subs(t,tp) for tp in tplins50 ]
        plt.plot(plotx,ploty,color = 'red')  
    if len(cntL) >0:
        plt.scatter(cntL[:,0],cntL[:,1],color ='blue',marker = '.') #  サンプル点
    if cpl != None:
        plt.scatter(cpxl,cpyl,color ='purple',marker = '*') #  制御点の描画
        for i in range(len(cpxl)) : plt.annotate(str(i),(cpxl[i],cpyl[i]))
    # 右輪郭の描画
    if bezR != None:
        plotx = [bezXr.subs(t,tp) for tp in tplins50 ]
        ploty = [bezYr.subs(t,tp) for tp in tplins50 ]
        plt.plot(plotx,ploty,color = 'red')  
    if len(cntR)  > 0:
        plt.scatter(cntR[:,0],cntR[:,1],color ='blue',marker = '.') #  サンプル点
    if cpr != None:
        plt.scatter(cpxr,cpyr,color ='red',marker = '*') #  制御点の描画
        for i in range(len(cpxr)):plt.annotate(str(i),(cpxr[i],cpyr[i]))
    # 中心軸の描画
    if bezC != None:
        plotx = [bezXc.subs(t,tp) for tp in tplins50 ]
        ploty = [bezYc.subs(t,tp) for tp in tplins50 ]
        plt.plot(plotx,ploty,color = 'red')  
        if cntC != None:
            plt.scatter(cntC[:,0],cntC[:,1],color ='blue',marker = '.') #  サンプル点
        if cpc != None:
            plt.scatter(cpxc,cpyc,color ='darkgreen',marker = '*') #  制御点の描画
            for i in range(len(cpxc)):plt.annotate(str(i),(cpxc[i],cpyc[i]))
                
        # ラダーの描画
        if  ladder== 'lr':  # 左右の同じパラメータ値の点を結ぶだけ
            plotSPlx = [bezXl.subs(t,tp) for tp in tplinsSP ]
            plotSPly = [bezYl.subs(t,tp) for tp in tplinsSP ]
            plotSPrx = [bezXr.subs(t,tp) for tp in tplinsSP ]
            plotSPry = [bezYr.subs(t,tp) for tp in tplinsSP ]       
            for x0,x1,y0,y1 in zip(plotSPlx,plotSPrx,plotSPly,plotSPry):
                plt.plot([x0,x1],[y0,y1],color = 'orange') 
                
        elif ladder == 'normal':
            # 中心軸上に設定したサンプル点における法線と両輪郭の交点のリストを求める。
            plot20lx = [xl if xl !=np.inf else [] for [xl,yl] in PosL ]
            plot20ly = [yl if yl !=np.inf else [] for [xl,yl] in PosL]
            #plot20cx = [bezXc.subs(t,tp) for tp in np.linspace(0, 1, n_samples) ]
            #plot20cy = [bezYc.subs(t,tp) for tp in np.linspace(0, 1, n_samples) ]
            plot20cx = PosC[:,0]
            plot20cy = PosC[:,1]
            plot20rx = [xr if xr !=np.inf else [] for [xr,yr] in PosR ]
            plot20ry = [yr if yr !=np.inf else [] for [xr,yr] in PosR ]
                  
            for x0,x1,y0,y1 in zip(plot20lx,plot20cx,plot20ly,plot20cy):
                if x0 != [] and y0 !=[]:
                    plt.plot([x0,x1],[y0,y1],color = 'orange') 
            for x0,x1,y0,y1 in zip(plot20rx,plot20cx,plot20ry,plot20cy):
                if x0 != [] and y0 !=[]:
                    plt.plot([x0,x1],[y0,y1],color = 'orange') 
            if saveImage:
                pltsaveimage(savepath,'RAD')
    
# 中心軸ベジエをもとにそれに輪郭点を左右に分割する
def reGetCntPairOLD(img,cnt,cpl,cpr,bezC,CAPCUT=0,TAILCUT=0):          
    xLu,xRu,yLu,yRu = cpl[0][0],cpr[0][0],cpl[0][1],cpr[0][1] # 近似曲線の上端の座標
    xLb,xRb,yLb,yRb = cpl[0][0],cpr[0][0],cpl[0][1],cpr[0][1] # 近似曲線の上端の座標
    bezXc,bezYc = bezC
        
    # 輪郭線の描画
    canvas = np.zeros_like(img)
    canvas = cv2.drawContours(canvas, cnt, -1, 255, thickness=1)
          
    # 軸と輪郭の交点
    (crpx0,crpy0),(crpx1,crpy1) = crossPoints(img,cnt,bezC)
    if crpy0 > crpy1: 
        crpx0,crpy0,cpyx1,crpy1 = crpx1,crpy1,crpx0,crpy0
    
    # 中心軸の延長で上端から最大径離れた地点を中心に最大直径より少し大きな円を０で描き輪郭を削る。
    dy = float((diff(bezYc,t)).subs(t,0.1))
    dx = float((diff(bezXc,t)).subs(t,0.1)) # t=0 は境界なので変な値にあることがあるため 0.1 としている
    dd = np.sqrt(dx*dx+dy*dy)
    dkusabi = (xRu-xLu)*abs(dx)/dd/2
    acc = dy/dx if dx != 0 else np.inf # 中心軸の傾き
    x00 = bezXc.subs(t,0) #  軸の再上端
    y00 = bezYc.subs(t,0)
    ddd = sqrt((crpx0-x00)**2+(crpy0-y00)**2) # ベジエ軸上端と輪郭上端の距離
    #dddd = 5*(ddd-dkusabi)**2*((ddd-dkusabi)*dy/dd/(xRu-xLu) -0.1)
    dddd = -10 if ddd/(xRu-xLu) < 0.2 else  ddd-dkusabi 
    if CAPCUT != 0: # 特別指定された削除調整量がある場合
        dddd = -CAPCUT
    diaMinus =  1024
    xdd = diaMinus*dx/sqrt(dx**2+dy**2) #  1024離れるためのX移動量
    dia_U = diaMinus-dddd # 
    x11_U = x00-xdd
    y11_U =  y00-xdd*acc if acc != np.inf else y00-diaMinus    
    distO2top = np.sqrt(float(((crpy0-y11_U)**2 + (crpx0-x11_U)**2))) - dia_U  
    if  distO2top > 0 : # 削除中心と輪郭頂点の距離が削除半径より遠い(ということは削れない)ならその分以上に長くする
        dia_U += distO2top+5
    #canvas =  cv2.circle(canvas,(int(x11),int(y11)),int(dia_U),0,-1) # 黒で円を描いて削る
    # 同様に下端を削る
    dy = float((diff(bezYc,t)).subs(t,0.9))
    dx = float((diff(bezXc,t)).subs(t,0.9)) # t=1 は境界なので変な値にあることがあるため 0.9 としている
    dd = np.sqrt(dx*dx+dy*dy)
    dkusabi = (xRb-xLb)*abs(dx)/dd/2
    acc = dy/dx if dx != 0 else np.inf # 中心軸の傾き
    x00 = bezXc.subs(t,1) #  軸の再下端
    y00 = bezYc.subs(t,1)
    xdd = diaMinus*dx/sqrt(dx**2+dy**2) # 
    ddd = sqrt((crpx1-x00)**2+(crpy1-y00)**2) # ベジエ軸下端と輪郭下端の距離
    dddd = -10 if (xRu-xLu)/(xRb-xLb) > 4  else  ddd-dkusabi 
    if TAILCUT != 0: # 特別指定された削除調整量がある場合
        dddd = -TAILCUT
    dia_B = diaMinus-dddd
    x11_B = x00+xdd
    y11_B =  y00+xdd*acc if acc != np.inf else y00+diaMinus
    distO2top = np.sqrt(float(((crpy1-y11_B)**2 + (crpx1-x11_B)**2))) - dia_B  
    if  distO2top > 0 : # 削除中心と輪郭頂点の距離が削除半径より遠い(ということは削れない)ならその分以上に長くする
        dia_B += distO2top+5
    canvas =  cv2.circle(canvas,(int(x11_U),int(y11_U)),int(dia_U),0,-1) # 黒で円を描いて削る        
    canvas =  cv2.circle(canvas,(int(x11_B),int(y11_B)),int(dia_B),0,-1) # 黒で円を描いて削る
    
    flag = True
    while True:    
        # 輪郭検出すれば２つの輪郭が見つかるはず。
        _, contours, hierarchy = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 線図形の輪郭は中間で折り返しになっている
        
        if len(contours) < 2:
            continue
        cnt0 = contours[0][:int(len(contours[0])/2+1)]
        cnt1 = contours[1][:int(len(contours[1])/2+1)]
        # 中程の点を比べて左にある方を左と判定する。
        c0 = cnt0[int(len(cnt0)/2)][0][0]
        c1 = cnt1[int(len(cnt1)/2)][0][0]
        if  c0 > c1: 
            cntL,cntR = cnt1,cnt0
        else:
            cntR,cntL = cnt1,cnt0

        if len(cntL)/len(cntR) > 0.9 and  len(cntR)/len(cntL) > 0.9: # 左右の輪郭長の差が１０％以内
            break
        else:
            if flag:
                dia_U += 2
                flag = not flag
                canvas =  cv2.circle(canvas,(int(x11_U),int(y11_U)),int(dia_U),0,-1) # 黒で円を描いて削る 
            else: 
                dia_B += 2
                flag = not flag
                canvas =  cv2.circle(canvas,(int(x11_B),int(y11_B)),int(dia_B),0,-1) # 黒で円を描いて削る
            
    print("左輪郭点の数 ", len(cntL),"　右輪郭点の数　", len(cntR))

    #  ２重かっこを１重に変換し、numpy array にしてから返す
    cntL = np.array([[x,y] for [[x,y]] in cntL])
    cntR = np.array([[x,y] for [[x,y]] in cntR])                
    return cntL,cntR,(crpx0,crpy0),(crpx1,crpy1)

# 中心軸ベジエをもとにそれに輪郭点を左右に分割する
def reGetCntPair(img,cnt,cpl,cpr,bezC,CAPCUT=0,TAILCUT=0):
    # xLu,xRu,yLu,yRu = cpl[0][0],cpr[0][0],cpl[0][1],cpr[0][1] # 近似曲線の上端の座標
    # xLb,xRb,yLb,yRb = cpl[0][0],cpr[0][0],cpl[0][1],cpr[0][1] # 近似曲線の上端の座標
    bezXc,bezYc = bezC
    # 輪郭線の描画
    canvas = np.zeros_like(img)
    canvas = cv2.drawContours(canvas, cnt, -1, 255, thickness=1)
    # 軸と輪郭の交点
    (crpx0,crpy0),(crpx1,crpy1) = crossPoints(img,cnt,bezC)
    if crpy0 > crpy1: # 0 番が上、１番が底
        crpx0,crpy0,cpyx1,crpy1 = crpx1,crpy1,crpx0,crpy0
    
    # 中心軸の延長で上端から最大径離れた地点を中心に最大直径より少し大きな円を０で描き輪郭を削る。
    dMinus = 1024
    dy = float((diff(bezYc,t)).subs(t,0.1))
    dx = float((diff(bezXc,t)).subs(t,0.1)) # t=0 は境界なので変な値にあることがあるため 0.1 としている
    dd = np.sqrt(dx*dx+dy*dy) # dy,dxを縦横とする直角三角形の斜辺の長さ
    acc = dy/dx if dx != 0 else np.inf # 中心軸の傾き
    x00 = bezXc.subs(t,0) #  軸の再上端
    y00 = bezYc.subs(t,0)
    xdd = dMinus*dx/dd #  1024離れるためのX移動量
    # 削除円の中心
    x11_U = x00-xdd
    y11_U =  y00-xdd*acc if acc != np.inf else y00-dMinus
    # 削除円の中心と輪郭登頂（軸と輪郭の交点）の距離
    distO2top = np.sqrt(float(((crpy0-y11_U)**2 + (crpx0-x11_U)**2))) 
    # 削除円の半径を設定 CAPCUT: # 特別指定された削除調整量
    dia_U = distO2top+5 if CAPCUT == 0 else distO2top+CAPCUT # 確実に輪郭を削るための＋５
 
    # 同様に下端を削る
    dy = float((diff(bezYc,t)).subs(t,0.9))
    dx = float((diff(bezXc,t)).subs(t,0.9)) # t=1 は境界なので変な値にあることがあるため 0.9 としている
    dd = np.sqrt(dx*dx+dy*dy)
    acc = dy/dx if dx != 0 else np.inf # 中心軸の傾き
    x00 = bezXc.subs(t,1) #  軸の再下端
    y00 = bezYc.subs(t,1)
    xdd = dMinus*dx/dd
    # 削除円の中心
    x11_B = x00+xdd
    y11_B =  y00+xdd*acc if acc != np.inf else y00+dMinus
    distO2bottom = np.sqrt(float(((crpy1-y11_B)**2 + (crpx1-x11_B)**2)))   
    # 削除円の半径を設定 TAILCUT: # 特別指定された削除調整量
    dia_B = distO2bottom+5 if TAILCUT == 0 else distO2bottom+TAILCUT # 確実に輪郭を削るための＋５
        
    #canvas =  cv2.circle(canvas,(int(x11),int(y11)),int(dia_U),0,-1) # 黒で円を描いて削る
    canvas =  cv2.circle(canvas,(int(x11_U),int(y11_U)),int(dia_U),0,-1) # 黒で円を描いて削る        
    canvas =  cv2.circle(canvas,(int(x11_B),int(y11_B)),int(dia_B),0,-1) # 黒で円を描いて削る
    
    flag = True
    while True:    
        # 輪郭検出すれば２つの輪郭が見つかるはず。
        _, contours, hierarchy = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) ==2: 
            # 線図形の輪郭は中間で折り返しになっている
            cnt0 = contours[0][:int(len(contours[0])/2+1)]
            cnt1 = contours[1][:int(len(contours[1])/2+1)]
            # 中程の点を比べて左にある方を左と判定する。
            c0 = cnt0[int(len(cnt0)/2)][0][0]
            c1 = cnt1[int(len(cnt1)/2)][0][0]
            if  c0 > c1: 
                cntL,cntR = cnt1,cnt0
            else:
                cntR,cntL = cnt1,cnt0

            if len(cntL) > 200-CAPCUT-TAILCUT and len(cntR) > 200-CAPCUT-TAILCUT: # どちらか削り足りないに違いない
                break

            else:
                if flag:
                    dia_U += 2
                    flag = not flag
                    canvas =  cv2.circle(canvas,(int(x11_U),int(y11_U)),int(dia_U),0,-1) # 黒で円を描いて削る 
                else: 
                    dia_B += 2
                    flag = not flag
                    canvas =  cv2.circle(canvas,(int(x11_B),int(y11_B)),int(dia_B),0,-1) # 黒で円を描いて削る
        # 輪郭が２分割されていない、または、分割されているがどちらかの長さが異常である場合は繰り返す
        
    print("左輪郭点の数 ", len(cntL),"　右輪郭点の数　", len(cntR))

    #  ２重かっこを１重に変換し、numpy array にしてから返す
    cntL = np.array([[x,y] for [[x,y]] in cntL])
    cntR = np.array([[x,y] for [[x,y]] in cntR])                
    return cntL,cntR,(crpx0,crpy0),(crpx1,crpy1)

# 上端、下端の削られた部分と中心線の交点を求める # これは
def crossPoints(img,cnt,bezC):
        bezXc,bezYc = bezC
        canvas1 = np.zeros_like(img)  # 描画キャンバスの準備
        canvas2 = canvas1.copy()
        canvas1 = cv2.drawContours(canvas1, cnt, -1, 1, thickness=1)     # 輪郭線の描画
        
        # 中心軸上端から、軸の延長方向に直線を描く
        y0 = float(bezYc.subs(t,0.)) # 上端の座標
        x0 = float(bezXc.subs(t,0.))
        dx = float(diff(bezXc,t).subs(t,0.1)) #  　傾きを求めようとしている
        dy = float(diff(bezYc,t).subs(t,0.1))
        acc = dy/dx if dx != 0 else np.inf # 傾き
        x1 = x0 - y0/acc if dx !=0 else x0
        y1 = 0
        canvas2 = cv2.line(canvas2,(int(float(x0)),int(float(y0))),(int(float(x1)),int(float(y1))),1,2) # 幅3（2*2-1）の直線を明るさ１で描く
        canvas = canvas1 + canvas2
        cross_points0 = np.where(canvas==2) # 交点　　　重なった場所は値が２となっている.
        if len(cross_points0[0]) != 0 : crpy0,crpx0= np.average(cross_points0,axis=1)  # その平均座標

        canvas1 = np.zeros_like(img)  # 描画キャンバスの準備
        canvas2 = canvas1.copy()
        canvas1 = cv2.drawContours(canvas1, cnt, -1, 1, thickness=1)     # 輪郭線の描画

        # 中心軸下端から、軸の延長方向に直線を描く
        y0 = float(bezYc.subs(t,1.)) # 上端の座標
        x0 = float(bezXc.subs(t,1.))
        dx = float(diff(bezXc,t).subs(t,0.9)) #  　傾きを求めようとしている
        dy = float(diff(bezYc,t).subs(t,0.9))
        acc = dy/dx if dx != 0 else np.inf # 傾き
        x1 = x0 + (500-y0)/acc if dx !=0 else x0
        y1 = 500
        canvas2 = cv2.line(canvas2,(int(float(x0)),int(float(y0))),(int(float(x1)),int(float(y1))),1,2) # 幅3（2*2-1）の直線を明るさ１で描く
        canvas = canvas1 + canvas2
        cross_points1 = np.where(canvas==2) # 交点　　　重なった場所は値が２となっている.
        if len(cross_points1[0]) != 0 : crpy1,crpx1= np.average(cross_points1,axis=1)  # その平均座標
        
        return (crpx0,crpy0),(crpx1,crpy1)
    
# 左右のベジエ曲線の平均関数により中心軸のサンプル点を生成し、それをベジエ曲線で近似する関数。
def getcenterBez(bezL,bezR,C=3,precPara2=0.01,n_samples = 20, openmode=False,debugmode=False):
        bezXl,bezYl = bezL
        bezXr,bezYr = bezR
        # 左右のベジエ曲線の平均を求める
        bezXc,bezYc = (bezXl+bezXr)/2,(bezYl+bezYr)/2
        # 基本的にはこれが中心軸を表すが、5次だと両端に弊害が現れることが多いのでサンプル点を生成して再近似する
        csamples = [[float(bezXc.subs(t,i)),float(bezYc.subs(t,i))] for i in np.linspace(0, 1, n_samples)] # サンプル点を生成
        csamples = np.array(csamples)
        cpxc,cpyc,bezXc,bezYc,tpc = fitBezierCurveN(csamples,precPara=precPara2,N=C,openmode=openmode,debugmode=debugmode)
        return (cpxc,cpyc),[bezXc,bezYc]
    
# 中心軸の垂直断面幅を求める測定点を求める
def calcWidthFunc(bezL,bezR,bezC,n_samples,samplemode=1):  
        # mode 0 均等分割　
        # mode 1 おおまかに最大幅の位置を調べ、その位置から下はサンプルを２倍にする。
        bezXl,bezYl = bezL
        bezXr,bezYr = bezR
        bezXc,bezYc = bezC
        
        if samplemode == 2: #  最大位置から下はサンプル数を２倍にする場合
        # 最大幅の位置を大まかに決定する
            samplespace = np.linspace(0.01,0.99,n_samples) # 0 と 1 は特異なので避ける
            csamples = [[float(bezXc.subs(t,ts)),float(bezYc.subs(t,ts))] for ts in samplespace] 
            lsamples = [[float(bezXl.subs(t,ts)),float(bezYl.subs(t,ts))] for ts in samplespace]
            rsamples = [[float(bezXr.subs(t,ts)),float(bezYr.subs(t,ts))] for ts in samplespace]
            dsamples = np.array([np.sqrt((c[0]-l[0])**2+(c[1]-l[1])**2)+np.sqrt((c[0]-r[0])**2+(c[1]-r[1])**2) for (c,l,r) in zip(csamples,lsamples,rsamples)]) 
            div_pos = np.argmax(dsamples)
            samplespace = np.r_[samplespace[0:div_pos],np.linspace(samplespace[div_pos],0.99,2*(n_samples-div_pos))]
        else:
            samplespace = np.linspace(0.01, 0.99, n_samples)
        
        # 中心軸上に設定したサンプル点における法線と両輪郭の交点のリストを求める。
        PlistL,PlistR,PlistC = [],[],[]
        PosL,PosR,PosC=[],[],[]
        x0,y0 = var('x0,y0')
        for ts in samplespace: 
            y0 = float(bezYc.subs(t,ts))
            x0 = float(bezXc.subs(t,ts))
            dx = float(diff(bezXc,t).subs(t,ts)) # x、y をそれぞれ t で微分　傾きを求めようとしている
            dy = float(diff(bezYc,t).subs(t,ts))
            ans = solve(-dx/dy*(bezXr-x0)+y0-bezYr,t) # 法線とベジエ輪郭の交点を求める
            ansR = [re(i) for i in ans if float(Abs(im(i)))<0.00000001] 
            # ↑理論的には、im(i) == 0  でいいのだが、数値計算誤差で虚部が０とならず、微小な値となる現象に現実的な対応
            s = [i for i in ansR if  i<=1.03 and -0.03<=i] # ０から１までの範囲の解を抽出
            PlistR.append(s[0]) if s != [] else PlistR.append(np.inf) 
            PosR.append([float(bezXr.subs(t,s[0])),float(bezYr.subs(t,s[0]))]) if s !=[] else PosR.append([np.inf,np.inf])
            ans = solve(-dx/dy*(bezXl-x0)+y0-bezYl,t) # 法線とベジエ輪郭の交点を求める
            ansL = [re(i) for i in ans if float(Abs(im(i)))<0.00000001]
            s = [i for i in ansL if  i<=1.03 and -0.03<=i]
            PlistL.append(s[0]) if s != [] else PlistL.append(np.inf) 
            PosL.append([float(bezXl.subs(t,s[0])),float(bezYl.subs(t,s[0]))]) if s !=[] else PosL.append([np.inf,np.inf])
            PlistC.append(ts)
            PosC.append([x0,y0])
        return PlistL,PlistR,PlistC,PosL,PosR,np.array(PosC)
    
# 曲がりのない形状を計算する
def shapeReconstruction(savepath,cnt,PosL,PosR,PosC,bezL,bezR,bezC,cntl,cntr,C=4,precPara=0.01,
                        showImage=False,saveImage=False):
        #bezXl,bezYl = bezL
        #bezXr,bezYr = bezR
        bezXc,bezYc = bezC
        n_samples = len(PosL)
            
        # 中心軸と実輪郭の交点を求めて、上端の削除された長さを求める
        canvas1 = np.zeros((384,384))  # 描画キャンバスの準備
        canvas2 = canvas1.copy()
        canvas1 = cv2.drawContours(canvas1, cnt, -1, 1, thickness=1)     # 輪郭線の描画
        # 中心軸上端から、軸の延長方向に直線を描く
        y0 = float(bezYc.subs(t,0.)) # 上端の座標
        x0 = float(bezXc.subs(t,0.))
        dx = float(diff(bezXc,t).subs(t,0.)) #  　傾きを求めようとしている
        dy = float(diff(bezYc,t).subs(t,0.))
        acc = dy/dx if dx != 0 else np.inf # 傾き
        x1 = x0 - y0/acc if dx !=0 else x0
        y1 = 0
        canvas2 = cv2.line(canvas2,(int(float(x0)),int(float(y0))),(int(float(x1)),int(float(y1))),1,2) # 幅3（2*2-1）の直線を明るさ１で描く
        canvas = canvas1 + canvas2
        cross_points = np.where(canvas==2) # 交点　　　重なった場所は値が２となっている.
        if len(cross_points[0]) != 0 : crosspy,crosspx= np.average(cross_points,axis=1)  # その平均座標
        caplength = 0 if len(cross_points[0]) == 0 else np.sqrt((crosspx-x0)**2+(crosspy-y0)**2) # 削られた分の長さ
        print("CAP(近似除外上端部)　{0:0.1f}".format(caplength))
        
        # 定積分により軸に沿った長さを求める
        '''s = var('s')
        dxdt = diff(bezXc,t)
        dydt = diff(bezYc,t)
        leng = integrate(sqrt(dxdt**2+dydt**2),(t,0,s)) # 長さをパラメータの関数として求める計算式
        radiusTable = []
        for i, tpara in enumerate(np.linspace(0,1,n_samples)):
            cx,cy = PosC[i][0],PosC[i][1]
            length = float(leng.subs(s,tpara))+caplength # 上端からの長さ
            lx,ly = PosL[i][0],PosL[i][1]
            rx,ry = PosR[i][0],PosR[i][1]
            if lx != np.inf and rx != np.inf:
                ll = np.sqrt(float(lx-cx)**2+float(ly-cy)**2)
                rl = np.sqrt(float(rx-cx)**2+float(ry-cy)**2)
                radishR = (ll+rl)/2 # 半径
            elif lx == np.inf and rx != np.inf:
                rl = np.sqrt(float(rx-cx)**2+float(ry-cy)**2)
                radishR = rl # 半径
            elif lx !=  np.inf and rx == np.inf:
                ll = np.sqrt(float(lx-cx)**2+float(ly-cy)**2)
                radishR = ll # 半径
            else:
                radishR = np.inf
            if radishR != np.inf :
                radiusTable.append([radishR,length])
        radiusTable = np.array(radiusTable)'''
        
        # 近似折れ線の長さの和で定積分を代替する
        SEGN = 1 # サンプル間の分割数。増やした方が良いはずだが、やってみると増やした方が定積分の結果より小さくなった。
        # SEGN =1 がもっとも定積分の結果に近いようなので、1としておく
        fx = [float(bezXc.subs(t,tp)) for tp in np.linspace(0,1,SEGN*n_samples)]
        fy = [float(bezYc.subs(t,tp)) for tp in np.linspace(0,1,SEGN*n_samples)]
        lengthTable = [caplength]
        tlength = caplength
        for index,i in enumerate(np.linspace(0,1,n_samples-1)):
            ii  = SEGN*index
            for n in range(SEGN):
                tlength += np.sqrt((fx[ii+n+1]-fx[ii+n])**2+(fy[ii+n+1]-fy[ii+n])**2)
            lengthTable.append(tlength)
        radiusTable = []
        for i in range(n_samples):
            cx,cy = PosC[i][0],PosC[i][1]
            lx,ly = PosL[i][0],PosL[i][1]
            rx,ry = PosR[i][0],PosR[i][1]
            if lx != np.inf and rx != np.inf:
                ll = np.sqrt(float(lx-cx)**2+float(ly-cy)**2)
                rl = np.sqrt(float(rx-cx)**2+float(ry-cy)**2)
                radishR = (ll+rl)/2 # 半径
            elif lx == np.inf and rx != np.inf:
                rl = np.sqrt(float(rx-cx)**2+float(ry-cy)**2)
                radishR = rl # 半径
            elif lx !=  np.inf and rx == np.inf:
                ll = np.sqrt(float(lx-cx)**2+float(ly-cy)**2)
                radishR = ll # 半径
            else:
                radishR = np.inf
            if radishR != np.inf :
                radiusTable.append([radishR,lengthTable[i]])
        radiusTable = np.array(radiusTable)
 
        #  延伸形状をベジエ曲線で近似
        cpxl,cpyl,shapeX,shapeY,_tpl = fitBezierCurveN(radiusTable,precPara=precPara,N=C)
        # 最大径とその位置を求める
        fx = np.array([float(shapeX.subs(t,i)) for i in np.linspace(0,1,101)]) # 0.01刻み
        fy = np.array([float(shapeY.subs(t,i)) for i in np.linspace(0,1,101)])
        xmax_index = np.argmax(fx)
        maxDia = 2*fx[xmax_index]
        btmline_index = xmax_index + np.argmin((fx[xmax_index:] - maxDia*0.1)**2) # 最大幅の２０％に一番近い幅のインデックス               
        radishLength = fy[btmline_index]
        print("ダイコンの長さ={0:0.1f}　　（CAPを含む）".format(float(radishLength)) )
        maxpos = 100*fy[xmax_index]/radishLength
        print("最大直径={0:0.2f} 最大直径の位置は、上端から{1:0.2f} % の位置".format(maxDia,maxpos))
        print("最大直径位置のパラメータ  {0:0.3f},".format(0.01*xmax_index),end="")
        print("径20％位置のパラメータ  {0:0.3f}".format(0.01*btmline_index))
        
        #結果の描画
        if showImage:
            plt.figure(figsize=(6,6),dpi=100)
            plt.gca().invert_yaxis() 
            plt.gca().set_aspect('equal', adjustable='box') # アスペクト比を１：１に

            # 輪郭の描画
            ploty = fy
            plotLx = fx
            plotRx = [-x for x in plotLx]
            plt.plot(plotLx,ploty,color = 'blue')  
            plt.plot(plotRx,ploty,color = 'blue')  
            plt.plot([-fx[xmax_index],fx[xmax_index]],[fy[xmax_index],fy[xmax_index]],color = 'red')
            plt.plot([-fx[btmline_index],fx[btmline_index]],[fy[btmline_index],fy[btmline_index]],color = 'red')
            
            def drawcap(cap, ra = 5):
                magicnumber2=10.59
                x = np.arange(-ra, ra+0.01, 0.01)
                plt.plot(x, cap*(np.cosh(np.pi*x/ra)-1)/magicnumber2,color='green')
            
            drawcap(cap=float(fy[0]),ra=float(fx[0]))
            if saveImage:
                pltsaveimage(savepath,'STRCH')
     
        return (cpxl,cpyl),shapeX,shapeY,radishLength, maxDia, maxpos, 0.01*xmax_index,0.01*btmline_index
    
# 差分の表示
def diffCnt2Bez(img,cnt,cntl,cntr,bezL,bezR, showImage=False):
    bezLx,bezLy = bezL
    bezRx,bezRy = bezR
    
    x00 = int(float(bezLx.subs(t,0)))
    y00 = int(float(bezLy.subs(t,0)))
    x01 = int(float(bezRx.subs(t,0)))
    y01 = int(float(bezRy.subs(t,0)))
    x10 = int(float(bezLx.subs(t,1)))
    y10 = int(float(bezLy.subs(t,1)))
    x11 = int(float(bezRx.subs(t,1)))
    y11 = int(float(bezRy.subs(t,1)))

    canvas1 = img.copy()
    canvas2 = np.zeros_like(img)
    
    # img の近似対象部分のみ切り出し
    acc0 = (y01-y00)/(x01-x00)
    acc1 = (y11-y10)/(x11-x10)
    x00e = x00 - 100
    y00e = int(y00 - 100*acc0)-3
    x01e = x01 + 100
    y01e = int(y01 + 100*acc0)-3
    x10e = x10 -100
    y10e = int(y10 - 100*acc1)+3
    x11e = x11 +100
    y11e = int(y11 +100*acc1)+3
    cv2.line(canvas1,(x00e,y00e),(x01e,y01e),0,2)
    cv2.line(canvas1,(x10e,y10e),(x11e,y11e),0,2)
    _lnum, labelimg, cnt, _cog =cv2.connectedComponentsWithStats(canvas1)
    areamax = np.argmax(cnt[1:,4])+1 # ０番を除く面積最大値のインデックス
    area = cnt[areamax][4]
    canvas1 = np.array(255*(labelimg==areamax),np.uint8)
    
    # ベジエ近似画像の描画
    tseq = np.linspace(0,1,400)
    chainL = [[[int(float(bezLx.subs(t,tp))),int(float(bezLy.subs(t,tp)))]] for tp in tseq] 
    chainR = [[[int(float(bezRx.subs(t,tp))),int(float(bezRy.subs(t,tp)))]] for tp in tseq[::-1]] 
    chain = np.array(chainL + chainR + [chainR[-1],chainL[0]])
    canvas2 = cv2.drawContours(canvas2,[chain],-1,255,-1)
    # XORを取って差分とする
    diffimg = cv2.bitwise_xor(canvas1,canvas2)
    # 上端、下端を結ぶ線分は誤差でないので取り除く
    cv2.line(diffimg,(x00e,y00e),(x01e,y01e),0,5)
    cv2.line(diffimg,(x10e,y10e),(x11e,y11e),0,5)
    diffareas = np.sum(diffimg/255)
    ncontours = len(cntl)+len(cntr)
    print('近似対象の面積は、{},  輪郭画素数は{},  ずれ {}画素  (平均量子化誤差を減じた実質誤差 {}) '.format(area,ncontours,diffareas,diffareas-ncontours/2))
    
    if showImage:
        color1 = cv2.merge((diffimg,diffimg,diffimg))
        plt.figure(figsize=(6,6),dpi=100)
        plt.gca().set_aspect('equal', adjustable='box') # アスペクト比を１：１に
        plotimg(color1)
        
    return canvas1,canvas2,area,ncontours, diffareas

def pltsaveimage(savepath,prefix):
        # 結果を保存する
        savedir,filename = os.path.split(savepath)
        os.makedirs(savedir, exist_ok=True) # 保存先フォルダがなければ作成
        plt.savefig(os.path.join(savedir,prefix+filename))

import os

def automeasure(datafile = '自動計測データ.xlsx', savedir='延伸シルエット', saveImage=True):
    # savedir 保存先
    # smooth 領域抽出に先立って画像をぼかすかどうかのフラグ ぼかす場合は excel ファイルの ssize 欄の数値が適用される
    # interactive 結果を１枚ずつ確認するかどうか
    # テストモードの場合は check 欄に１のある画像だけが処理対象となる
    
    global df
    # バッチ司令ファイルの読み込み
    df = pd.read_excel(datafile)
    # df = pd.read_csv('画像リストUTF8.csv', sep=',')
    
    for radish in range(len(df)):
            idata = df.iloc[radish]
            topdir = idata['topdir']  #  画像ファイルのパスのベース
            subdir = idata['subdir']  #  サブディレクトリ
            filename = idata['filename'] #  ファイル名
            rename = idata['rename']
            dCCT0 = idata['CCUT0'] # 最初の上部削減％　 defailt 5%ライン
            dTCT0 = idata['TCUT0'] # 最初の下部削減％ default 95%ライン
            dROT = idata['ROT'] # 自動で決める向きではうまくいかない場合の回転量指示　左回りが正
            dM = idata['M']
            dN = idata['N']
            dC = idata['C']
            dL = idata['L']
            dprecPara1= idata['precPara1']
            dprecPara2 = idata['precPara2']
            dCAPCUT = idata['CAPCUT']
            dTAILCUT = idata['TAILCUT']
            dsamplemode = idata['sample mode']
            dn_samples1 = idata['n_samples1']
            dn_samples2 = idata['n_samples2']
                
            check = idata['処理対象'] #  処理対象かどうかのフラグ　　test がTrueの時のみ意味がある
            if test and check !=1 : #  test 時で check が 1 でない画像はスルーする
                continue 
                
            path = os.path.join(topdir,subdir,filename)
            savepath = os.path.join(savedir,subdir,rename)
            print("処理対象画像 {} -> {} \n".format(path,savepath))
                
            print(" 近似パラメータ　M {0} N {1} C {2} L{3} p1 {4:0.3f} p2{5:0.3f}".format(dM,dN,dC,dL,dprecPara1,dprecPara2))
            print(" 近似用サンプル数 {0}, 延伸モード {1}. 延伸サンプル数 {2}".format(dn_samples1,dsamplemode,dn_samples2))
            print(" カスタムパラメータ　CCUT0 {0}, TCUT0 {1}, ROT {2} CAPCUT {3} TAILCUT {4}\n".format(dCCT0,dTCT0,dROT,dCAPCUT,dTAILCUT)) 
        
            src= cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            # 直線当てはめした場合の方向ベクトル
            # _ret,img = cv2.threshold(src,127,255,cv2.THRESH_BINARY)
            # _image, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #cnt = contours[0]
            # [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)

            img = getstandardShape(src, unitSize=UNIT, thres=0.25, setrotation = dROT, showResult=False)
            
            '''cv2.imshow(str(dROT),img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)'''
            
            approxAndMeasurement(savepath, df, radish,img,M=dM,N=dN,C=dC,L=dL,
                                 cutC=dCCT0,cutB=dTCT0,precPara1=dprecPara1, 
                                 precPara2=dprecPara2,samplemode=dsamplemode,n_samples=dn_samples1,
                                 n_samples2=dn_samples2,CAPCUT=dCAPCUT,TAILCUT=dTAILCUT,
                                 openmode=False, debugmode=False,showImage=True,saveImage=saveImage)

            # df.to_excel(datafile, index=True, header=True)

    return df

def approxAndMeasurement(savepath,df, radish, img,M=4,N=5,C=4,L=5,cutC=0,cutB=95, \
                         precPara1=0.05,precPara2=0.01,samplemode=0,n_samples=20,n_samples2=30,\
                         CAPCUT=0,TAILCUT=0,openmode=False,debugmode=False,\
                         showImage=True,saveImage=False,savedir=None):
    starttime = time.time()
    # df 記録用データフレーム
    # radish 記録する行
    # img 処理対象シルエット画像    
    # 制御点ををデータフレームに記録する    
    def CPrecord(cps,prefix):
        (CPX,CPY) = cps
        for i,(px,py) in enumerate(zip(list(CPX),list(CPY))):
            df.loc[radish,prefix+'X'+str(i)]=px
            df.loc[radish,prefix+'Y'+str(i)]=py
    
    #フェーズ１　仮中心軸の生成
    print('仮分割…',end='')
    ## 輪郭線を左右に仮分割, 
    cntl,cntr,cnt = preGetLRdata(img,tlevel = cutC, blevel=cutB,bracket=1)
    
    print('ベジエあてはめ１…',end='')
    ## ベジエ曲線あてはめ（パス１）
    cpl,cpr,_cpc, bezL,bezR,bezC,_cntL,_cntR = cntPair2bez(cntl,cntr,N=M, precPara=precPara1,samplemode = 0,openmode=openmode,debugmode=debugmode)
    print('輪郭線左右分割…',end='')
    ## 中心軸をもとにしてより妥当な左右の輪郭をえる
    
    cntl,cntr,_TopP,_ = reGetCntPair(img,cnt,cpl,cpr,bezC,CAPCUT=CAPCUT,TAILCUT=TAILCUT)
    
    print('ベジエあてはめ2…',end='')
    ## ベジエ曲線あてはめ （パス２）
    cpl,cpr,cpc, bezL,bezR,bezC,cntL,cntR= cntPair2bez(cntl,cntr,N=N,n_samples=n_samples, samplemode = samplemode, precPara=precPara2, openmode=openmode,debugmode=debugmode)
    CPrecord(cpl,'LP')
    CPrecord(cpr,'RP')
    
    print('左右平均点へのベジエあてはめ…',end='')
    ## 中心軸へのベジエあてはめ
    cpc2,bezC2  = getcenterBez(bezL,bezR,C=C,precPara2=precPara2,n_samples = n_samples, openmode=openmode,debugmode=debugmode)
    CPrecord(cpc2,'CP')
    
    print('幅サンプル生成…',end='')
    ## 幅のサンプリング
    _PlistL,_PlistR,_PlistC, PosL,PosR,PosC = calcWidthFunc(bezL,bezR,bezC2,n_samples=n_samples2,samplemode=samplemode)
    print('あてはめ結果表示…',end='')
    ## 結果の表示
    drawBez2(savepath,img,bezL,bezR,bezC=bezC2,ladder='normal',PosL=PosL,PosR=PosR,PosC=PosC, n_samples=n_samples2,saveImage=saveImage) 
    print('延伸形状復元…',end='')
    ## 延伸形状復元＆計測
    CPs,shapeX,shapeY,radishLength, maxDia ,maxpos,t_max,t_bottom = shapeReconstruction(savepath,cnt,PosL,PosR,PosC,bezL,bezR,bezC2,cntl,cntr,C=L,precPara=precPara2,\
                                                                         showImage=showImage,saveImage=saveImage)
    df.loc[radish,'最大径']=maxDia
    df.loc[radish,'長さ']=radishLength
    df.loc[radish,'最大径位置']=maxpos
    df.loc[radish,'最大径位置のt値']=t_max
    df.loc[radish,'径20％位置のt値']=t_bottom
    print('曲線の方程式\n',shapeX,'\n',shapeY)
    CPrecord(CPs,'SP')

    print('ずれ計算')
    ## 面積のずれの計算
    _canvas1,_canvas2, area,contournum, difference = diffCnt2Bez(img,cnt,cntl,cntr,bezL,bezR, showImage=False)
    df.loc[radish,'誤差']= difference
    df.loc[radish,'近似対象輪郭画素数'] = contournum
    df.loc[radish,'近似対象面積']= area
    datetimenow = '{0:%Y/%m/%d/%H:%M}'.format(datetime.datetime.now()) # 処理した日時
    df.loc[radish,'処理対象']=datetimenow
    print(datetimenow)
    df.to_excel('自動計測データ.xlsx', index=True, header=True)

    elapsed_time = time.time() - starttime
    print ("処理時間:{0}".format(elapsed_time) + "[sec]")

# すベて実行
automeasure()