import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
from PIL import Image
import math
import pandas as pd

from sympy import *
from sympy.abc import a,b,c
# init_session()
px,py =var('px:4'),var('py:4')
t = symbols('t')
from sympy import var
from rdlib2 import *

import datetime
import time

# バッチ司令ファイルの読み込み
df = pd.read_excel('自動計測データ.xlsx')
# df = pd.read_csv('画像リストUTF8.csv', sep=',')
df.head(5)

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
   
import os

def automeasure(datafile = '自動計測データ.xlsx', savedir='伸身シルエット', saveImage=True):
    # savedir 保存先
    # smooth 領域抽出に先立って画像をぼかすかどうかのフラグ ぼかす場合は excel ファイルの ssize 欄の数値が適用される
    # interactive 結果を１枚ずつ確認するかどうか
    # テストモードの場合は check 欄に１のある画像だけが処理対象となる
    
    global df,rdimg, rdcnt,rdcimg #  シルエット画像、輪郭、輪郭画像は度々使うのでグローバルにしておく
    # バッチ司令ファイルの読み込み
    df = pd.read_excel(datafile)
    # df = pd.read_csv('画像リストUTF8.csv', sep=',')
    
    wcalcmode = 1 #  0 は幅を求める方法　　0 なら作図で、１なら関数式で
    
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
            if check !=1 : #  check が 1 でない画像はスルーする
                continue 
                
            path = os.path.join(topdir,subdir,filename)
            savepath = os.path.join(savedir,subdir,rename)
            print("処理対象画像 {} -> {} \n".format(path,savepath))
                
            print(" 近似パラメータ　M {0} N {1} C {2} L{3} p1 {4:0.3f} p2 {5:0.3f}".format(dM,dN,dC,dL,dprecPara1,dprecPara2))
            print(" 近似用サンプル数 {0}, 伸身モード {1}. 伸身サンプル数 {2}".format(dn_samples1,dsamplemode,dn_samples2))
            print(" カスタムパラメータ　CCUT0 {0}, TCUT0 {1}, ROT {2} CAPCUT {3} TAILCUT {4}\n".format(dCCT0,dTCT0,dROT,dCAPCUT,dTAILCUT)) 
        
            src= cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            # 直線当てはめした場合の方向ベクトル
            # ret,img = cv2.threshold(src,127,255,cv2.THRESH_BINARY)
            # image, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # cnt = contours[0]
            # [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)

            rdimg = getstandardShape(src, unitSize=UNIT, thres=0.25, setrotation = dROT, showResult=False)
            _img,contours,hierarchy = cv2.findContours(rdimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            rdcnt = contours[0] # 輪郭線情報　global 変数　
            rdcimg = np.zeros_like(rdimg)  # 輪郭画像も作っておく
            rdcimg = cv2.drawContours(rdcimg, rdcnt, -1, 255, thickness=1)  # 輪郭線の描画
            
            '''cv2.imshow(str(dROT),img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)'''
            
            approxAndMeasurement(savepath, radish,M=dM,N=dN,C=dC,L=dL,
                                 cutC=dCCT0,cutB=dTCT0,precPara1=dprecPara1, 
                                 precPara2=dprecPara2,samplemode=dsamplemode,n_samples=dn_samples1,
                                 n_samples2=dn_samples2,CAPCUT=dCAPCUT,TAILCUT=dTAILCUT,
                                 openmode=False, debugmode=False,showImage=True,
                                 saveImage=saveImage,wcalcmode = wcalcmode)

            # df.to_excel(datafile, index=True, header=True)

    return df

def approxAndMeasurement(savepath,radish,M=4,N=5,C=4,L=5,cutC=0,cutB=95, \
                         precPara1=0.05,precPara2=0.01,samplemode=0,n_samples=20,n_samples2=30,\
                         CAPCUT=0,TAILCUT=0,openmode=False,debugmode=False,\
                         showImage=True,saveImage=False,wcalcmode=0):
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
    cntl,cntr = preGetLRdata(tlevel = cutC, blevel=cutB,bracket=1)
    
    print('ベジエあてはめ１…',end='')
    ## ベジエ曲線あてはめ（パス１）
    cpl,cpr,cpc, bezL,bezR,bezC,cntL,cntR = cntPair2bez(cntl,cntr,N=M, precPara=precPara1,samplemode = 0,openmode=openmode,debugmode=debugmode)
    print('\n輪郭線左右分割…',end='')
    ## 中心軸をもとにしてより妥当な左右の輪郭をえる
    
    cntl,cntr,TopP,_ = reGetCntPair(cpl,cpr,bezC,CAPCUT=CAPCUT,TAILCUT=TAILCUT)
    
    print('ベジエあてはめ2…',end='')
    ## ベジエ曲線あてはめ （パス２）
    cpl,cpr,cpc, bezL,bezR,bezC,cntL,cntR= cntPair2bez(cntl,cntr,N=N,n_samples=n_samples, samplemode = samplemode, precPara=precPara2, openmode=openmode,debugmode=debugmode)
    CPrecord(cpl,'LP')
    CPrecord(cpr,'RP')
    
    print('左右平均点へのベジエあてはめ１',end='')
    ## 中心軸へのベジエあてはめ
    cpc2,bezC2  = getcenterBez(bezL,bezR,C=C,precPara2=precPara2,n_samples = n_samples, openmode=openmode,debugmode=debugmode)
    CPrecord(cpc2,'CP')
    
    print('幅サンプル生成…',end='')
    
    for loop in range(3):
        ## 幅のサンプリング
        print("loop",loop)
        ## 幅サンプル生成
        PosL,PosR,PosC = calcWidthFunc2(bezC2,n_samples=n_samples2)    
        ## 中心軸を再決定 
        cpxc,cpyc,bezXc,bezYc,_tpc = fitBezierCurveN(PosC,precPara=precPara2,N=C,openmode=openmode,debugmode=debugmode)
        bezC2 = (bezXc,bezYc)
    
    print('あてはめ結果表示…',end='')
    ## 結果の表示
    drawBez2(savepath,bezL,bezR,bezC=bezC2,ladder='normal',PosL=PosL,PosR=PosR,PosC=PosC, n_samples=n_samples2,saveImage=saveImage) 
    print('伸身形状復元…',end='')
    ## 伸身形状復元＆計測
    CPs,shapeX,shapeY,radishLength, maxDia ,maxpos,t_max,t_bottom = shapeReconstruction(savepath,PosL,PosR,PosC,bezL,bezR,bezC2,cntl,cntr,C=L,precPara=precPara2,\
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
    _canvas1,_canvas2, area,contournum, difference = diffCnt2Bez(cntl,cntr,bezL,bezR, showImage=False)
    df.loc[radish,'誤差']= difference
    df.loc[radish,'近似対象輪郭画素数'] = contournum
    df.loc[radish,'近似対象面積']= area
    datetimenow = '{0:%Y/%m/%d/%H:%M}'.format(datetime.datetime.now()) # 処理した日時
    df.loc[radish,'処理対象']=datetimenow
    print(datetimenow)
    df.to_excel('自動計測データ.xlsx', index=True, header=True)
    
    elapsed_time = time.time() - starttime
    print ("処理時間:{0}".format(elapsed_time) + "[sec]")
import os

def automeasure(datafile = '自動計測データ.xlsx', savedir='伸身シルエット', saveImage=True):
    # savedir 保存先
    # smooth 領域抽出に先立って画像をぼかすかどうかのフラグ ぼかす場合は excel ファイルの ssize 欄の数値が適用される
    # interactive 結果を１枚ずつ確認するかどうか
    # テストモードの場合は check 欄に１のある画像だけが処理対象となる
    
    global df,rdimg, rdcnt,rdcimg #  シルエット画像、輪郭、輪郭画像は度々使うのでグローバルにしておく
    # バッチ司令ファイルの読み込み
    df = pd.read_excel(datafile)
    # df = pd.read_csv('画像リストUTF8.csv', sep=',')
    
    wcalcmode = 1 #  0 は幅を求める方法　　0 なら作図で、１なら関数式で
    
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
            if check !=1 : #  check が 1 でない画像はスルーする
                continue 
                
            path = os.path.join(topdir,subdir,filename)
            savepath = os.path.join(savedir,subdir,rename)
            print("処理対象画像 {} -> {} \n".format(path,savepath))
                
            print(" 近似パラメータ　M {0} N {1} C {2} L{3} p1 {4:0.3f} p2 {5:0.3f}".format(dM,dN,dC,dL,dprecPara1,dprecPara2))
            print(" 近似用サンプル数 {0}, 伸身モード {1}. 伸身サンプル数 {2}".format(dn_samples1,dsamplemode,dn_samples2))
            print(" カスタムパラメータ　CCUT0 {0}, TCUT0 {1}, ROT {2} CAPCUT {3} TAILCUT {4}\n".format(dCCT0,dTCT0,dROT,dCAPCUT,dTAILCUT)) 
        
            src= cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            # 直線当てはめした場合の方向ベクトル
            # ret,img = cv2.threshold(src,127,255,cv2.THRESH_BINARY)
            # image, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # cnt = contours[0]
            # [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)

            rdimg = getstandardShape(src, unitSize=UNIT, thres=0.25, setrotation = dROT, showResult=False)
            _img,contours,hierarchy = cv2.findContours(rdimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            rdcnt = contours[0] # 輪郭線情報　global 変数　
            rdcimg = np.zeros_like(rdimg)  # 輪郭画像も作っておく
            rdcimg = cv2.drawContours(rdcimg, rdcnt, -1, 255, thickness=1)  # 輪郭線の描画
            
            '''cv2.imshow(str(dROT),img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)'''
            
            approxAndMeasurement(savepath, radish,M=dM,N=dN,C=dC,L=dL,
                                 cutC=dCCT0,cutB=dTCT0,precPara1=dprecPara1, 
                                 precPara2=dprecPara2,samplemode=dsamplemode,n_samples=dn_samples1,
                                 n_samples2=dn_samples2,CAPCUT=dCAPCUT,TAILCUT=dTAILCUT,
                                 openmode=False, debugmode=False,showImage=True,
                                 saveImage=saveImage,wcalcmode = wcalcmode)

            # df.to_excel(datafile, index=True, header=True)

    return df

def approxAndMeasurement(savepath,radish,M=4,N=5,C=4,L=5,cutC=0,cutB=95, \
                         precPara1=0.05,precPara2=0.01,samplemode=0,n_samples=20,n_samples2=30,\
                         CAPCUT=0,TAILCUT=0,openmode=False,debugmode=False,\
                         showImage=True,saveImage=False,wcalcmode=0):
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

    print ("処理時間:{0}".format(elapsed_time) + "[sec]")
    #フェーズ１　仮中心軸の生成
    print('仮分割…',end='')
    ## 輪郭線を左右に仮分割, 
    cntl,cntr = preGetLRdata(tlevel = cutC, blevel=cutB,bracket=1)
    
    print('ベジエあてはめ１…',end='')
    ## ベジエ曲線あてはめ（パス１）
    cpl,cpr,cpc, bezL,bezR,bezC,cntL,cntR = cntPair2bez(cntl,cntr,N=M, precPara=precPara1,samplemode = 0,openmode=openmode,debugmode=debugmode)
    print('\n輪郭線左右分割…',end='')
    ## 中心軸をもとにしてより妥当な左右の輪郭をえる
    
    cntl,cntr,TopP,_ = reGetCntPair(cpl,cpr,bezC,CAPCUT=CAPCUT,TAILCUT=TAILCUT)
    
    print('ベジエあてはめ2…',end='')
    ## ベジエ曲線あてはめ （パス２）
    cpl,cpr,cpc, bezL,bezR,bezC,cntL,cntR= cntPair2bez(cntl,cntr,N=N,n_samples=n_samples, samplemode = samplemode, precPara=precPara2, openmode=openmode,debugmode=debugmode)
    CPrecord(cpl,'LP')
    CPrecord(cpr,'RP')
    
    print('左右平均点へのベジエあてはめ１',end='')
    ## 中心軸へのベジエあてはめ
    cpc2,bezC2  = getcenterBez(bezL,bezR,C=C,precPara2=precPara2,n_samples = n_samples, openmode=openmode,debugmode=debugmode)
    CPrecord(cpc2,'CP')
    
    print('幅サンプル生成…',end='')
    
    print ("処理時間:{0}".format(elapsed_time) + "[sec]")
    # 中心軸を基準とした断面の決定と断面の中心列の近似による中心軸の決定を３回繰り返す
    for loop in range(3):
        ## 幅のサンプリング
        print("loop",loop)
        ## 幅サンプル生成
        PosL,PosR,PosC = calcWidthFunc2(bezC2,n_samples=n_samples2)    
        ## 中心軸を再決定 
        cpxc,cpyc,bezXc,bezYc,_tpc = fitBezierCurveN(PosC,precPara=precPara2,N=C,openmode=openmode,debugmode=debugmode)
        bezC2 = (bezXc,bezYc)
    
    print('あてはめ結果表示…',end='')
    ## 結果の表示
    drawBez2(savepath,bezL,bezR,bezC=bezC2,ladder='normal',PosL=PosL,PosR=PosR,PosC=PosC, n_samples=n_samples2,saveImage=saveImage) 
    print('伸身形状復元…',end='')
    ## 伸身形状復元＆計測
    CPs,shapeX,shapeY,radishLength, maxDia ,maxpos,t_max,t_bottom = shapeReconstruction(savepath,PosL,PosR,PosC,bezL,bezR,bezC2,cntl,cntr,C=L,precPara=precPara2,\
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
    _canvas1,_canvas2, area,contournum, difference = diffCnt2Bez(cntl,cntr,bezL,bezR, showImage=False)
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