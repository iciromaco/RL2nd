# 2018.12.2 refineTparaR bug fixed
# 2018.12.3 srange(n,stt,end,smin,smax) 微修正
# 2018.12.3 fitBezierCurveN.refineTparaN で設定ポイント数を均等から上端・末端付近は それぞれ n_points, 中央部も n_points 合計3＊n_ポイントとした
# 2018.12.8 getCoG の仕様変更　バウンディングボックスに加えて面積も一緒に返すように変更
# 2018.12.8 getCoGandTip のガウスぼかしのカーネルサイズの基準を画像の幅からバウンディングボックスの対角線長さと面積から見積もった幅に変更
# 2018.12.10 refineTparaN を少し修正。openmode でない時、tpara の推定値を０〜１に、openmode の時は -0.1〜1.1 に、候補に１が入っていなかったので含めるように修正
# 2019.01.04 fitBezierCurveN で経路に沿った距離を計算するコードを自作から findArcLength に置き換え

def assertglobal(params,verbose=False):
    global CONTOURS_APPROX, HARRIS_PARA, CONTOURS_APPROX, SHRINK, \
            HARRIS_PARA, GAUSSIAN_RATE1, GAUSSIAN_RATE2, UNIT, RPARA
    for item in params:
        if item == 'CONTOURS_APPROX':
            CONTOURS_APPROX = params[item] # 輪郭近似精度
        elif item == 'HARRIS_PARA':
            HARRIS_PARA = params[item] # ハリスコーナー検出で、コーナーとみなすコーナーらしさの指標  1.0 なら最大値のみ
        elif item == 'CONTOURS_APPROX' :
            CONTOURS_APPROX = params[item] # 輪郭近似精度
        elif item == 'SHRINK':
            SHRINK = params[item] # 0.75 # 収縮膨張で形状を整える時のパラメータ
        elif item == 'GAUSSIAN_RATE1':
            GAUSSIAN_RATE1= params[item] # 先端位置を決める際に使うガウスぼかしの程度を決める係数
        elif item == 'GAUSSIAN_RATE2':
            GAUSSIAN_RATE2 = params[item] # 仕上げに形状を整えるためのガウスぼかしの程度を決める係数
        elif item == 'UNIT':
            UNIT = params[item] # 最終的に長い方の辺をこのサイズになるよう拡大縮小する
        elif item == 'RPARA':
            RPARA = params[item]# 見込みでサーチ候補から外す割合
        if verbose:
            print(item, "=", params[item])

assertglobal(params = {
    'HARRIS_PARA':1.0, # ハリスコーナー検出で、コーナーとみなすコーナーらしさの指標  1.0 なら最大値のみ
    'CONTOURS_APPROX':0.0002, # 輪郭近似精度
    'SHRINK':0.8, # 0.75 # 収縮膨張で形状を整える時のパラメータ
    'GAUSSIAN_RATE1':0.2, # 先端位置を決める際に使うガウスぼかしの程度を決める係数
    'GAUSSIAN_RATE2':0.1, # 仕上げに形状を整えるためのガウスぼかしの程度を決める係数
    'UNIT':256, # 最終的に長い方の辺をこのサイズになるよう拡大縮小する
    'RPARA':1.0 # 見込みサーチのサーチ幅全体に対する割合 ３０なら左に３０％右に３０％の幅を初期探索範囲とする
})

#  汎用の関数

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from skimage.morphology import skeletonize, skeletonize_3d
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage import io
import skimage.data
import skimage.color
# from sympy import *
from sympy import diff,Symbol,Matrix,symbols,solve,simplify,binomial
from sympy.abc import a,b,c
# init_session()
from sympy import var

# (1)画像のリストアップ
def listimage(path='シルエット', needThum=False):

    directory = os.listdir(path)
    
    # まずフォルダを全部リストアップ　→ folders
    folders = []
    for x in directory:
        if os.path.isdir(path + '/'+x) and x[0] != '.' and x[0] !='_':  #パスに取り出したオブジェクトを足してフルパスに
            folders.append(path + '/'+x)
    # print(folders)

    # folders の各フォルダの中にある jpg ファイルをリストアップ → ffiles
    # ffiles はフォルダごとのリストのリスト
    ffiles = []
    for x in folders:
        files = []
        for f in os.listdir(x):
            entry = x + '/'+f
            if os.path.isfile(entry) and f[-4:]=='.jpg' and f[0] != '.':
                files.append(entry)
        ffiles.append(files)
    
    if needThum: # サムネイルの作成
        i = 0
        for cat in ffiles:
            # print(folders[i])
            sam = Image.new('RGB', (800,600),(0,0,0))
            row = col = 0
            for rad in cat:
                img = Image.open(rad, 'r')
                thumbnail_size = (100, 100)
                img.thumbnail(thumbnail_size)
                sam.paste(img,(col,row))
                col += 100
                if col == 800:
                    col = 0
                    row += 100
                # plt.imshow(sam)
            sam.save('{}THUM.png'.format(folders[i]), 'PNG')
            print("saved {}".format(folders[i]))
            i +=1

    return folders,ffiles

# (2)画像を表示

# プロット用関数
IMGON = True
def plotimg(img,layout="111"):
    if IMGON is False: return
    if len(img.shape) == 2:
        pltgry(img,layout)
    elif len(img.shape) ==3:
        pltcol(img,layout)

def pltgry(img,layout="111"):
    plt.subplot(layout)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB))

def pltcol(img,layout="111"):
    plt.subplot(layout)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    
def mkparaimage(img1,img2):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    if img1.ndim == 2:
        img11 = np.zeros((h1,w1,3))
        img11[:,:,0] = img11[:,:,1]=img11[:,:,2]=img1
    else:
        img11=img1
    if img2.ndim == 2:
        img22 = np.zeros((h2,w2,3))
        img22[:,:,0]=img22[:,:,1]=img22[:,:,2]=img2
    else:
        img22=img2
    paraimg = 255*np.ones((max(h1,h2),w1+w2+10,3),dtype=np.uint8)
    
    paraimg[0:h1,0:w1,:] = img11
    paraimg[0:h2,w1+10:,:]=img22
    
    return paraimg

# (3) mkparaimage で２枚並べた画像を表示
def imshowpara(img1,img2):
    plotimg(mkparaimage(img1,img2))

# (4) マージンをつける
def makemargin(img,mr=2):
    h,w = img.shape[:2]
    w2 = mr*w
    h2 = mr*h
    x1 = int((w2-w)/2)
    y1 = int((h2-h)/2)
    img2 = np.zeros((h2,w2),np.uint8)
    img2[y1:y1+h,x1:x1+w] = img
    return img2

# (5) 処理結果画像（fimg)に処理前画像（bimg)の輪郭を描く
def draw2(bimg,fimg):
    if len(bimg.shape)==3:
        bimg2 = cv2.cvtColor(bimg,cv2.COLOR_BGR2GRAY)
    else:
        bimg2 = bimg.copy()
    if len(fimg.shape)==3:
        fimg2 = fimg.copy()
    else:
        fimg2 = cv2.cvtColor(fimg,cv2.COLOR_GRAY2BGR)
    # 処理前画像の輪郭
    # canvas = np.zeros_like(fimg2)
    canvas = fimg2.copy()
    _ret,bwimg = cv2.threshold(bimg2,128,255,cv2.THRESH_BINARY) # 白画素は255にする
    _img,cnt,_hierarchy = cv2.findContours(bwimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # canvas = cv2.drawContours(canvas, cnt, -1, (255,230,230), thickness=-1)
    canvas = cv2.drawContours(canvas, cnt, -1, (255,0,200), thickness=1)
    # return cv2.addWeighted(fimg2, 0.5, canvas, 0.5,0)
    return canvas[:,:,::-1] # opencv -> PIL

# (6) (x1,y1)から（x2,y2) に向かう直線のX軸に対する角度
def getDegreeOfALine(x1,y1,x2,y2):
        dx = x2-x1
        dy = y2-y1
        if dx == 0 :
            if dy > 0:
                deg = 90
            else:
                deg = -90
        elif dx >0:
            deg = 180.0*np.arctan(dy/dx)/np.pi
        else:
            deg = 180*(1+np.arctan(dy/dx)/np.pi)
        return deg
        
# (7) (x1,y1)から（x2,y2) に向かう直線の延長上の十分離れた２点の座標
def getTerminalPsOnLine(x1,y1,x2,y2):
        dx = x2-x1
        dy = y2-y1
        s1 = int(x1 + 1000*dx)
        t1 = int(y1 + 1000*dy)
        s2 = int(x1 - 1000*dx)
        t2 = int(y1 - 1000*dy)
        return s1,t1,s2,t2
    
# (8) ガウスぼかし、膨張収縮、輪郭近似で形状を整える関数
# 形状の細かな変化をガウスぼかし等でなくして大まかな形状にする関数
def RDreform(img,ksize=5,shrink=SHRINK,nsize=4*UNIT):
    
    # ガウスぼかしを適用してシルエットを滑らかにする
    img2 = cv2.GaussianBlur(img,(ksize,ksize),0) # ガウスぼかしを適用
    _ret,img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY) # ２値化
    # 収縮・膨張によりヒゲ根を除去する
    area0 = np.sum(img2) # img2 の画素数*255 になるはず
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize)) # 円形カーネル
    tmpimg = cv2.erode(img2,kernel,iterations = 1) # 収縮１回目
    area1 = np.sum(tmpimg) # 収縮したので area0 より少なくなる
    n = 1
    while area1  > shrink*area0: # 面積が SHRINK倍以下になるまで繰り返す
        tmpimg = cv2.erode(tmpimg,kernel,iterations = 1)
        area1 = np.sum(tmpimg) 
        n += 1
    img3 = cv2.dilate(tmpimg,kernel,iterations = n) # 同じ回数膨張させる
    # あらためて輪郭を求め直す
    # まず最大の領域を取り出す
    _lnum, labelimg, cnt, _cog =cv2.connectedComponentsWithStats(img3)
    areamax = np.argmax(cnt[1:,4])+1 # ０番を除く面積最大値のインデックス
    img3 = np.zeros_like(img3)
    img3[labelimg==areamax]=255
    _img,cnt,_hierarchy = cv2.findContours(img3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #  あらためて輪郭を抽出
    outimg = np.zeros_like(img3)
    # outimg = cv2.drawContours(outimg, cnt, 0, 255, thickness=-1) 
    # 近似輪郭を求めて描きなおす
    # area = cv2.contourArea(cnt[0]) # 面積
    perimeter = cv2.arcLength(cnt[0],True) # 周囲長
    epsilon = CONTOURS_APPROX*perimeter # 周囲長をもとに精度パラメータを決定
    # 概形抽出
    approx = cv2.approxPolyDP(cnt[0],epsilon,True)
    outimg = cv2.drawContours(outimg, [approx], 0, 255, thickness=-1) 

    return outimg

# 重心の位置を求める
def getCoG(img):
    _lnum, _img, cnt, cog = cv2.connectedComponentsWithStats(img)
    areamax = np.argmax(cnt[1:,4])+1 # ０番を除く面積最大値のインデックス
    c_x,c_y = np.round(cog[areamax]) # 重心の位置を丸めて答える
    # x,y,w,h,areas = cnt[areamax] # 囲む矩形の x0,y0,w,h,面積
    return c_x,c_y,cnt[areamax]

# (9)重心と先端の位置を返す関数
#   先端位置はシルエットをガウスぼかしで滑らかにした上で曲率の高い場所
def getCoGandTip(src, showResult=False, useOldImage=True):    
    # useOldImage = True なら元の画像を使って結果を表示、Falseなら滑らかにした画像
    img = makemargin(src) # 作業用のマージンを確保
    img2 = img.copy() # 加工前の状態を保存
    # （あとでぼかすが、ぼかす前の）元画像の最大白領域の面積とバウンディングボックスを求める
    c_x,c_y,(_x0,y0,w,h,areas) = getCoG(img)
    radishwidth = areas/np.sqrt(w*w+h*h) # 面積をバウンディングボックスの対角の長さで割ることで大根の幅を大まかに見積もる
    # ガウスぼかしを適用してシルエットを滑らかにする
    ksize = int(GAUSSIAN_RATE1*radishwidth)*2+1 # ぼかし量  元の図形の幅に応じて決める
    img = cv2.GaussianBlur(img,(ksize,ksize),0) # ガウスぼかしを適用
    # ２値化してシルエットを求め直す
    _ret,img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) # ２値化
    
    # コア全体の重心の位置を求める
    c_x,c_y,(_x0,y0,_w,h,areas) = getCoG(img)
    # 全体を囲む矩形の中間の高さ
    h2 = int(h/2)
    # Harris コーナ検出
    himg = np.float32(img)
    himg = cv2.cornerHarris(himg,blockSize=3,ksize=3,k=0.04)
    # コーナー度合いが最大の領域を求める
    wimg = np.zeros_like(img)
    wimg[himg>=HARRIS_PARA*himg[y0+h2:,:].max()]=255 # 下半分のコーナー度最大値の領域を２５５で塗りつぶす。
    # 最大値に等しい値の領域が１点とは限らないし、いくつかの点の塊になるかもしれない
    _lnum, _img, cnt, cog = cv2.connectedComponentsWithStats(wimg[y0+h2:,:])
    areamax = np.argmax(cnt[1:,4])+1 # ０番を除く面積最大値のインデックス
    t_x,t_y = np.round(cog[areamax]) # 重心の位置
    t_y += y0+h2

    # コーナーの場所のマーキング（デバッグ用）
    # himg = cv2.dilate(himg,None,iterations = 3)
    # img3[himg>=HARRIS_PARA*himg.max()]=[0,0,255]

    if showResult: # 
        if useOldImage:
            img3 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
        else:
            img3 = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        plt.figure(figsize=(10,7),dpi=75)
        img3=cv2.circle(img3,(int(t_x),int(t_y)),5,(0,255,0),2)
        img3=cv2.circle(img3,(int(c_x),int(c_y)),5,(255,255,0),2)
        x1,y1,x2,y2= getTerminalPsOnLine(c_x,c_y,t_x,t_y)
        img3=cv2.line(img3,(x1,y1),(x2,y2),(255,0,255),2)                 
        img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
        plt.subplot(122), plt.imshow(img3)
        plt.show()
        
    # 結果を返す (c_x,c_y) 重心　　(t_x,t_y)  先端の位置 img2 滑らかになったシルエット画像
    dx = int(img.shape[1]-src.shape[1])/2
    dy = int(img.shape[0]-src.shape[0])/2
    c_x,c_y,t_x,t_y = c_x-dx, c_y-dy, t_x-dx, t_y-dy 
    return c_x,c_y,t_x,t_y

# (10) 回転した上でマージンをカットした画像を返す
def roteteAndCutMargin(img,deg,c_x,c_y): 
    # 非常に稀であるが、回転すると全体が描画領域外に出ることがあるので作業領域を広く確保
    # mat = cv2.getRotationMatrix2D((x0,y0), deg-90, 1.0) # アフィン変換マトリクス
    bigimg = makemargin(img,mr=10) # 作業用のマージンを確保
    h3,w3 = img.shape[:2]
    h4,w4 = bigimg.shape[:2]
    
    if deg != 0:
        mat = cv2.getRotationMatrix2D((c_x+(w4-w3)/2,c_y+(h4-h3)/2), deg, 1.0) # アフィン変換マトリクス
        # アフィン変換の適用
        bigimg = cv2.warpAffine(bigimg, mat, (0,0),1)

    # 再び最小矩形を求めて切り出す。ただし、マージンを５つける
    _nLabels, _labelImages, data, _center = cv2.connectedComponentsWithStats(bigimg) 
    ami = np.argmax(data[1:,4])+1 # もっとも面積の大きい連結成分のラベル番号　（１のはずだが念の為）
    resultimg = bigimg[data[ami][1]-5:data[ami][1]+data[ami][3]+5,data[ami][0]-5:data[ami][0]+data[ami][2]+5]

    return resultimg

# (11) 重心から上の重心と、重心位置で断面の中心を返す関数
#   この関数ではぼかしは行わない。
def getUpperCoGandCoC(src):
    _lnum, _img, cnt, cog = cv2.connectedComponentsWithStats(src)
    ami = np.argmax(cnt[1:,4])+1 
    _c_x,c_y = np.round(cog[ami]) # 重心
    halfimg = src[:int(c_y),:].copy() # 重心位置から上を取り出す。
    _lnum, _img, cnt, cog = cv2.connectedComponentsWithStats(halfimg)
    ami =  np.argmax(cnt[1:,4])+1 
    uc_x,uc_y = np.round(cog[ami]) # 上半分の重心
    sliceindex = np.where(src[int(c_y)]!=0) # 重心断面の白画素数位置
    left = np.min(sliceindex) #  断面における最も左の白画素位置
    right = np.max(sliceindex) #  断面における最も右の白画素位置
    ccx = int((left+right)/2) #  断面中央位置
    return uc_x,uc_y,ccx,c_y

# (12) シルエット画像の標準化
# 画像サイズをある程度揃えたい

# ２値画像が前提。最大白領域のみを抽出。margincut=True なら、バウンディングボックスで切り出す。
def getMaxAreaComponents(img, margincut=True):  
    _ret,img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY) # ２値化
    _image, contours, _hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[np.argmax([len(c) for c in contours])] # 最も長い輪郭
    resultimg = np.zeros_like(img) 
    resultimg = cv2.drawContours(resultimg, [cnt], -1, 255, thickness=-1) 
    if margincut:
        _nLabels, labelImages, data, _center = cv2.connectedComponentsWithStats(resultimg)
        ami = np.argmax(data[1:,4])+1 # もっとも面積の大きい連結成分のラベル番号　（１のはずだが念の為）
        img = np.zeros_like(resultimg) 
        img[labelImages==ami]=255# 最大面積の画素のみ抽出
        resultimg = img[data[ami][1]:data[ami][1]+data[ami][3],data[ami][0]:data[ami][0]+data[ami][2]] # マージンなしで切り出して返す
    return resultimg
  

# 長辺の1.5倍サイズの枠の中央に対象を配置した画像を返す
def makeUnitImage(img,unitSize=UNIT):
    # 長辺が UNIT ピクセルになるよう縮小し、(1.5xUNIT)x(1.5xUNIT)の画像の中央に配置する。
    h,w = img.shape[:2]
    s_r = unitSize/w if w > h else unitSize/h #  縮小率    
    rsh,rsw = int(s_r*h),int(s_r*w) # リサイズ後のサイズ
    x0 = int((1.5*unitSize-rsw)/2) # はめ込みの基準点
    y0 = int((1.5*unitSize-rsh)/2)
    canvas = np.zeros((int(1.5*unitSize),int(1.5*unitSize)),np.uint8) # キャンバスの確保
    canvas[y0:y0+rsh,x0:x0+rsw] = cv2.resize(img,(rsw,rsh)) # リサイズして中央にはめ込み
    return canvas

def getstandardShape(src, unitSize=UNIT, krate=GAUSSIAN_RATE2,thres = 0.25, setrotation = 0, norotation = False):
    # src 画像, unitSize 長軸をこの長さに正規化、thres 方向のx成分がこれ以下なら回転処理を施さない  setrotation 強制回転角
    img = getMaxAreaComponents(src, margincut=True)
    # 長辺が UNIT ピクセルになるよう縮小し、(1.5xUNIT)x(1.5xUNIT)の画像の中央に配置する。   
    img = makeUnitImage(img,unitSize=unitSize*2) # ヒゲがあるのでこの時点では大きめにして作業を進める
    # 輪郭抽出
    _image, contours, _hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[np.argmax([len(c) for c in contours])] # 最も長い輪郭
    
    # 輪郭線全体を直線近似して向きを決定する
    [vx,_vy,_x,_y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)

    if norotation or np.abs(vx) <= thres and setrotation == 0: # 非回転が指定されているか、回転角の指定がなく、角度が浅いなら回転させない
        img = roteteAndCutMargin(img,0,0,0)
    else:
        if setrotation != 0: # 回転角が指定されているならその角度
            deg = setrotation
            c_x,c_y,_x,_y,_w,_h = getCoG(img) # 重心 c_x,c_y
        else: # 何も指定がなく、傾いている場合
            # 重心と先端の位置を調べてそれを垂直になるように回転する
            c_x,c_y,t_x,t_y = getCoGandTip(img,showResult=False)
            deg = getDegreeOfALine(c_x,c_y,t_x,t_y)-90 
            if abs(deg) > 45: # 45度を超えるとすると先端位置を誤検出しているに違いないので回転させるのはやめる
                deg = 0
        # 重心と先端を結ぶラインがY軸となるように回転し余白はカット
        img = roteteAndCutMargin(img,deg,c_x,c_y)

    # 大きさを標準化したいが、無駄に根が長いと相対的に重要部分が小さくなるのでまず根を削る
    # 作業用のマージンを確保
    tmpimg = makemargin(img)     
    # 少し形を整えておく RDreform は上の方で定義してある
    tmpimg = RDreform(tmpimg)
    # reform で根が削られているのであらためてバウンディングボックスで囲んで取り出す。
    tmpimg = getMaxAreaComponents(tmpimg, margincut=True)  
    # 指定サイズに拡大縮小
    tmpimg = makeUnitImage(tmpimg,unitSize=unitSize) 
            
    # 最後にもう一度だけガウスぼかしを適用してシルエットを滑らかにする
    ksize = 2*int(( krate*unitSize)/2)+1 # ぼかし量  元の図形の幅に応じて決める
    tmpimg = cv2.GaussianBlur(tmpimg,(ksize,ksize),0) # ガウスぼかしを適用
    # ２値化し、最大の輪郭だけを抜き出して描いておく
    resultimg = getMaxAreaComponents(tmpimg, margincut=False)
    
    return resultimg

# (13) 輪郭点のサンプリング
def threeLinesSeq(src, showImage =  False, n_samples=16): 
    global tupple
    _img,cnt,_hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #  あらためて輪郭を抽出
    x0,y0,_w,h = cv2.boundingRect(cnt[0])
    x = cnt[0][:,0][:,0] # 輪郭のX座標の列
    y = cnt[0][:,0][:,1] # 輪郭のY座標の列
    # 輪郭点を各Y座標について左右ペアで求める。同時にそれらの中点の座標を加えた３つ組をリスト化する。
    # 同じY座標を持つ点が３つ以上ある場合は、もっとも左ともっとも右の点を仮の対応点として抽出
    cut10 = y0+int(10*h/100) # シルエッ上端から高さ１０％の位置
    cut20 = y0+int(20*h/100) # シルエッ上端から高さ2０％の位置
    cut100 = y0+int(100*h/100)
    ll,rl = [],[]
    # まず、高さに対して１０％から100％の位置のシルエット輪郭とその中央点はサンプル点として無条件に採用
    for i in range(cut10,cut100,1):
        xs = x[y==i]
        l = min(xs)
        r = max(xs)
        ll.append([l,i])
        rl.append([r,i])
    # 上記に含まれない輪郭のうち、高さに対して上端部から１０％と２０％の位置の中点を結ぶ直線の
    # 法線より下にある輪郭点追加採用する。
    xs = x[y==cut10] # 高さ１０％の位置にある輪郭点のリスト
    xc10 = (min(xs)+max(xs))/2 # 中央の点の座標（これはfloat)
    xs = x[y==cut20] # 高さ2０％の位置にある輪郭点のリスト    
    xc20 = (min(xs)+max(xs))/2 # 中央の点の座標
    # 点(xc10, cut10) における（仮の）中心軸の法線の傾きは
    #  (xc10-xc20)/(cut20-cut10)で表せる
    #  傾きがマイナスの時は左、プラスの時は右の点を通るこの傾きの直線より下にある輪郭を追加採用する。
    slope = (xc10-xc20)/(cut20-cut10)
    add = []
    x0 = ll[0][0] if slope < 0 else rl[0][0] #  傾きが負なら左、正なら右が基準
    for y1 in range(y0,cut10,1):
        xs = x[y==y1]
        x1 = max(xs) if slope < 0 else min(xs)  # 追加候補点のx座標
        if y1 >= slope*(x1-x0)+cut10 :
            add.append([x1,y1]) if slope < 0 else add.append([x1,y1])
    if slope < 0:
        rl =  add+rl
    elif slope > 0:
        ll = add+ll
        
    # 左右のサンプル点のうち数の少ない方に合わせて対応づけを行い、両者の距離をその点での太さとみなす。
    # 太さの最大値を求め、太さが２０％未満のサンプル点は除去する。
    tupple = []
    # print("len(ll),{},len(rl),{}".format(len(ll),len(rl)))
    num = len(ll) if len(ll) < len(rl) else len(rl)
    for i in range(num):
        if len(ll) < len(rl):
            m = int(i*len(rl)/len(ll))
            d = np.sqrt((ll[i][0]-rl[m][0])**2+(ll[i][1]-rl[m][1])**2)
            tupple.append([[d],ll[i],[(ll[i][0]+rl[m][0])/2,(ll[i][1]+rl[m][1])/2],rl[m]])
        else:
            m = int(i*len(ll)/len(rl))
            d = np.sqrt((ll[m][0]-rl[i][0])**2+(ll[m][1]-rl[i][1])**2)
            tupple.append([[d],ll[m],[(ll[m][0]+rl[i][0])/2,(ll[m][1]+rl[i][1])/2],rl[i]])

    tupple = np.array(tupple)
    dmax_index = np.argmax(tupple[:,0])
    dmax = tupple[dmax_index][0][0]
    correct = []
    for i in range(len(tupple)):
        if tupple[i][0] > 0.2*dmax:
            correct.append(tupple[i]) # 幅が２０％を切るものはカットする
    tupple = np.array([[d,l,c,r] for d,l,c,r in correct])
            
    # 確認のために描画してみる。
    if showImage:
        test = cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)
        for d,l,c,r in correct:
            cv2.circle(test,(int(l[0]),int(l[1])),1,(255,0,255),-1)
            cv2.circle(test,(int(c[0]),int(c[1])),1,(255,255,0),-1)
            cv2.circle(test,(int(r[0]),int(r[1])),1,(0,255,0),-1)
        (x1,y1) = (int(tupple[dmax_index][1][0]),int(tupple[dmax_index][1][1]))
        (x2,y2) = (int(tupple[dmax_index][3][0]),int(tupple[dmax_index][3][1]))
        cv2.line(test,(x1,y1),(x2,y2),(0,0,255),1)
        plt.figure(figsize=(6,6),dpi=100)
        plt.imshow(test[:,:,::-1])
        
    llist = np.array([l for d,l,c,r in tupple])
    rlist = np.array([r for d,l,c,r in tupple])
    clist = np.array([c for d,l,c,r in tupple])
        
    n = int(len(tupple)/n_samples-1)
    llist = np.array(llist)[::n]
    rlist = np.array(rlist)[::n]
    clist = np.array(clist)[::n]
    
    return llist,rlist,clist,dmax

# (14) ベジエフィッティング　（この関数は　firBezierCurveN があるのでもう使わないが、古いプログラムのため残す）
'''def fitBezierCurve3(points,precPara=0.01,mode=2, debugmode=False):
    # mode = 2：両端固定、　mode =3 : 下のみ固定, mode4 :  全制御点自由
    # ベジエ曲線を定義するのに使うシンボルの宣言
    P = [Symbol('P' + str(i)) for i in range(4)]
    px,py =var('px:4'),var('py:4')
    for i in range(4):
        P[i] = Matrix([px[i],py[i]]) 
    t = symbols("t")
    
    # いわゆる自乗誤差の一般式
    s,t= symbols('s,t')
    loss1 = (s - t)**2
    # 最小自乗法の目的関数の一般式
    def lossfunc(listA,listB):
        return sum([loss1.subs([(s,a),(t,b)]) for (a,b) in zip(listA,listB)])/2
    
    # ３次のベジエ曲線の定義式制御点 P0~P3 とパラメータ　　t　の関数として定義
    bez3 = (1-t)**3*P[0] + 3*(1-t)**2*t*P[1] + 3*(1-t)*t**2*P[2] + t**3*P[3]
    
    # 初期の推定パラメータの決定
    ## サンプル点間の差分を求める
    points1 = points[1:] #  ２つ目から後ろのサンプル点
    ds = points1-points[:-1] # サンプル点間の差分ベクトル
    la = [np.sqrt(e[0]*e[0]+e[1]*e[1]) for e in ds] # サンプル点間の直線距離のリスト
    axlength = np.sum(la) # 折れ線近似による経路長
    tpara0 = np.zeros(len(points),np.float32) # パラメータ格納配列
    tpara = tpara0.copy()
    tpara[0]=0.0 # 最初の点のパラメータ推定値は０とする
    for i in range(len(la)):
        tpara[i+1] = tpara[i]+la[i] # 各サンプル点での積算経路長
    tpara = tpara/axlength # 全経路長で割ってパラメータとする　（０〜１）

    #  パラメトリック曲線　linefunc 上で各サンプル点に最寄りの点のパラメータを対応づける
    def refineTpara(pl,linefunc):
        (funcX,funcY) = linefunc # funcX,funcY は t の関数
        # 各サンプル点に最も近い曲線上の点のパラメータ t を求める。
        trange = np.arange(-0.1,1.1,0.005) # 推定範囲は -0.1 〜　１．１
        onpoints = [[s,funcX.subs(t,s),funcY.subs(t,s)] for s in trange] # 曲線上の点
        tpara = np.zeros(len(pl),np.float32) # 新しい 推定 t パラメータのリスト用の変数のアロケート
        refineTparaR(pl,tpara,0,len(pl),0,len(onpoints),onpoints)
        return tpara

    # 範囲全体をサーチするのはかなり無駄なので、約６割ぐらいに狭める
    def srange(n,stt,end,smin,smax):
        if end - stt < 3 : # 残り３点未満なら全域サーチする
            left,right = smin,smax
        else:
            left = smin+0.5*(smax-smin)*(n-stt)/(end-stt-1)
            right= smax - 0.5*(smax-smin)*(end-n-1)/(end-stt-1)
        left = (int(left-0.5)   if int(left-0.5) > smin else smin)
        right = (int(right+0.5) if int(right+0.5) < smax else smax)
        return left, right
        
    #  探索範囲内での対応づけ再帰関数
    # pl 点列、(stt,end) 推定対象範囲（番号）, (smin,smax) パラメータの探索範囲         
    def refineTparaR(pl,tpara,stt,end,smin,smax,onpoints):
        if stt >end:
            return 
        else:
            nmid = int((end+stt)/2) # 探索対象の中央のデータを抜き出す
            px,py = points[nmid] # 中央のデータの座標
            smin1, smax1 = srange(nmid,stt,end,smin,smax)
            differ = onpoints[smin1:smax1].copy() 
            differ = differ - np.array([0.0,px,py]) # 差分の配列
            distance = [x*x+y*y for _t,x,y in differ] # 自乗誤差の配列
            nearest_i = smin1+np.argmin(distance) # 誤差最小のインデックス
            tpara[nmid] = onpoints[nearest_i][0] # 中央点のパラメータが決定
            if nmid-stt >= 1 : # 左にまだ未処理の点があるなら処理する
                refineTparaR(pl,tpara, stt,nmid,smin,nearest_i,onpoints)
            if end-(nmid+1) >=1 : # 右にまだ未処理の点があるなら処理する
                refineTparaR(pl,tpara,nmid+1,end,nearest_i+1,smax,onpoints) 
                
    while True:
        linepoints = [bez3.subs(t,t_) for t_ in tpara] # 曲線上の点列
        linepointsX = [x  for  [x,y] in linepoints] # X
        linepointsY = [y  for  [x,y] in linepoints] # Y
        EsumX = lossfunc(listA=points[:,0],listB=linepointsX) #  X方向のずれの評価値
        EsumY = lossfunc(listA=points[:,1],listB=linepointsY) #  Y 方向のずれの評価値
        # px0,px1, px2, px3, py1, py2,py3,py4 で偏微分
        if  mode<4:
            # P3は固定なので座標を代入
            EsumX = EsumX.subs(px[3],points[-1][0])
            EsumY = EsumY.subs(py[3],points[-1][1])
        if mode<3:
            # P0も固定なので座標を代入
            EsumX = EsumX.subs(px[0],points[0][0])
            EsumY = EsumY.subs(py[0],points[0][1])
        dx_1 = diff(EsumX,px1)
        dx_2 = diff(EsumX,px2)
        dy_1 = diff(EsumY,py1)
        dy_2 = diff(EsumY,py2)
        if mode>2:
            dx_0 = diff(EsumX,px0)
            dy_0 = diff(EsumY,py0)
        if mode>3:
            dx_3 = diff(EsumX,px3)
            dy_3 = diff(EsumY,py3)

        # 連立させて解く
        if mode == 4:
            resultX = solve([dx_0,dx_1,dx_2,dx_3],[px[0],px[1],px[2],px[3]])
            resultY = solve([dy_0,dy_1,dy_2,dy_3],[py[0],py[1],py[2],py[3]])
        elif mode ==3:
            resultX = solve([dx_0,dx_1,dx_2],[px[0],px[1],px[2]])
            resultY = solve([dy_0,dy_1,dy_2],[py[0],py[1],py[2]])
        else : # mode ==2
            resultX = solve([dx_1,dx_2],[px[1],px[2]])
            resultY = solve([dy_1,dy_2],[py[1],py[2]])
            
        # 解をベジエの式に代入
        if mode == 4:
            bezresX = bez3[0].subs([(px[0],resultX[px0]),(px[1],resultX[px1]),(px[2],resultX[px2]),(px[3],resultX[px3])])
            bezresY = bez3[1].subs([(py[0],resultY[py0]),(py[1],resultY[py1]),(py[2],resultY[py2]),(py[3],resultY[py3])])
        elif mode == 3:
            bezresX = bez3[0].subs([(px[0],resultX[px0]),(px[1],resultX[px1]),(px[2],resultX[px2]),(px[3],points[-1][0])])
            bezresY = bez3[1].subs([(py[0],resultY[py0]),(py[1],resultY[py1]),(py[2],resultY[py2]),(py[3],points[-1][1])])
        else : # mode ==2
            bezresX = bez3[0].subs([(px[0],points[0][0]),(px[1],resultX[px1]),(px[2],resultX[px2]),(px[3],points[-1][0])])
            bezresY = bez3[1].subs([(py[0],points[0][1]),(py[1],resultY[py1]),(py[2],resultY[py2]),(py[3],points[-1][1])])
            
        rx,ry = resultX,resultY
        if mode == 4:
            cpx = [rx[px0],rx[px1],rx[px2],rx[px3]]
            cpy = [ry[py0],ry[py1],ry[py2],ry[py3]]
        elif mode == 3:
            cpx = [rx[px0],rx[px1],rx[px2],points[-1][0]]
            cpy = [ry[py0],ry[py1],ry[py2],points[-1][1]]
        else: # mode==2
            cpx = [points[0][0],rx[px1],rx[px2],points[-1][0]]
            cpy = [points[0][1],rx[px1],ry[py2],points[-1][1]]
            
        tpara0 = tpara.copy()
        tpara = refineTpara(points,(bezresX,bezresY))
        diffpara = 0
        for i in range(len(tpara)) :
            diffpara += np.sqrt((tpara[i]-tpara0[i])**2)
        diffpara = diffpara/len(tpara)
        if debugmode:
            print("diffpara",diffpara)
        if diffpara < precPara/100:
            break
    
    return np.array(cpx),np.array(cpy),bezresX,bezresY,tpara
    # cpx,cpy → ４つの制御点、bezresX,bezresY ベジエ曲線の定義式
    # tpara 制御点
'''
# 解なしの部分に np.inf が入っているのでその抜けを前後から補完推定してデータを埋める
def eraseinf(plist):
    while np.sum(plist) == np.inf: # 無限を含むなら除去を繰り返す
        for i in range(len(plist)):
            if np.sum(plist[i]) == np.inf :
                print("欠",end="")
                if (i !=0 and i !=len(plist)-1) and np.sum(plist[i-1]+plist[i+1]) != np.inf: # 当該は無限で、前後は無限ではない場合
                    plist = np.r_[plist[0:i],[(plist[i-1]+plist[i+1])/2],plist[i+1:]]
                elif len(plist[i:])>=3 and np.sum(plist[i+1]+plist[i+2]) != np.inf:
                    plist = np.r_[plist[0:i],[plist[i+2]-2*(plist[i+2]-plist[i+1])],plist[i+1:]]
                elif len(plist[0:i])>=2 and np.sum(plist[i-1]+plist[i-2]) != np.inf:
                    plist = np.r_[plist[0:i],[plist[i-2]-2*(plist[i-2]-plist[i-1])],plist[i+1:]]
    print("")
    return plist
    
#  (14-2) N次ベジエフィッティング
def fitBezierCurveN(points,precPara=0.01,N=5, openmode=False,debugmode=False):
    
    # order 次数、openmode: 両端点フリー、Falseの時は両端点固定
    # ベジエ曲線を定義するのに使うシンボルの宣言

    P = [Symbol('P' + str(i)) for i in range(N+1)]
    px = [var('px'+str(i)) for i in range(N+1)]
    py = [var('py'+str(i)) for i in range(N+1)]
    dx_ = [var('dx_'+str(i)) for i in range(N+1)]
    dy_ = [var('dy_'+str(i)) for i in range(N+1)]
 
    # inf データの部分を補完する
    points = eraseinf(points)
 
    for i in range(N+1):
        P[i] = Matrix([px[i],py[i]]) 
    
    # いわゆる自乗誤差の一般式
    s,t= symbols('s,t')
    loss1 = (s - t)**2
    # 最小自乗法の目的関数の一般式
    def lossfunc(listA,listB):
        return sum([loss1.subs([(s,a),(t,b)]) for (a,b) in zip(listA,listB)])/2
    v = var('v')
    # N次のベジエ曲線の定義式制御点 P0~PN とパラメータ　　t　の関数として定義
    v = 1-t
    bezN = Matrix([0,0])
    for i in range(0,N+1):
        bezN = bezN + binomial(N,i)*v**(N-i)*t**i*P[i]

    # 初期の推定パラメータの決定
    ## サンプル点間の差分を求める
    '''points1 = points[1:] #  ２つ目から後ろのサンプル点
    ds = points1-points[:-1] # サンプル点間の差分ベクトル
    la = [np.sqrt(e[0]*e[0]+e[1]*e[1]) for e in ds] # サンプル点間の直線距離のリスト
    axlength = np.sum(la) # 折れ線近似による経路長
    npoints = len(points)
    tpara0 = np.zeros(npoints,np.float32) # パラメータ格納配列
    tpara = tpara0.copy()
    tpara[0]=0.0 # 最初の点のパラメータ推定値は０とする
    for i in range(len(la)):
        tpara[i+1] = tpara[i]+la[i] # 各サンプル点での積算経路長
    tpara = tpara/axlength # 全経路長で割ってパラメータとする　（０〜１）
    '''
    tpara = [cv2.arcLength(points[:i+1],False)  for i in range(len(points))]/axlength #上の１１行をこの１行で置き換えた
    
    #  パラメトリック曲線　linefunc 上で各サンプル点に最寄りの点のパラメータを対応づける
    def refineTparaN(pl,linefunc,npoints=50):
        (funcX,funcY) = linefunc # funcX,funcY は t の関数
        # 各サンプル点に最も近い曲線上の点のパラメータ t を求める。
        # trange = np.arange(-0.1,1.1,1/(2*npoints)) # 推定範囲は -0.1 〜　１．１ サンプル数の2倍の候補点を設定
        if openmode:
            trange0 = np.arange(-0.1,0.25,1/(4*npoints))
            trange2 = np.arange(0.75,1.1,1/(4*npoints))
            trange2 = np.r_[trange2,[1.1]]
        else:
            trange0 = np.arange(0,0.25,1/(4*npoints))
            trange2 = np.arange(0.75,1,1/(4*npoints))
            trange2 = np.r_[trange2,[1]]
        trange1 = np.arange(0.25,0.75,1/(2*npoints))
        trange = np.r_[trange0[:-1],trange1,trange2[1:]] # 最初の１/4 と最後の1/4 は近似精度が落ちるので２倍の点を用意する。
        onpoints = [[s,funcX.subs(t,s),funcY.subs(t,s)] for s in trange] # 曲線上の点
        tpara = np.zeros(len(pl),np.float32) # 新しい 推定 t パラメータのリスト用の変数のアロケート
        refineTparaR(pl,tpara,0,len(pl),0,len(onpoints),onpoints) #  ０からなので、len(pl) 番はない　len(onpoints)番もない
        return tpara

    # 範囲全体をサーチするのはかなり無駄なので、約６割ぐらいに狭める
    def srange(n,stt,end,smin,smax):
        if RPARA == 1:
            return smin,smax
        else:
            if end - stt < 3 : # 残り３点未満なら全域サーチする
                left,right = smin,smax
            else:
                srange = smax-smin
                left = smin + srange*((n-stt)/(end-stt) - RPARA/2 if (n-stt)/(end-stt) > RPARA/2 else 0)  
                left = smin+(n-stt)-1 if left < smin+(n-stt)-1 else left   # n番の左に未定がn-stt個あるので最低その分残してしておかないといけない
                right = smax - srange*((end-n)/(end-stt) - RPARA/2 if (end-n)/(end-stt) > RPARA/2 else 0)
                right = smax-(end-n) if right > smax-(end-n) else right
            left = (int(left-0.5)   if int(left-0.5) > smin else smin) # 0.5 は　RPARAとは無関係なので注意
            right = (int(right+0.5) if int(right+0.5) < smax else smax)
            return left, right
       
    #  探索範囲内での対応づけ再帰関数
    # pl 点列、(stt,end) 推定対象範囲（番号）, (smin,smax) パラメータの探索範囲         
    def refineTparaR(pl,tpara,stt,end,smin,smax,onpoints):
        nmid = int((end+stt-1)/2) # 探索対象の中央のデータを抜き出す
        px,py = points[nmid] # 中央のデータの座標
        smin1, smax1 = srange(nmid,stt,end,smin,smax) # 初期サーチ範囲を決める
        if  smin1 == smax1 and stt == end:
            nearest_i = smin1 # = smax1
        else:
            while True:
                zahyo = (np.array(onpoints[smin1:smax1]).copy())[:,1:] # onpoints リストの座標部分のみ取り出し            
                differ = zahyo - np.array([px,py]) # 差分の配列
                distance = [x*x+y*y for x,y in differ] # 自乗誤差の配列
                nearest_i = smin1+np.argmin(distance) # 誤差最小のインデックス
                if (nearest_i > smin1 and nearest_i < smax1) or nearest_i == smin or nearest_i == smax:
                    if nearest_i - smin < nmid-stt: # 小さい側の残りリソースが不足
                        nearest_i = nearest_i+(nmid-stt)-(nearest_i-smin) 
                    elif smax-nearest_i < end-nmid: # 大きい側のリソースが不足
                        nearest_i = nearest_i-(end-nmid)+(smax-nearest_i)
                    break
                if smin1==smax1:
                    print("SAME")
                if nearest_i == smax1:
                    print(">",end=="")
                    (smin1,smax1) = (smax1-1, smax1 + 3) if smax1 + 3 < smax else (smax1-1,smax)
                elif nearest_i == smin1 :
                    print("<",end="")
                    (smin1,smax1) = (smin1 - 3, smin1+1) if smin1 - 3 > smin else (smin,smin1+1)
                 
        # nmid番のサンプル点に最も近い点のパラメータは、onpoints の nearest_i 番と確定
        tpara[nmid] = onpoints[nearest_i][0] # 中央点のパラメータが決定
        if nmid-stt >= 1 : # 左にまだ未処理の点があるなら処理する
            refineTparaR(pl,tpara, stt,nmid,smin,nearest_i,onpoints)
        if end-(nmid+1) >=1 : # 右にまだ未処理の点があるなら処理する
            refineTparaR(pl,tpara,nmid+1,end,nearest_i+1,smax,onpoints) 
                
    trynum = 0
    while True:
        trynum += 1
        linepoints = [bezN.subs(t,t_) for t_ in tpara] # 曲線上の点列の式表現
        linepointsX = [x  for  [x,y] in linepoints] # X座標のリスト、式表現
        linepointsY = [y  for  [x,y] in linepoints] # Y座標のリスト、式表現
        EsumX = lossfunc(listA=points[:,0],listB=linepointsX) #  X方向のずれの評価値
        EsumY = lossfunc(listA=points[:,1],listB=linepointsY) #  Y 方向のずれの評価値
        # px0,px1, px2, px3, ... py1, py2,py3 ...で偏微分
        
        if  not openmode : # 両端点を固定
            EsumX = EsumX.subs(px[-1],points[-1][0])
            EsumY = EsumY.subs(py[-1],points[-1][1])
            EsumX = EsumX.subs(px[0],points[0][0])
            EsumY = EsumY.subs(py[0],points[0][1])
        for i in range(0,N+1):
            dx_[i] = diff(EsumX,px[i])
            dy_[i] = diff(EsumY,py[i])  
 
        # 連立させて解く
        if not openmode :
            resultX = solve([dx_[i] for i in range(1,N)],[px[i] for i in range(1,N)])
            resultY = solve([dy_[i] for i in range(1,N)],[py[i] for i in range(1,N)])
        else : 
            resultX = solve([dx_[i] for i in range(N+1)],[px[i] for i in range(N+1)])
            resultY = solve([dy_[i] for i in range(N+1)],[py[i] for i in range(N+1)])
        
        if len(resultX) == 0 or len(resultY) == 0: # 方程式が解けない　非常にまれなケース
            return False,np.array([]),np.array([]),None,None,None
        
        # 解をベジエの式に代入
        if not openmode:
            bezresX = bezN[0].subs([(px[0],points[0][0]),(px[-1],points[-1][0])])
            bezresY = bezN[1].subs([(py[0],points[0][1]),(py[-1],points[-1][1])])
            for i in range(1,N):
                bezresX = bezresX.subs(px[i],resultX[px[i]])
                bezresY = bezresY.subs(py[i],resultY[py[i]])
        else: 
            bezresX = bezN[0]
            bezresY = bezN[1]           
            for i in range(0,N+1):
                bezresX = bezresX.subs(px[i],resultX[px[i]])
                bezresY = bezresY.subs(py[i],resultY[py[i]])
            
        rx,ry = resultX,resultY
        if not openmode:
            cpx = [points[0][0]]+[rx[px[i]] for i in range(1,N)]+[points[-1][0]]
            cpy = [points[0][1]]+[ry[py[i]] for i in range(1,N)]+[points[-1][1]]    
        else: # openmode
            cpx = [rx[px[i]] for i in range(N+1)]
            cpy = [ry[py[i]] for i in range(N+1)]
        
        tpara0 = tpara.copy() # 元の t の推定値
        tpara = refineTparaN(points,(bezresX,bezresY),npoints=npoints) # 新たに推定値を求める
        diffpara = 0
        for i in range(len(tpara)) :
            diffpara += np.sqrt((tpara[i]-tpara0[i])**2) # 変化量の合計
        print(".",end='')
        diffpara = diffpara/len(tpara)
        if debugmode:
            print("TRY {0} diffpara {1:0.5f} : {2:0.5f}".format(trynum,diffpara*100,precPara*1.05**(0 if trynum <=5 else trynum-5)))
        if trynum <= 5:
            if diffpara < precPara/100:
                break
        else:
            if diffpara < precPara/100*1.05**(trynum-5): # 収束しない時のために、条件を徐々に緩めていく
                break
    print("o",end="")
        
    return True,np.array(cpx),np.array(cpy),bezresX,bezresY,tpara
    # cpx,cpy 制御点、bezresX,bezresY ベジエ曲線の定義式
    # tpara 制御点   
    
# (15) 輪郭と軸のサンプルデータ　　data  を３本の3次ベジエ曲線で近似する　（この関数はもう使わないが、古いプログラムのため残す）
'''
def fitBezierAndDraw(data,mode=2,showImage=False,img=None,withImage=False):
    [datal,datac,datar] = data
    
    # 初期の推定パラメータの決定
    datal = cpxl,cpyl,bezXl,bezYl,tpl = fitBezierCurve3(points=datal,precPara=0.01,mode=mode)
    datac = cpxc,cpyc,bezXc,bezYc,tpc = fitBezierCurve3(points=datac,precPara=0.01,mode=mode)
    datar = cpxr,cpyr,bezXr,bezYr,tpr = fitBezierCurve3(points=datar,precPara=0.01,mode=mode)
    
    # 結果の描画
    if showImage:
        drawThreeLines(datal,datac,datar,img=img,withImage=withImage,invertImage=True)
    
    return [cpxl,cpyl,bezXl,bezYl,tpl],[cpxc,cpyc,bezXc,bezYc,tpc],[cpxr,cpyr,bezXr,bezYr,tpr]
'''
# (16) ３本のベジエ曲線を描画する
def drawThreeLines(ldata,cdata,rdata,img=None,withImage=False,invertImage=False):

    [cpxl,cpyl,bezXl,bezYl,_tpl]=ldata # cpxx , cpyx 制御点４つの座標配列、
    [cpxc,cpyc,bezXc,bezYc,_tpc]=cdata # bezXx,bezYx ベジエ 曲線、
    [cpxr,cpyr,bezXr,bezYr,_tpr]=rdata # tpl 制御点のパラメータ
    
    t = symbols("t")
    # t, p0,p1,p2,p3 = symbols("t, p0,p1,p2,p3")
    # ３次のベジエ曲線の定義式制御点 P0~P3 とパラメータ　　t　の関数として定義
    # bez3 = (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3
    
    # 結果の描画
    if True:
        tplinsOver = np.linspace(-0.03, 1.03,55)
        # tplinsLong = np.linspace(-0.3, 1.3,50)
        tplins50 = np.linspace(0, 1, 50)
        if withImage: # 画像付きの場合は長く
            tplinsOver = np.linspace(-0.3, 1.3,50)
        else:
            tplinsOver = np.linspace(-0.1, 1.1,55)           
        plt.figure(figsize=(6,6),dpi=100)
        plt.gca().set_aspect('equal', adjustable='box') # アスペクト比を１：１に
        if invertImage: # Y軸反転の有無
            plt.gca().invert_yaxis() 
        if withImage:
            plt.imshow(192+(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)/4).astype(np.uint8))
        # 左輪郭の描画
        plotx = [bezXl.subs(t,tp) for tp in tplinsOver ]
        ploty = [bezYl.subs(t,tp) for tp in tplinsOver ]
        plt.plot(plotx,ploty,color = 'green') # 左の曲線の余分な分の描画        
        plotx = [bezXl.subs(t,tp) for tp in tplins50 ]
        ploty = [bezYl.subs(t,tp) for tp in tplins50 ]
        plt.plot(plotx,ploty,color='blue') # 左の曲線の０−１部分の描画
        plt.scatter(cpxl,cpyl,color ='blue',marker = '*') #  制御点の描画
        
        # 中心軸の描画

        plotx = [bezXc.subs(t,tp) for tp in tplinsOver ]
        ploty = [bezYc.subs(t,tp) for tp in tplinsOver ]
        plt.plot(plotx,ploty,color = 'green') # 軸の曲線の余分な分の描画)
        plotx = [bezXc.subs(t,tp) for tp in tplins50 ]
        ploty = [bezYc.subs(t,tp) for tp in tplins50 ]
        plt.plot(plotx,ploty,color = 'red')  # 軸の曲線の0-1部分の描画
        plt.scatter(cpxc,cpyc,color ='red',marker = '*') #  制御点の描画)
        
        # 右輪郭の描画
        plotx = [bezXr.subs(t,tp) for tp in tplinsOver ]
        ploty = [bezYr.subs(t,tp) for tp in tplinsOver ]
        plt.plot(plotx,ploty,color = 'green') # 軸の曲線の余分な分の描画))
        plotx = [bezXr.subs(t,tp) for tp in tplins50 ]
        ploty = [bezYr.subs(t,tp) for tp in tplins50 ]
        plt.plot(plotx,ploty,color='purple')  
        plt.scatter(cpxr,cpyr,color='purple',marker = '*')

'''
# (1)画像のリストアップ
listimage(path='シルエット', needThum=False)

# (2)２枚の画像をサイズを並べた画像を作成する
mkparaimage(img1,img2)

plotimg(img,layout="111"):　# グレイでもカラーでも表示

pltgry(img,layout="111"): # グレイ画像の表示

pltcol(img,layout="111"):　# カラー画像の表示

# (3) mkparaimage で２枚並べた画像を表示
imshowpara(img1,img2)

# (4) マージンをつける
makemargin(img,mr=2)

# (5) 処理結果画像（fimg)に処理前画像（bimg)の輪郭を描く
draw2(bimg,fimg)

# (6) (x1,y1)から（x2,y2) に向かう直線のX軸に対する角度
getDegreeOfALine(x1,y1,x2,y2)

# (7) (x1,y1)から（x2,y2) に向かう直線の延長上の十分離れた２点の座標
getTerminalPsOnLine(x1,y1,x2,y2)

# (8) ガウスぼかし、膨張収縮、輪郭近似で形状を整える関数
# 形状の細かな変化をガウスぼかし等でなくして大まかな形状にする関数
RDreform(img,ksize=5,shrink=SHRINK)

# (9)重心と先端の位置を返す関数
#   先端位置はシルエットをガウスぼかしで滑らかにした上で曲率の高い場所
getCoGandTip(src, showResult=False, useOldImage=True):

# (10) 回転した上でマージンをカットした画像を返す
def roteteAndCutMargin(img,deg,c_x,c_y)

# (11) 重心から上の重心と、重心位置で断面の中心を返す関数
#   この関数ではぼかしは行わない。
getUpperCoGandCoC(src)

# (12) シルエット画像の標準化
# 画像サイズをある程度揃えたい
getstandardShape(src, unitSize=UNIT,showResult=False):

# (13) 輪郭点のサンプリング
threeLinesSeq(src, showImage =  False)

# (14) ベジエフィッティング
fitBezierCurve3(points,precPara=0.01,mode=2):

#  (14-2) ４次ベジエフィッティング
fitBezierCurveN(points,precPara1=0.01,N=5,open=False,open=False,debugmode=False):
# open = False：両端固定、それ以外は全制御点自由

# (15) 輪郭と軸のサンプルデータ　　data  を３本のベジエ曲線で近似する
fitBezierAndDraw(data,mode=2)

# (16) ３本のベジエ曲線を描画する
drawThreeLines(ldata,cdata,rdata,img=None,withImage=False)
'''