# # ダイコンの両端点の自動決定
#

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math
# 最寄りの２のべき乗サイズになるよう padding
def getbase2img(img):
    h = img.shape[0]
    w = img.shape[1]
    lh = math.ceil(math.log(h,2))
    lw = math.ceil(math.log(w,2))
    newh = 2** lh
    neww = 2** lw
    print("２のべき乗サイズに変換", h,w,"->",newh,neww)
    if len(img.shape)==3:
        newimg = np.zeros((newh,neww,3),dtype=np.uint8)
    else: # gray
        newimg = np.zeros((newh,neww),np.uint8)
    newimg[:h,:w] = img.copy()
    return newimg

# Grabcut をマージン5%だけを背景指定して得られる画像を返す. 
# offset がマイナスの場合は rect で指定した矩形を使う  指定は幅高さに対する割合 (0,0,1,1)なら枠なし
def grabcut(img, offset=5, rect=None):
    h,w = img.shape[0],img.shape[1]
    if offset >= 0:
        offsetY,offsetX = int(h*offset/100),int(w*offset/100)
        rect =(offsetX,offsetY,img.shape[1]-2*offsetX,img.shape[0]-2*offsetY)
    else:
        (x1,y1,w1,h1) = rect 
        rect  = (int(x1*w),int( y1*h), int(w1*w), int(h1*h) )
    bgdmodel = np.zeros((1,65),np.float64)
    fgdmodel = np.zeros((1,65),np.float64)
    mask = np.zeros(img.shape[:2],np.uint8)
    cv2.grabCut(img,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    return img

# grabcut されていることが前提で、背景は完全黒である画像の完全黒部分を余白としてカットする
def margincut(img,pad=5, needRect=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    _img,contours, _hcy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    x1,x2,y1,y2 = [],[],[],[]
    for i in range( len(contours)):
        # ret = (x, y, w, h)
        ret = cv2.boundingRect(contours[i])
        x1.append(ret[0])
        y1.append(ret[1])
        x2.append(ret[0] + ret[2])
        y2.append(ret[1] + ret[3])
        
    x_min = min(x1)
    y_min = min(y1)
    x_max = max(x2)
    y_max = max(y2)
    
    # pad 分だけ余白をつけてコピー
    objecth = y_max-y_min
    objectw = x_max-x_min
    padY,padX = int(objecth*pad/100),int(objectw*pad/100)    
    newimg = np.zeros((objecth+2*padY, objectw+2*padX,3),np.uint8)
    newimg[padY:padY+objecth,padX:padX+objectw] = img[y_min:y_max,x_min:x_max].copy()
    if needRect == True:
        return newimg,  (x_min,y_min,x_max,y_max)
    else:
        return newimg
    
ShrinkTarget = 128
# 多重解像度画像の生成
def mulresimg(img,minsize=ShrinkTarget,pad=-1,offset=5, rect=None):
    if pad >= 0:  # マージンカットする場合
        w = img.shape[1]
        h = img.shape[0]
        size = np.array([w,h]).min()
        # grabcut は面積に対して線型以上の計算量かかる。512x512で数秒かかるのでそれ以上のサイズで実行するのは
        # 現実的ではない。マージンカットが目的であれば２５６程度の結果を使って差し支えない。
        if size > 256:
            w1 = w*256//size
            h1 = h*256//size
            refimg = cv2.resize(img,(w1,h1))
            timg = grabcut(refimg,offset=offset,rect=rect)
            timg,(x1,y1,x2,y2)= margincut(timg,pad=pad,needRect=True)            
            ox1,oy1,ox2,oy2 = int(x1*w/w1),int(y1*h/h1), int(x2*w/w1),int(y2*h/h1)
            objecth, objectw = oy2-oy1, ox2-ox1
            padY,padX = int(objecth*pad/100),int(objectw*pad/100)
            newimg = np.zeros((((oy2-oy1)+2*padY),((ox2-ox1)+2*padX),3),np.uint8)
            newimg[padY:padY+objecth,padX:padX+objectw]=img[oy1:oy2,ox1:ox2]
            img = newimg
        else:
            img = grabcut(img, offset=offset,rect=rect)
            img = margincut(img, pad=pad)   
    img = getbase2img(img)  # まず 2 のべき乗サイズにする
    mulimg = [img]
    while np.max(img.shape[:2]) > minsize: # 縦横どちらかもが minsize 以下になるまで縮小
        img = cv2.pyrDown(img)
        mulimg.append(img)
    return  mulimg

# 　多重解像度画像を一枚にまとめて表示
def showMimg(mimg, needShow=True):
    w = (mimg[0].shape[1])*3//2
    h = mimg[0].shape[0]
    if len(mimg[0].shape)==3:
        all = np.zeros((h,w,mimg[0].shape[2]),np.uint8)
        c = 3
    else: # gray
        all = np.zeros((h,w),np.uint8)
        c = 1
    y,x = 0,0
    for  index,img in enumerate(mimg):
        w = img.shape[1]
        h = img.shape[0]
        all[y:y+h,x:x+w]=img
        if index%2 == 0:
            x += w
        else:
            y += h
            
    if needShow:
        if c == 1:
            plt.imshow(all,cmap='gray')
        else:
            plt.imshow(all[:,:,::-1])
        plt.show()
        
    return(all)

#  大きな画像をキー操作で拡大縮小して観察するためのメソッド
def showBigImage(img):
    cv2.namedWindow("BigImage",cv2.WINDOW_AUTOSIZE)
    ratio = 1.0
    posx,posy = 0,0
    h0 = img.shape[0]
    w0 = img.shape[1]
    WSIZE = 512
    padding = np.zeros((WSIZE,WSIZE,3),np.uint8)
    window = padding.copy()
    endy = posy+WSIZE if posy+WSIZE < h0 else h0
    endx = posx+WSIZE if posx+WSIZE < w0 else w0

    window[:h0,:w0]=img[posy:endy,posx:endx]

    cv2.imshow("AA",window)

    tmpimg = img.copy()

    while(1):

        h = tmpimg.shape[0]
        w = tmpimg.shape[1]
        endy = posy+WSIZE if posy+WSIZE < h else h
        endx = posx+WSIZE if posx+WSIZE < w else w
        window = padding.copy()
        window[:endy-posy,:endx-posx]=tmpimg[posy:endy,posx:endx]
        cv2.namedWindow("AA",cv2.WINDOW_AUTOSIZE)
        cv2.imshow("AA",window)

        k = cv2.waitKey(1) # 1milisecond 入力を受け付ける　　0にすると無限に待ってしまう

        #if k > 0 :
        #    print(k)

        if k == 27:         # 終了は ESC
            break

        elif k == ord('w') or k == 63232: # ↑
            if posy > 0 :
                posy -= 10
            if posy < 0:
                posy = 0
        elif k == ord('z') or k == 63233: # ↓
            if posy < h :
                posy += 10
            if posy > h:
                posy = h
        elif k == ord('a') or k == 63234: # ←
            if posx > 0 :
                posx -= 10
            if posx < 0:
                posx = 0
        elif k == ord('d') or k == 63235: # →
            if posx < w :
                posx += 10
            if posx > w:
                posy = w

        elif k == ord('+'):
            ratio *= 1.414
            newh = int(h0*ratio)
            neww = int(w0*ratio)
            posy = posy+70 if posy+70 < newh-512 else newh-512
            posx = posx+70 if posx+70 < neww-512 else neww-512
            tmpimg = cv2.resize(img,(neww,newh))
        elif k == ord('-'):
            ratio /= 1.414
            newh = int(h0*ratio)
            neww = int(w0*ratio)
            posy = posy-70 if posy > 70 else 0
            posx = posx-70 if posx > 70 else 0
            tmpimg = cv2.resize(img,(neww,newh))
    cv2.destroyAllWindows()
    cv2.waitKey(1)


## インタラクティブな対象抽出
import tkinter
import tkinter.filedialog
import tkinter.messagebox

# %gui tk
root=tkinter.Tk()
root.withdraw()

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
MAGENTA = [255,0,255]    # sure BG
BLACK = [0,0,0]
WHITE = [255,255,255]   # sure FG
MINRECTSIZE = 400 # 領域指定とそうでない操作の切り分けのための矩形面積の下限

IMAGESIZE = 512  # 強制的に画像サイズをの数字以下に縮小する。

DRAW_BG = {'color' : MAGENTA, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

# filename = "horyou2c6k.jpg"
# filename = "yumiko.png"

def grabcut_main():
    global img,img2,orig,output,value,mask,shrinkN,filename, mouseCallBacker

    print("画像ファイルを選んで下さい")
    filename = readFilePath()

    #orig = getbase2img(cv2.imread(filename))  # 2のべき乗のサイズになるようパディング
    orig = cv2.imread(filename)
    img = orig.copy()
    # img, shrinkN = shrink(img,size=512)   # 表示のため縦横いずれも512以下になるまで縮小
    img = grabcut_resize(img, size=512)   # 表示のため縦横いずれも512以下になるまで縮小
    img2 = img.copy()                        # a copy of original image

    mask = np.zeros(img.shape[:2],np.uint8) # mask initialized to PR_BG
    output = np.zeros(img.shape,np.uint8)           # output image to be shown

    # input and output windows
    cv2.namedWindow('output')
    cv2.namedWindow('input')
    cv2.imshow('output',output)
    cv2.imshow('input', img)

    mouseCallBacker = grabcut_myMouse('input')

    cv2.moveWindow('input',img.shape[1],90)

    print(" マウスの左ドラッグで抽出対象を囲って下さい \n")

    do_grabcut_keyEventLoop()

    cv2.destroyAllWindows()
    cv2.waitKey(1)

def do_grabcut_keyEventLoop():
    global img,img2,output,value,mask, rect,frame_or_mask, mouseCallBacker
    # キーイベントループ
    while(1):

        k = cv2.waitKey(1) # 1milisecond 入力を受け付ける　　0にすると無限に待ってしまう

        if k == 27:         # 終了は ESC
            break

        elif k == ord('0'): # 背景領域の指定
            print(" マウス左ボタンで背景領域を指定 \n")
            value = DRAW_BG
        elif k == ord('1'): # 対象の指定
            print(" マウス左ボタンで切り出し対象領域を指定 \n")
            value = DRAW_FG
        elif k == ord('2'): # 背景かも知れない領域の指定
            print(" マウス左ボタンで背景領域を指定 \n")
            value = DRAW_PR_BG
        elif k == ord('3'): # 前景かもしれない領域の指定
            print(" マウス左ボタンで切り出し対象領域を指定 \n")
            value = DRAW_PR_FG

        elif k == ord('+'):
            mouseCallBacker.thicknessUp()

        elif k == ord('-'):
            mouseCallBacker.thicknessDown()


        elif k == ord('9'): # 90度回転
            print(" 回転します\n")
            img = img.transpose(1,0,2)[::-1,:,:]
            img2=img2.transpose(1,0,2)[::-1,:,:]
            mask=mask.transpose(1,0)[::-1,:]
            output=output.transpose(1,0,2)[::-1,:,:]
            width , height = img.shape[1], img.shape[0]
            rect= (rect[1],height-rect[0]-rect[2],rect[3],rect[2])
            cv2.imshow('output',grabcut_redmasked(output))
            cv2.imshow('input',img)

        elif k == ord('s'): # 画像の保存
            bar = np.zeros((img.shape[0],5,3),np.uint8)
            res = np.hstack((img2,bar,img,bar,output))

            print("抽出結果を保存するパスを選んで下さい（拡張子は不要）")
            savepath = saveFilePath()
            cv2.imwrite(savepath+".png",output)
            cv2.imwrite('grabcut_output.png',res)
            print("抽出結果は保存先:{}に、\n, それとは別に合成画像を grabcut_output.png に結果を保存しました.\n".format(savepath+".png"))

        elif k == ord('r'): # reset everything
            print("リセット \n")
            mouseCallBacker.init()
            img = img2 .copy()  # 画像を復元
            mask = np.zeros(img.shape[:2],np.uint8) # mask initialized to PR_BG
            output = np.zeros(img.shape,np.uint8)           # 出力画像用
            mouseCallBacker.thickness = 3

        elif k == 13 : #  Enter キー  セグメンテーションの実行
            print("セグメンテーションの実行中。新しいメッセージが表示されるまでお待ち下さい。 \n")
            print(mask.shape,rect)
            if (frame_or_mask == 0):         # grabcut with rect
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                mask = mask.copy()
                cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
                frame_or_mask = 1
            elif frame_or_mask == 1:         # grabcut with mask
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
                print(" 抽出がうまくいっていない場合は、手動でタッチアップしてから再度 N  を押して下さい。\n ０、２　背景領域の指定、１，３ 抽出対象領域の指定 \n")

        mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
        output = cv2.bitwise_and(img2,img2,mask=mask2)

class grabcut_myMouse:
    def __init__(self, windowname):
        self.init()
        cv2.setMouseCallback(windowname, self.callBack, None)

    def init(self):
        global value,mask,framing,framed,drawing,rect,frame_or_mask
        # setting up flags
        rect = (0,0,1,1)
        drawing = False         # 描画モードオン
        framing = False           # 選択枠設定中
        framed = False       # 枠設定は完了している
        frame_or_mask = 100      # flag for selecting rect or mask mode
        value = DRAW_FG         # drawing initialized to FG
        self.thickness = 3           # ブラシサイズ

    def thicknessUp(self):
        self.thickness +=1

    def thicknessDown(self):
        if self.thickness > 0:
            self.thickness -=1

    def callBack(self, event, x, y, flags, param=None) :
        global img,img2,output,value,mask,framing,framed,drawing,rect,frame_or_mask
        # フレーム設定フェーズの処理
        if framed == False:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.lx,self.ly = x,y
                framing = True # 矩形描画モードオン
            elif event == cv2.EVENT_MOUSEMOVE:
                if framing == True :
                    img = img2.copy()
                    cv2.rectangle(img,(self.lx,self.ly),(x,y),BLUE,2)
                    rect = (min(self.lx,x),min(self.ly,y),abs(self.lx-x),abs(self.ly-y))
            elif event == cv2.EVENT_LBUTTONUP:
                framing = False
                tmps = abs(self.lx-x)*abs(self.ly-y)  # 指定矩形の面積
                if tmps > MINRECTSIZE:
                    framed = True
                    cv2.rectangle(img,(self.lx,self.ly),(x,y),BLUE,2)
                    rect = (min(self.lx,x),min(self.ly,y),abs(self.lx-x),abs(self.ly-y))
                    frame_or_mask = 0
                    print(" Enterキーを押せば抽出を始めます。終わるまでしばらくお待ち下さい \n")
        else: # 枠指定がすでに済んでいる場合
            if drawing == True:
                cv2.circle(img,(x,y),self.thickness,value['color'],-1)
                cv2.circle(mask,(x,y),self.thickness,value['val'],-1)
                if event == cv2.EVENT_LBUTTONUP:
                    drawing = False
            else:
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True

        cv2.imshow('output',grabcut_redmasked(output))
        cv2.imshow('input',img)

# 画像サイズの縦か横の大きい方が指定サイズになるようにリサイズする。
def grabcut_resize(img, size=512):
    maxsize = np.max(np.array(img.shape[:2]))
    height = size*img.shape[0]//maxsize
    width = size*img.shape[1]//maxsize
    output = np.zeros((height+40, width+40,3),np.uint8)
    output[20:20+height,20:20+width]=cv2.resize(img,(width,height))
    return output

def grabcut_redmasked(img):
    ones = np.ones(img.shape[:2],np.uint8)
    zeros = np.zeros(img.shape[:2],np.uint8)
    ret,img2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV)
    b,g,r = cv2.split(img2)
    red = cv2.bitwise_and(ones,b)
    red = cv2.bitwise_and(red,g)
    red = cv2.bitwise_and(red,r)
    red = cv2.merge((zeros,zeros,red*128))
    red = cv2.bitwise_or(red,img)
    return red

# ２値画像の連結成分のうち最大のものを返すメソッド　　オプションで結果画像も返す
def getMainArea(bwimg, inverse=False, needRect=False):
    if inverse:
        bwimg = 255-bwimg
    labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(bwimg)
    
    areas = contours[1:,4]  # 面積のリスト  0番は背景（黒）なので除く
    maxindex = np.argmax(areas)+1
    rect = tuple(contours[maxindex][:4])

    retimg = np.ones(bwimg.shape,np.uint8)
    
    if needRect:
        return labelimg[max], rect
    else:
        return retimg*(labelimg==maxindex)*255
    
# 与えら得た画像を２値化したとき最大のバウンダリ矩形を返すメソッド　　オプションで結果画像も返す
def getRect(img, inverse=False, needImage=False):
    gray  = img
    if len(gray.shape) >2:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if inverse:
        gray = 255-gray
        
    # 大きめのガウシアンフィルタでぼかした後に大津の方法で２階調化
    blur = cv2.GaussianBlur(gray,(25,25),0)  
    _,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 2値画像のラベリングを実行
    labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(otsu)
    areas = contours[:,4]  # 面積のリスト  0番は背景（黒）なので除く
    
    def calcCircleLevel(labelimg,label):
        retimg = np.ones(labelimg.shape,np.uint8)
        retimg = retimg*(labelimg==label)*255
        _,cont,_ = cv2.findContours(retimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perim = cv2.arcLength(cont[0],True)
        # areas1 = cv2.contourArea(cont[0])
        if perim > 0:
            circle_level = (4.0 * np.pi * areas[label] / (perim * perim)) # 円形度　
        else:
            circle_level = np.nan
        return circle_level
    
    # エリアごとのダイコンっぽさを計算
    likeness = []  #  ダイコンっぽさ＝面積と円形度の積で定義する
    for lab in range(0,labelnum):
        cl = calcCircleLevel(labelimg,lab)
        ln =  cl*areas[lab]
        likeness.append(ln)
        print("label {}   areas {}  circle level {}    likeness {}".format(lab,areas[lab],cl, ln ))
    
    whatiwant = np.argmax(np.array(likeness[1:]))+1
    rect = tuple(contours[whatiwant][:4])
    
    if needImage:
        retimg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for label in range(1,labelnum):
            x,y = GoCs[label]
            thickness = int(gray.shape[0]/100+1)
            retimg = cv2.circle(retimg, (int(x),int(y)), thickness, (0,0,255), -1)    
            x,y,w,h,size = contours[label]
            retimg = cv2.rectangle(retimg, (x,y), (x+w,y+h), (255,255,0), thickness)         
        return  rect, retimg
    else:
        return rect
    
    rect = tuple(contours[whatiwant][:4])
    
    if needImage:
        retimg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for label in range(1,labelnum):
            x,y = GoCs[label]
            thickness = int(gray.shape[0]/100+1)
            retimg = cv2.circle(retimg, (int(x),int(y)), thickness, (0,0,255), -1)    
            x,y,w,h,size = contours[label]
            retimg = cv2.rectangle(retimg, (x,y), (x+w,y+h), (255,255,0), thickness)         
        return  rect, retimg
    else:
        return rect
    
# ファイルオープン
import tkinter
import tkinter.filedialog
import tkinter.messagebox
# %gui tk
root=tkinter.Tk()
root.withdraw()


def readFilePath():
    fTyp=[('画像ファイルの選択',['jpg','png'])]
    filename=tkinter.filedialog.askopenfilename(filetypes=fTyp,initialdir = './pics')
    print("処理対象ファイル",filename)
    return filename

def saveFilePath():
    filename = tkinter.filedialog.asksaveasfilename()
    return filename

##############
MINIMAXSIZE = 256
def getRadish(img,  minimaxsize = MINIMAXSIZE, needDetail=False, inverse=False):
    
    height, width =  img.shape[0],img.shape[1]
    
    if inverse: #  背景が白の場合 grabcut で対象を切り出すことで結果的に背景を黒に変える
        img =  py.grabcut(img, offset=150/np.min(img.shape[:2]))
        
    (x,y,w,h) = py.getRect(img,needImage=False) # 画像中の白系領域で最も大きな領域を囲む矩形を求める
    
    dh,dw = 0.05*h,0.05*w
    rrect = ((x-dw)/width, (y-dh)/height, (w+2*dw)/width, (h+2*dh)/height) # 若干大きめにする

    mm = py.mulresimg(img,pad=3,offset=-1,rect=rrect) # 多重解像度画像の生成
    
    for index,img in enumerate(mm):
          if np.min(img.shape[:2]) <= minimaxsize:
            break
    
    # これから処理対象とする画像
    src = mm[index] 
    src = py.margincut(src, pad=10) # ピラミッド化で余分な空白がついているのでカット 10%の余白は残す

    # ２値化のためにグレー画像をつくる
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

    # 大きめのガウシアンフィルタでぼかした後に大津の方法で２階調化
    blur = cv2.GaussianBlur(gray,(35,35),0)  
    _,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                           
    # 指標等がいっしょに写り込んでいる可能性があるので、最大の白領域だけ抜き出す
    otsu = py.getMainArea(otsu, inverse=False, needRect=False)
        
    iter = int(0.02*np.min(src.shape[:2])) 
    
    # 膨張処理で確実に背景である領域をマスク
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)) # 円形カーネル
    #kernel =cv2.getStructuringElement(cv2.MORPH_CROSS,(iter,iter)) # 十字カーネル
    mask1 = 255-cv2.dilate(otsu,kernel,iterations = iter)
    
    # 収縮処理で確実に内部である領域をマスク
    mask2 = cv2.erode(otsu,kernel,iterations = iter)

    # grabcut　用のマスクを用意 
    grabmask = np.ones(src.shape[:2],np.uint8)*2
    # 
    grabmask [mask1==255]=0     # 黒　　背景　　
    grabmask [mask2==255]=1    # 白　前景

    # plt.imshow(grabmask*127)
    # plt.show()
    
    # grabcut の作業用エリアの確保
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # MASK による grabcut 実行
    grabmask, bgdModel, fgdModel = cv2.grabCut(src,grabmask,None,bgdModel,fgdModel,20,cv2.GC_INIT_WITH_MASK)
    grabmask = np.where((grabmask==2)|(grabmask==0),0,1).astype('uint8')
    grabimg = src*grabmask[:,:,np.newaxis]

    if needDetail:
        return grabimg, mm, index, mask1,mask2, grabmask
    else:
        return grabimg