#%% [markdown]
# # 中心軸の上下端点を指定するツール
# 
# ## **Usage**
# 
# ```main(file = "自動計測データ.xlsx", target=10)```
# 
# 指定したエクセルファイルの「処理対象欄」に指定した数値が入っている個体だけが処理対象です。  
# 指定番号は実行時に 引数 target で指定できます。（デフォルトは10）
# 
# ## 使い方
# 
# 中心軸の上端と末端の２点をクリックして、ENTERで確定してください。 
# 
# 座標と削除半径がエクセルファイルに記録されます。
# 
# - 「ESC」で終了します。(または「Q」または「０」）
# - 「Enter」で確定次に進みます。
# - 「R」でやり直しできます。ただし、上・下両方ともやり直しになります。
# -  カーソルキー やアルファベットキーに削除半径の拡大縮小が割り当ててあります。
# 
# 
# ## キーバインド
# |機能|キー|
# |:---:|:---:|
# |+1|→、d|
# |+3|↑、w|
# |-1|←、a|
# |=3|↓、s|
# |やり直し|r|
# |確定|Enter（１回目は確認、２回目で本当に確定）|
# 
# 私の MacBook では、ESC, Q, カーソルキー のキーコードが取得できませんでした。反応しない場合はプログラムを書き換えて別のキーを割り当ててください。
# 
# ## おすすめ
# 
# 上端の凸凹や末端が曲がりが激しい場所は形状近似に悪影響が出ますので、削除した方がよいでしょう。
# 
# 
# # Excelファイルの仕様
# 
# 省略、わかると思います。
# 
# 

#%%
import numpy as np
import cv2
import pandas as pd
import os
from rdlib2 import getstandardShape,getCoG
import matplotlib.pyplot as plt
# %matplotlib inline
from skimage.morphology import skeletonize
from skimage import morphology, color

class MarkApp():
    UNIT = 256 # 正規化サイズ
    def __init__(self,file="計測指示＆記録.xlsx",refimgdir="一軍",target=10):
        # global 変数
        self.datafile = file # 指示＆記録用 excelファイルのファイル名
        self.refimgdir = refimgdir
        self.target = target
        # self.src # 原画像
        # self.rdimg  # 正規化画像
        # self.rdcolor  # 対象画像のカラー版（表示参照用）
        # self.rdcnt  # 輪郭線情報
        # self.org # 参照用原画像
        # self.blur # スケルトン抽出用ボケ画像
        # self.df  # 作業用のデータフィールド
        self.x0, self.y0 = 0,0 # バウンダリ矩形の基準点
        self.height,self.width = 0,0 # バウンダリ矩形のシェイプ
        self.c_x, self.c_y = 0,0 # 重心位置
        self.topdx = 0 # 上削除円の中心と削除半径
        self.topdy = 0
        self.topdr = 10
        self.btmdx = 0 # 下削除円の中心と削除半径
        self.btmdy = 0
        self.btmdr = 10
        self.angle = 0 # 指定回転角
        self.anglemode = False # 角度指定モードか否かのフラグ。端点指定モードと角度指定モードがある。
        self.tflag, self.bflag = False,False # 上下の指定点が確定したかどうかのフラグ
        self.dic = {} # スケルトン画素からその点で描くべき削除円半径を対応づける検索辞書
        self.radius = 10 # 削除円半径のベース距離 これに drd を加えた距離が実際に登録される半径となる
        self.drd = 0 # 削除円の半径の、デフォルトからの増分 中心がスケルトンにあるときは、基準距離を dic 情報から得られる距離に読み換える
        self.uppercnt, self.lowercnt = None,None # 上部輪郭点（上から高さの 1/4 までに入る輪郭点）と下部輪郭点（下から高さの半分まで、スケルトンも含む）
        self.sx,self.sy = 0,0
        # self.imgfilename # ファイル名をウィンドウ名として使うので、グローバル化

    # 初期化
    def initcondition(self,allreset=True):
        if allreset: # 初回と　R リセットの場合はこちら
            self.angle = 0
            rotbool = True
        else: # 回転させて確定した場合は強制回転した形状を再構成
            rotbool = False
        self.rdimg = getstandardShape(self.src, unitSize=self.UNIT, thres=0.25, setrotation = self.angle, norotation = rotbool) 
        self.rdgray = self.rdimg.copy()
        self.collectImgInfo()
        self.rdcolor = cv2.cvtColor(self.rdgray, cv2.COLOR_GRAY2BGR)
        self.tflag,self.bflag = False,False # 上下の点が決まっているかどうかのフラグ
        self.radius = 10 # 円の半径
        self.drd = 0 # キー指定での半径加算分
        self.anglemode = False 
        cv2.namedWindow(self.imgfilename)
        cv2.namedWindow("reffer")
        cv2.imshow("reffer",self.org)
        cv2.moveWindow("reffer", 0, 500)

    # 輪郭点、芯線、重心などの情報を作成する
    def collectImgInfo(self):
        # 重心位置　c_x,c_y とバウンダリボックス
        # 回転させてある可能性があるので、rdimg を使ってはいけない
        # 輪郭線情報
        _img,contours,_hierarchy = cv2.findContours(self.rdgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        rdcnt = contours[np.argmax([len(c) for c in contours])] # 輪郭線情報　global 変数
        c_x,c_y,self.x0,self.y0,self.width,self.height = getCoG(self.rdgray) # 重心の位置、バウンダリボックスの左上(x0,y0), 幅と高さ
        self.c_x,self.c_y = int(round(c_x)),int(round(c_y))

        # 上端部輪郭と末端部輪郭 
        self.uppercnt = [[x,y] for [[x,y]] in rdcnt if y <= self.y0 + int(self.height/4)]
        self.lowercnt = [[x,y] for [[x,y]] in rdcnt if y >= self.y0+int(0.5*self.height)]

        # スケルトンも候補に入れる
        # スケルトン抽出用にボケ画像を作る　回転させると髭だらけになるのでぼかし必須
        self.blur = cv2.GaussianBlur(self.rdgray,(7,7),0)
        _r,self.blur = cv2.threshold(self.blur,127,255,cv2.THRESH_BINARY)

        rdimg1 = self.blur/255   # scikit-learn の細線化は１ビット画像でないといけない
        skimg = morphology.medial_axis(rdimg1)
        self.rdgray[skimg] = 128

        # スケルトン下部分の座標配列
        ys,xs = np.where(skimg)
        skpoints = [[x,y] for (x,y) in zip(xs,ys) if y >= self.y0+int(0.5*self.height) ]
        # 各点について、最も近い輪郭点までの距離を求めておく
        radlist = []
        for p in skpoints:
            diff1 = [np.sqrt((p[0]-x)**2+(p[1]-y)**2) for [x,y] in self.lowercnt if x < p[0]] # その点より左の輪郭との距離
            diff2 = [np.sqrt((p[0]-x)**2+(p[1]-y)**2) for [x,y] in self.lowercnt if x >= p[0]] # その点より右の輪郭との距離
            if len(diff1) > 0 :
                if len(diff2) > 0:
                    lenfornearest = max(int(min(diff1)),int(min(diff2)))+3 # 左右それぞの最短のうちの大きい方＋３
                else: # 左はあるが右はない　超レアケース
                    lenfornearest = int(min(diff1))+3
            else: # 右はあるが、左はないケース。両方ないというのはありえない。
                    lenfornearest = int(min(diff2))+3  
            radlist.append(lenfornearest)

        self.dic = {}
        for [x,y],d in zip(skpoints,radlist):
            if d < 20:
                self.dic[(x,y)]=d
        skpoints = [key for key,d in zip(self.dic.keys(),self.dic.values()) if d < 20]     
        # 下エリア吸着点にスケルトンを加える
        self.lowercnt = self.lowercnt+skpoints
    
    # 処理対象画像を重心周りに回転 処理対象は rdimg、回転角は初期状態に対する角度
    def imgrotation(self,angle):
        # 回転行列を作る
        rotation_matrix = cv2.getRotationMatrix2D((self.c_x,self.c_y), angle, 1)
        size=self.rdimg.shape
        self.rdgray = cv2.warpAffine(self.rdimg, rotation_matrix, size, flags=cv2.INTER_CUBIC)
        self.rdcolor = cv2.cvtColor(self.rdgray,cv2.COLOR_GRAY2BGR)
        return self.rdcolor

    # メインプログラム
    def main(self):
        self.df = pd.read_excel(self.datafile)
        self.df.reset_index(inplace=True, drop=True) # エクセルファイルの編集でインデックスが欠落したり入れ替わっている場合があるとおかしくなるので振りなおしておく
        for radish in range(len(self.df)):
            idata = self.df.iloc[radish]
            topdir = idata['topdir']  #  画像ファイルのパスのベース
            subdir = idata['subdir']  #  サブディレクトリ
            self.imgfilename = idata['filename'] #  ファイル名
            check = idata['処理対象'] #  処理対象かどうかのフラグ　　test がTrueの時のみ意味がある
            if check != self.target : #  check が 10 でない画像はスルーする
                    continue
            path = os.path.join(topdir,subdir,self.imgfilename)
            print("処理対象画像 {}\n".format(path))
            # シルエット画像の読み込み
            self.src= cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # 参照用に元画像を読み込み
            org= cv2.imread(os.path.join(self.refimgdir,subdir,self.imgfilename),1) 
            self.org = cv2.resize(org,(int(org.shape[1]/2),int(org.shape[0]/2)))
            # あとで用いる計測ツールで読み込んだ時の画像状態を作る

            # 初期設定                    
            self.initcondition()

            while True:
                    wkimg = self.rdcolor.copy()
                    cv2.setMouseCallback(self.imgfilename, self.onMouse)
                    cv2.imshow(self.imgfilename, wkimg)

                    key = cv2.waitKey(0)
                    if key==ord('r') or key == ord('R'):
                        self.initcondition()
                    elif key==ord('m'): # mode change
                        self.anglemode = not self.anglemode 
                        if self.anglemode == False:
                            self.initcondition(allreset=False)
                    elif key==2 or key==ord('1') or key==ord('a'): # LEFT 
                        if self.anglemode:
                            self.angle = self.angle + 1
                            self.rdcolor = self.imgrotation(self.angle)
                        else:
                            self.radius = self.radius - 1  if self.radius > 2 else 1                    
                    elif key==3 or key==ord('2') or key==ord('d'): # RIGHT
                        if self.anglemode:
                            self.angle = self.angle - 1
                            self.rdcolor = self.imgrotation(self.angle)
                        else:
                            self.radius = self.radius + 1
                    elif key==0 or key==ord('3') or key==ord('w'): # UP
                        if self.anglemode:
                            self.angle = self.angle + 3
                            self.rdcolor = self.imgrotation(self.angle)
                        else:
                            self.radius = self.radius + 3
                    elif key==1 or key==ord('4') or key==ord('s'): # DOWN
                        if self.anglemode:
                            self.angle = self.angle - 3
                            self.rdcolor = self.imgrotation(self.angle)
                        else:
                            self.radius = self.radius - 3 if self.radius > 4 else 2
                    elif key==13: # Enter で確定　次へ,
                        if not(self.tflag and self.bflag):
                            continue
                        font = cv2.FONT_HERSHEY_PLAIN
                        print("確定しますか? ENTER->確定  R -> やり直し")
                        cv2.putText(self.rdcolor,"OK?",(10,60),font,1,(0,255,0))
                        cv2.putText(self.rdcolor,"Ent->Record",(10,75),font,1,(0,255,0))
                        cv2.putText(self.rdcolor,"R->Reset",(10,90),font,1,(0,255,0))
                        key2 = cv2.waitKey(0)
                        if key2 == 13:
                            self.df.loc[radish,'TOPX'] = self.topdx
                            self.df.loc[radish,'TOPY'] = self.topdy
                            self.df.loc[radish,'TOPDR'] = self.topdr
                            self.df.loc[radish,'BTMX'] = self.btmdx
                            self.df.loc[radish,'BTMY'] = self.btmdy
                            self.df.loc[radish,'BTMDR'] = self.btmdr
                            self.df.loc[radish,'ROT'] = self.angle
                            self.df.loc[radish,'処理対象'] = 1
                            self.df.to_excel(self.datafile, index=True, header=True)
                            print("確定しました\n\n")
                            print("{} {} {}, {} {} {} {:0.2f}".format(self.topdx,self.topdy,self.topdr,self.btmdx,self.btmdy,self.btmdr,self.angle))
                            break
                        elif key2 == ord('r') or key2 == ord('R'):
                            self.initcondition()
                    elif key==27 or key==ord('q') or key == ord('0'): # ESC で終了: # ESC で終了
                        break
            if key==27 or key==ord('q') or key == ord('0'): # ESC で終了
                cv2.destroyAllWindows()
                break
        self.df.to_excel(self.datafile, index=True, header=True) # だめ押しでもう１度書き込んでおく
        cv2.waitKey(1)
    
    # (x,y)に最も近い輪郭上の点を答える
    def nearestPos(self,x,y):
        cnt = self.uppercnt if y <= self.c_y else self.lowercnt
        diff = [[x-self.cx,y-self.cy] for [self.cx,self.cy] in cnt] # 輪郭点と(x,y)を結ぶベクトル
        distance = [np.sqrt(dx*dx+dy*dy) for dx,dy in diff] # ベクトルの長さ = 距離
        mi = np.argmin(distance)
        return cnt[mi],distance[mi]

    def drawcircle(self,wkimg):
        [self.cx,self.cy],_d = self.nearestPos(self.sx,self.sy)
        cx,cy = self.cx,self.cy
        if (cx,cy) in self.dic:
            if self.radius < self.dic[(cx,cy)]:
                self.drd = self.dic[(cx,cy)]-self.radius
        else:
            self.drd = 0
        cv2.circle(wkimg,(cx,cy),5,(0,0,255),-3)
        cv2.circle(wkimg,(cx,cy),self.radius+self.drd,(255,0,255),2)
        cv2.circle(wkimg,(self.c_x,self.c_y), 3, (255,128,0), 3)
        
    def drawcursor(self,wkimg):
        [h,w] = wkimg.shape[:2] # バウンダリの h,w でないことに注意
        sx,sy = self.sx,self.sy
        c_x,c_y = self.c_x,self.c_y
        cv2.line(wkimg,(self.sx,0),(self.sx,h),(255,0,0),1)
        cv2.line(wkimg,(0,self.sy),(w,self.sy),(255,0,0),1)
        if sx!=c_x: # self.anglemode and 
            cv2.line(wkimg,(sx-500, int(sy-(500*(sy-c_y)/(sx-c_x)))), (c_x+500, int(c_y+(500*(sy-c_y)/(sx-c_x)))), (255, 0, 0),1)  

    # マウスのコールバック関数　マウスイベントに対する応答
    def onMouse(self, event, x, y, flags,params):    
        self.sx,self.sy = x,y
        wkimg = self.rdcolor.copy()
        # クリックされた時
        if event == cv2.EVENT_LBUTTONUP:
            [cx,cy] ,_distance = self.nearestPos(self.sx,self.sy)
            print("(登録座標({},{}) クリック座標{},{}) - ".format(cx,cy,self.sx,self.sy))
            if self.sy <= self.c_y and self.tflag == False: # 上端確定
                self.topdx,self.topdy,self.topdr = cx,cy,self.radius
                self.tflag = True
            elif self.sy > self.c_y and self.bflag == False:
                self.btmdx,self.btmdy,self.btmdr = cx,cy,self.radius+self.drd
                self.bflag = True
            else:
                return
            cv2.circle(self.rdcolor,(cx,cy),self.radius+self.drd,(0,0,255),-1)
            cv2.circle(self.rdcolor,(cx,cy),3,(255,255,255),-1)
        # マウスが移動ている間は十字カーソルを表示
        if event == cv2.EVENT_MOUSEMOVE:
            wkimg = self.rdcolor.copy()
            self.drawcursor(wkimg)
            self.drawcircle(wkimg)
            cv2.imshow(self.imgfilename, wkimg)

app = MarkApp(file="計測指示＆記録.xlsx",refimgdir="一軍",target=10)
app.main()
cv2.destroyAllWindows()
cv2.waitKey(1)



