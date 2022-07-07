import argparse
import os
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backend_bases import MouseButton
import re

def make_parser():
    parser = argparse.ArgumentParser("pre/re-configure BRA system")
    parser.add_argument("-n", "--name", type=str, default=None, help="configure file name")
    parser.add_argument("-t", "--target", type=str, default=None, help="configure target scene(image file location)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("-s", "--setting", action="store_true", help="set parameter with GUI")
    action.add_argument("-v", "--visualize", action="store_true", help="view the visualize setting")
    return parser

def onclick(marks, texts, ax, base_scale = 1):
    def fn(event):
        if event.dblclick and event.button is MouseButton.LEFT:
            ix, iy = event.xdata, event.ydata
            if ix != None or iy != None:
                circle = Circle((ix, iy), 5*base_scale, color='red')
                ax.add_patch(circle)
                text = ax.annotate(
                        f'{len(marks)+1}', (ix, iy),
                        color='white', weight='bold', fontsize=10,
                        ha='center', va='center')
                marks.append(circle)
                texts.append(text)
        elif event.button is MouseButton.RIGHT:
            if len(marks) > 0:
                circle = marks.pop(-1)
                circle.remove()
                text = texts.pop(-1)
                text.remove()
        plt.draw()
    return fn

def onscroll(ax,base_scale = 2):
    def fn(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        plt.draw()
    return fn

def uiMark(background, title, default=None):
    size = (background.size[0] + background.size[1]) * 0.001
    fig, ax = plt.subplots(1)
    marks = []
    texts=[]
    # 載入現有資料
    if default is not None:
        for p in default:
            circle = Circle((p['x'], p['y']), 5*size, color='red')
            ax.add_patch(circle)
            text = ax.annotate(
                    f'{len(marks)+1}', (p['x'], p['y']),
                    color='white', weight='bold', fontsize=10,
                    ha='center', va='center')
            marks.append(circle)
            texts.append(text)
    ax.imshow(background)
    plt.connect('scroll_event', onscroll(ax))
    plt.connect('button_press_event', onclick(marks, texts, ax, size))
    plt.title(label=title, fontweight=40)
    plt.show()
    return marks


if __name__ == "__main__":
    # generate from https://patorjk.com/software/taag/
    # font is Crawford2
    print(
        """
______________________________________________________________________________
       ____   ____    ____         __   ___   ____   _____  ____   ____ 
      |    \ |    \  /    |       /  ] /   \ |    \ |     ||    | /    |
      |  o  )|  D  )|  o  |      /  / |     ||  _  ||   __| |  | |   __|
______|     ||    /_|     |_____/  /__|  O  ||  |  ||  |____|  |_|  |__|______
      |  O  ||    \ |  _  |    /   \_ |     ||  |  ||   _]  |  | |  |_ |
      |     ||  .  \|  |  |    \     ||     ||  |  ||  |    |  | |     |
      |_____||__|\_||__|__|     \____| \___/ |__|__||__|   |____||___,_|
______________________________________________________________________________
        """
    )
    args = make_parser().parse_args()
    confName = None
    targetName = None
    # 若有name參數，則會先檢查是否已存在，若存在則跳出提示
    # 「指定設定文件已經存在，接下來的操作會修改原始文件」
    if args.name is not None:
        confName = args.name
        targetName = args.target
        if len(os.path.splitext(args.name)[1]) == 0:
            confName = f'{args.name}.yaml'
        if os.path.exists(confName):
            print(f'[>] File "{confName}" has exited, following operation will modify it.')

    
    if args.setting is not None:
        if targetName is not None:
            obj = {
                'target':targetName,
                'BEV':{
                    'position':[],
                    'mapping':{},
                },
                'Fences':{},
                'Threshold':{},
            }
            # 讀入現存的檔案
            if os.path.exists(confName):
                configStream = open(confName, 'r')
                obj = yaml.safe_load(configStream)
                configStream.close()
            print(f'[>] Starting setting perspective parameter for square...')
            background = Image.open(targetName)
            marks = uiMark(background, "Please label perspective mapping source", default=obj['BEV']['position'])
            print('[>] parameters:')
            for mark in marks[:-1]:
                print(f'\t├─{mark.center}')
            print(f'\t└─{marks[-1].center}')
            obj['BEV']['position'] = [{
                'x':float(mark.center[0]),
                'y':float(mark.center[1])
            } for mark in marks]
            # 載入舊有的映射點資訊
            # 默認為舊有的資料
            # 還需額外確認輸入的是正常的數字（regex）
            print('[>] Please type width (P2-P3 & P4-P1) of this square mapping for...')
            default_str = f" ({obj['BEV']['mapping']['width']})" if 'width' in obj['BEV']['mapping'] else ""
            map_width = input(f'[mapping width  <{default_str}] ')
            print('[>] Please type height(P1-P2 & P3-P4) of this square mapping for...')
            default_str = f" ({obj['BEV']['mapping']['height']})" if 'height' in obj['BEV']['mapping'] else ""
            map_height = input(f'[mapping height <{default_str}] ')
            obj['BEV']['mapping'] = {
                'width':int(map_width) if len(map_width)!=0 else obj['BEV']['mapping']['width'],
                'height':int(map_height) if len(map_height)!=0 else obj['BEV']['mapping']['height']
            }

            # 由於同意場景的虛擬柵欄數量可能不一樣，需要先確認
            print(f'\n[>] Starting setting fence parameter...')
            print('[>] Please type amount of fence you want...')
            default_str = f" ({len(obj['Fences'])})" if len(obj['Fences'])!=0 else ""
            fence_amount = input(f'[amount <{default_str}] ')
            fence_amount = int(fence_amount) if len(fence_amount)!=0 else len(obj['Fences'])
            fence_idxs = list(obj['Fences'].keys()) if fence_amount==len(obj['Fences']) else None
            for i in range(fence_amount):
                print(f'[Fence {i+1}]')
                default_str = f" ({fence_idxs[i]})" if fence_idxs is not None else ""
                name = input(f'  ├───name{default_str} : ')
                name = name if len(name)!=0 else fence_idxs[i]
                print('  └───location : ')
                predata = obj['Fences'][name] if name in obj['Fences'] else None
                fence_marks = uiMark(background, f'Please label area of Fence {name}', default=predata)
                for mark in fence_marks[:-1]:
                    print(f'        ├─{mark.center}')
                print(f'        └─{fence_marks[-1].center}')
                obj['Fences'][name] = [{
                                    'x':float(mark.center[0]),
                                    'y':float(mark.center[1])
                                } for mark in fence_marks]

            # 系統中還存在一些閥值
            print(f'\n[>] Starting setting threshold...')
            print('[>] Please type amount of threshold you want...')
            default_str = f" ({len(obj['Threshold'])})" if len(obj['Threshold'])!=0 else ""
            threshold_amount = input(f'[amount <{default_str}] ')
            threshold_amount = int(threshold_amount) if len(threshold_amount)!=0 else len(obj['Threshold'])
            threshold_idxs = list(obj['Threshold'].keys()) if threshold_amount==len(obj['Threshold']) else None
            for i in range(threshold_amount):
                print(f'[Threshold {i+1}]')
                default_str = f" ({threshold_idxs[i]})" if threshold_idxs is not None else ""
                name = input(f'  ├───name{default_str} : ')
                name = name if len(name)!=0 else threshold_idxs[i]
                default_value = obj['Threshold'][name] if name in obj['Threshold'] else None
                default_value = ':'.join(map(str,default_value)) if isinstance(default_value, list) else default_value
                default_str = f" ({default_value})" if default_value is not None else ""
                value = input(f'  └───value{default_str} : ')
                if len(value)!=0:
                    if re.match("\d+\.?\d*:\d+\.?\d*", value) is not None:
                        value = value.split(':')
                        value = [float(v) for v in value]
                        value.sort()
                    elif re.match("\d+\.\d*", value) is not None:
                        value = float(value)
                    else:
                        value = int(value)
                else:
                    value = obj['Threshold'][name]
                obj['Threshold'][name] = value
            
            with open(confName, 'w') as outfile:
                yaml.dump(obj, outfile, default_flow_style=False)

    # while True:
    #     # 詢問:「你需要甚麼」
    #     # 選項:
    #     #       -可視化 （若沒有現有的內容，則跳出警語）> （詢問要可視化的項目項目）
    #     #       -完整互動式設置 （若文件已存在，則會再次警告，覆蓋原始設定）
    #     #       -項目設定 （跳出可設定的細項）
    #     #       -保存
    #     #       -離開 （離開前會先檢查記憶體項目與實際項目使否不同，若不同則警告）
    #     print('(v)isualization (S)etting (')

    #     # 目前暫定設定項目:所有、透視、虛擬柵欄、屬性過濾閥值等
    #     # 若找不到目標影像，也會跳警告