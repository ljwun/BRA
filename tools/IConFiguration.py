import argparse
import os
import os.path as osp
import sys
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.backend_bases import MouseButton
import re
import math
import numpy as np
import cv2
from rich.console import Console
from rich.tree import Tree
from rich.traceback import install
install(show_locals=True)
import collections.abc
from typing import Any
from collections import deque
console = Console()

__proot__ = osp.normpath(osp.join(osp.dirname(__file__), ".."))
sys.path.append(__proot__)
from distance.mapping import Mapper

def make_parser():
    parser = argparse.ArgumentParser("pre/re-configure BRA system")
    parser.add_argument("-n", "--name", type=str, default=None, help="configure file name")
    parser.add_argument("-t", "--target", type=str, default=None, help="configure target scene(image file location)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("-s", "--setting", action="store_true", help="set parameter with GUI")
    action.add_argument("-v", "--visualize", action="store_true", help="view the visualize setting")
    return parser

def onclick(marks, mark_info, polygon, polygon_info, texts, ax, base_scale = 1, ex_fn=None):
    def fn(event):
        if event.dblclick and event.button is MouseButton.LEFT:
            ix, iy = event.xdata, event.ydata
            if ix != None or iy != None:
                circle = Circle((ix, iy), mark_info['size'], color=mark_info['color'], alpha=mark_info['alpha'])
                ax.add_patch(circle)
                text = ax.annotate(
                        f'{len(marks)+1}', (ix, iy),
                        color='white', weight='bold', fontsize=10,
                        ha='center', va='center')
                marks.append(circle)
                texts.append(text)
                if len(marks) >= 3:
                    if len(polygon) > 0:
                        polygon.pop().remove()
                    polygon.append(
                        Polygon(
                            [circle.center for circle in marks],
                            color=polygon_info['color'], 
                            alpha=polygon_info['alpha'],
                            zorder=polygon_info['zorder']
                        )
                    )
                    ax.add_patch(polygon[0])
        elif event.button is MouseButton.RIGHT:
            if len(marks) > 0:
                circle = marks.pop(-1)
                circle.remove()
                text = texts.pop(-1)
                text.remove()
            if len(marks) < 3 and len(polygon) > 0:
                polygon.pop().remove()
            elif len(marks) >= 3:
                if len(polygon) > 0:
                    polygon.pop().remove()
                polygon.append(
                    Polygon(
                        [circle.center for circle in marks],
                        color=polygon_info['color'], 
                        alpha=polygon_info['alpha'],
                        zorder=polygon_info['zorder']
                    )
                )
                ax.add_patch(polygon[0])
        plt.draw()
        if ex_fn is not None:
            ex_fn(marks, texts, ax, base_scale)
    return fn

def onscroll(ax, marks, mark_info, base_scale = 2):
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
        mark_info['size'] *= math.sqrt(scale_factor)
        for circle in marks:
            circle.radius = mark_info['size']
            circle.set_alpha(mark_info['alpha'])
        plt.draw()
    return fn

def uiMark(background, title, default=None, fn=None):
    size = (background.size[0] + background.size[1]) * 0.001
    fig, ax = plt.subplots(1)
    marks = []
    polygon = []
    texts=[]
    mark_info = {
        'size': 7*size,
        'color': 'red',
        'alpha': 0.6
        }
    polygon_info = {
        'color': 'green',
        'alpha': 0.4,
        'zorder':0.8,
    }
    # 載入現有資料
    if default is not None:
        for p in default:
            circle = Circle((p['x'], p['y']), mark_info['size'], color=mark_info['color'], alpha=mark_info['alpha'])
            ax.add_patch(circle)
            text = ax.annotate(
                    f'{len(marks)+1}', (p['x'], p['y']),
                    color='white', weight='bold', fontsize=10,
                    ha='center', va='center')
            marks.append(circle)
            texts.append(text)
        if len(marks) > 3:
            polygon.append(
                Polygon(
                    [circle.center for circle in marks],
                    color=polygon_info['color'], 
                    alpha=polygon_info['alpha'],
                    zorder=polygon_info['zorder']
                )
            )
            ax.add_patch(polygon[0])
    ax.imshow(background)
    plt.connect('scroll_event', onscroll(ax, marks, mark_info))
    plt.connect('button_press_event', onclick(marks, mark_info, polygon, polygon_info, texts, ax, size, ex_fn=fn))
    plt.title(label=title, fontweight=40)
    plt.show()
    return marks

def parameter_tree(title, values):
    tree = Tree(title)
    for v in values:
        tree.add(v)
    return tree

def prompt(
    prompt_str : str, 
    default : Any=None, 
    reg : str='.*', 
    choice : [list[Any]]=[], 
    parser : collections.abc.Callable[[Any]:Any] = lambda x:x
)->Any:
    if len(choice) == 0:
        choice_str = ""
    else:
        choice_str = f"({', '.join([f'[bold yellow]{v}[/]' for v in choice])})" 
    while True:
        value = console.input(f'{prompt_str}{choice_str} ')
        if len(value) == 0 and default is None:
            console.print('[bold red]Empty value.Please try again.')
            continue
        elif len(value) == 0:
            value = f'{default}'
        if re.match(reg, value) is None:
            console.print('[bold red]Invalid value.Please try again.')
            continue
        if len(choice) > 0 and parser(value) not in choice:
            console.print('[bold red]Value should in choice.Please try again.')
            continue
        return parser(value)

def perspectiveWarp(points, M):
    p = []
    for x, y in points:
        D = M[2, 0] * x + M[2, 1] * y + M[2, 2]
        p.append((
            int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / D),
            int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / D)
        ))
    return p
                                       
# ref from https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

if __name__ == "__main__":
    # generate from https://patorjk.com/software/taag/
    # font is Crawford2
    title="""
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
    console.print(f'[bold red]{title}[/]')
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
                    'mapping':{
                        'height':None,
                        'width':None,
                        'biasX':None,
                        'biasY':None,
                        'viewW':None,
                        'viewH':None,
                    },
                    'view':[],
                    'position':[],
                },
                'trigger_description':{
                    'threshold':{
                        'target':None,
                        'nonTarget':None,
                        'waiting':None,
                    },
                    'collector':[],
                    'base':[],
                }
            }

            if os.path.exists(confName):
                configStream = open(confName, 'r')
                obj = update(obj, yaml.safe_load(configStream))
                configStream.close()

            process_pipeline = deque()
            background = Image.open(targetName)

            if len(obj['BEV']['position']) == 4:
                console.print(f'[>] Read BEV positions from file:{targetName}.')
            else:
                process_pipeline.append('position')
            if (
                obj['BEV']['mapping']['height'] is not None and 
                obj['BEV']['mapping']['width'] is not None and
                obj['BEV']['mapping']['biasX'] is not None and 
                obj['BEV']['mapping']['biasY'] is not None and
                obj['BEV']['mapping']['viewW'] is not None and
                obj['BEV']['mapping']['viewH'] is not None
            ):
                console.print(f'[>] Read BEV parameters from file:{targetName}.')
            elif (
                obj['BEV']['mapping']['width'] is None or
                obj['BEV']['mapping']['height'] is None
            ):
                process_pipeline.append('mapWH')
            else:
                process_pipeline.append('view')
            process_pipeline.append('result')

            while len(process_pipeline) > 0:
                process = process_pipeline.popleft()
                if process == 'position':
                    console.print(f'[>] Starting setting [bold red]perspective parameter[/] for square...')
                    marks = uiMark(background, "Please label perspective mapping source", default=obj['BEV']['position'])
                    console.print(
                        parameter_tree(
                            '[>] parameters:', 
                            [f'{mark.center}' for mark in marks]
                        )
                    )
                    obj['BEV']['position'] = [{
                        'x':float(mark.center[0]),
                        'y':float(mark.center[1])
                    } for mark in marks]
                elif process == 'mapWH':
                    console.print('[>] Please type width ([bold red]P2-P3[/] & [bold red]P4-P1[/]) of this square mapping for...')
                    default = obj['BEV']['mapping']['width']
                    default_str = "" if default is None else f"([bold green]{default}[/])"
                    obj['BEV']['mapping']['width'] = prompt(f'[mapping width  <{default_str}] ', default=default, reg="^\d*$", parser=lambda x:int(x))
                    console.print('[>] Please type height([bold red]P1-P2[/] & [bold red]P3-P4[/]) of this square mapping for...')
                    default = obj['BEV']['mapping']['height']
                    default_str = "" if default is None else f"([bold green]{default}[/])"
                    obj['BEV']['mapping']['height'] = prompt(f'[mapping height  <{default_str}] ', default=default, reg="^\d*$", parser=lambda x:int(x))
                elif process == 'view':
                    console.print(f'[>] Starting setting [bold red]perspective view area[/]...')
                    mapH, mapW = obj['BEV']['mapping']['height'], obj['BEV']['mapping']['width']
                    points = [(p['x'], p['y']) for p in obj['BEV']['position']]
                    points = np.asarray(points, dtype='float32')
                    view = [(p['x'], p['y']) for p in obj['BEV']['view']]
                    view = np.asarray(view, dtype='float32')
                    marks = uiMark(background, "Please label view area", default=obj['BEV']['view'])
                    if len(marks) != 4:
                        console.print(f'[bold red]Need 4 labels, but just get {len(marks)}.')
                        process_pipeline.appendleft('view')
                        continue
                    obj['BEV']['view'] = [{
                        'x':float(mark.center[0]),
                        'y':float(mark.center[1])
                    } for mark in marks]
                    midPtr = np.float32([
                        [0, 0], [0, mapH],
                        [mapW, mapH], [mapW, 0]]
                    )
                    midM = cv2.getPerspectiveTransform(points, midPtr)
                    corners = perspectiveWarp(
                        np.float32([mark.center for mark in marks]),
                        midM
                    )
                    maxXY, minXY = np.amax(corners, axis=0), np.amin(corners, axis=0)
                    xmin, xmax = minXY[0], maxXY[0]
                    ymin, ymax = minXY[1], maxXY[1]
                    obj['BEV']['mapping']['biasX'] = 0 - xmin
                    obj['BEV']['mapping']['biasY'] = 0 - ymin
                    obj['BEV']['mapping']['viewW'] = xmax - xmin
                    obj['BEV']['mapping']['viewH'] = ymax - ymin
                elif process == 'result':
                    mapH, mapW = obj['BEV']['mapping']['height'], obj['BEV']['mapping']['width']
                    points = [(p['x'], p['y']) for p in obj['BEV']['position']]
                    points = np.asarray(points, dtype='float32')
                    biasX, biasY = obj['BEV']['mapping']['biasX'], obj['BEV']['mapping']['biasY']
                    console.print(obj['BEV']['mapping'])
                    dstPtr = np.float32([
                        [biasX, biasY], [biasX, biasY+mapH],
                        [biasX+mapW, biasY+mapH], [biasX+mapW, biasY]]
                    )
                    M = cv2.getPerspectiveTransform(points, dstPtr)
                    view = (obj['BEV']['mapping']['viewW'], obj['BEV']['mapping']['viewH'])
                    BEV_est = cv2.warpPerspective(np.array(background), M, view)
                    plt.imshow(BEV_est)
                    plt.show()
                        
                    then = prompt(f'[>] Edit again?', default='No', choice=['position', 'mapWH', 'view', 'No'])
                    if then == 'No':
                        break
                    else:
                        process_pipeline.append(then)
                        if then != 'view':
                            process_pipeline.append('view')
                        process_pipeline.append('result')




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
