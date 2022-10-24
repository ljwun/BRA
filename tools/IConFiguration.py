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
from typing import Any, List, Tuple, Dict
from collections import deque
console = Console()

__proot__ = osp.normpath(osp.join(osp.dirname(__file__), ".."))
sys.path.append(__proot__)
from compute_block import PolygonExtension

def make_parser():
    parser = argparse.ArgumentParser("pre/re-configure BRA system")
    parser.add_argument("-n", "--name", type=str, default=None, help="configure file name")
    parser.add_argument("-t", "--target", type=str, default=None, help="configure target scene(image file location)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("-s", "--setting", action="store_true", help="set parameter with GUI")
    action.add_argument("-v", "--visualize", action="store_true", help="view the visualize setting")
    return parser

def onclick(marks, mark_info, polygon, polygon_info, texts, ax, base_scale = 1, ex_fn=None, ext=None):
    def fn(event):
        if event.dblclick and event.button is MouseButton.LEFT:
            if polygon[0] is not None:
                polygon[0].remove()
                polygon[0] = None
            if polygon[1] is not None:
                polygon[1].remove()
                polygon[1] = None
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
                    polygon[0] = Polygon(
                        [circle.center for circle in marks],
                        color=polygon_info[0]['color'], 
                        alpha=polygon_info[0]['alpha'],
                        zorder=polygon_info[0]['zorder']
                    )
                    ax.add_patch(polygon[0])
                    if ext is not None and ext > 0:
                        ext_marks = PolygonExtension([{'x':p.center[0], 'y':p.center[1]} for p in marks], ext)
                        polygon[1] = Polygon(
                            [(p['x'], p['y']) for p in ext_marks],
                            color=polygon_info[1]['color'], 
                            alpha=polygon_info[1]['alpha'],
                            zorder=polygon_info[1]['zorder']
                        )
                        ax.add_patch(polygon[1])
        elif event.button is MouseButton.RIGHT:
            if len(marks) > 0:
                circle = marks.pop(-1)
                circle.remove()
                text = texts.pop(-1)
                text.remove()
            if polygon[0] is not None:
                polygon[0].remove()
                polygon[0] = None
            if polygon[1] is not None:
                polygon[1].remove()
                polygon[1] = None
            if len(marks) >= 3:
                polygon[0] = Polygon(
                    [circle.center for circle in marks],
                    color=polygon_info[0]['color'], 
                    alpha=polygon_info[0]['alpha'],
                    zorder=polygon_info[0]['zorder']
                )
                ax.add_patch(polygon[0])
                if ext is not None and ext > 0:
                    ext_marks = PolygonExtension([{'x':p.center[0], 'y':p.center[1]} for p in marks], ext)
                    polygon[1] = Polygon(
                        [(p['x'], p['y']) for p in ext_marks],
                        color=polygon_info[1]['color'], 
                        alpha=polygon_info[1]['alpha'],
                        zorder=polygon_info[1]['zorder']
                    )
                    ax.add_patch(polygon[1])
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

def uiMark(background, title, default=None, fn=None, ext=None):
    size = (background.size[0] + background.size[1]) * 0.001
    fig, ax = plt.subplots(1)
    marks = []
    polygon = [None, None]
    texts=[]
    mark_info = {
        'size': 7*size,
        'color': 'red',
        'alpha': 0.6
        }
    polygon_info = [
        {
            'color': 'green',
            'alpha': 0.4,
            'zorder':0.8,
        },
        {
            'color': 'red',
            'alpha': 0.3,
            'zorder':0.7,
        }
    ]
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
            polygon[0] = Polygon(
                [circle.center for circle in marks],
                color=polygon_info[0]['color'], 
                alpha=polygon_info[0]['alpha'],
                zorder=polygon_info[0]['zorder']
            )
            ax.add_patch(polygon[0])
            if ext is not None and ext > 0:
                ext_marks = PolygonExtension([{'x':p.center[0], 'y':p.center[1]} for p in marks], ext)
                polygon[1] = Polygon(
                    [(p['x'], p['y']) for p in ext_marks],
                    color=polygon_info[1]['color'], 
                    alpha=polygon_info[1]['alpha'],
                    zorder=polygon_info[1]['zorder']
                )
                ax.add_patch(polygon[1])
    ax.imshow(background)
    plt.connect('scroll_event', onscroll(ax, marks, mark_info))
    plt.connect('button_press_event', onclick(marks, mark_info, polygon, polygon_info, texts, ax, size, ex_fn=fn, ext=ext))
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
    choice : List[Any]=[], 
    parser : collections.abc.Callable[[Any],Any] = lambda x:x
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

def OEDparse(OED:str)->List:
    nodes = []
    for node_desc in OED:
        triggers = []
        name_pos_end=re.match('\((\w+?)\)', node_desc).span()[1]
        triggers_desc = re.findall('(?:F|E|A)(?:{.*?})?', node_desc[name_pos_end:])
        for trigger_desc in triggers_desc:
            trigger_parameter =  dict()
            if len(trigger_desc) > 3:
                pstring = re.split(',\s?', trigger_desc[2:-1])
                kv = [re.split('\s?=\s?', p) for p in pstring]
                for i, (k, v) in enumerate(kv):
                    if re.match('^((-|\+)?\d+(\.\d*)?)$', v) is not None:
                        trigger_parameter[k] = float(v)
                    else:
                        trigger_parameter[k] = v
            trigger = None
            if trigger_desc[0] == 'F':
                trigger = {
                    'type':'Fence',
                    'parameter':{
                        'vertex_type':{
                            'x':trigger_parameter['vtx'] if 'vtx' in trigger_parameter else None,
                            'y':trigger_parameter['vty'] if 'vty' in trigger_parameter else None,
                        },
                        'position':None
                    }
                }
                if 'relay' in trigger_parameter:
                    trigger['parameter']['relay_ext'] = trigger_parameter['relay']
            elif trigger_desc[0] == 'E':
                trigger = {
                    'type':'Edge',
                }
            elif trigger_desc[0] == 'A':
                trigger = {
                    'type':'AttrFilter',
                    'parameter':{
                        'range':{
                            'min':trigger_parameter['min'] if 'min' in trigger_parameter else None,
                            'max':trigger_parameter['max'] if 'max' in trigger_parameter else None,
                        }
                    }
                }
            else:
                continue
            triggers.append(trigger)
        nodes.append({
            'name':re.findall('\((\w+?)\)', node_desc)[0],
            'triggers':triggers,
        })
    return nodes

def IEdit(
    node_name:str,
    trigger:Dict,
    all_update:bool
)->Dict:
    if trigger['type'] == 'Fence':
        vertex_type = trigger['parameter']['vertex_type']
        if all_update or vertex_type['x'] is None:
            console.print(f'[red bold][> Node-{node_name}][/] Please entering vertex type x ...')
            default = vertex_type['x']
            default_str = "" if default is None else f"([bold green]{default}[/])"
            vertex_type['x'] = prompt(f'\[vertex_type of x  <{default_str}] ', default=default, reg='^((-|\+)?\d+(\.\d*)?)$', parser=lambda x:float(x))
        if all_update or vertex_type['y'] is None:
            console.print(f'[red bold][> Node-{node_name}][/] Please entering vertex type y ...')
            default = vertex_type['y']
            default_str = "" if default is None else f"([bold green]{default}[/])"
            vertex_type['y'] = prompt(f'\[vertex_type of y  <{default_str}] ', default=default, reg='^((-|\+)?\d+(\.\d*)?)$', parser=lambda x:float(x))
        if all_update or 'relay_ext' not in trigger['parameter']:
            console.print(f'[red bold][> Node-{node_name}][/] Need enable relay moode ?')
            default = "No" if 'relay_ext' not in trigger['parameter'] else "Yes"
            default_str = f"([bold green]{default}[/])"
            enable = prompt(f'\[[red bold]relay mode]  <{default_str}] ', default=default, choice=["Yes", "No"])
            if enable=='Yes':
                default = trigger['parameter']['relay_ext'] if 'relay_ext' in trigger['parameter'] else None
                default_str = "" if default is None else f"([bold green]{default}[/])"
                trigger['parameter']['relay_ext'] = prompt(f'\[extension pixels  <{default_str}] ', default=default, reg='^((-|\+)?\d+(\.\d*)?)$', parser=lambda x:float(x))
        ext = trigger['parameter']['relay_ext'] if 'relay_ext' in trigger['parameter'] else None
        console.print(f'[red bold][> Node-{node_name}][/] Starting to label positions of virtual fence ...')
        marks = uiMark(background, "Please label positions of virtual fence", default=trigger['parameter']['position'], ext=ext)
        trigger['parameter']['position']=[{
            'x':float(mark.center[0]),
            'y':float(mark.center[1])
        } for mark in marks]
    elif trigger['type'] == 'Edge':
        pass
    elif trigger['type'] == 'AttrFilter':
        attr_range = trigger['parameter']['range']
        if all_update or attr_range['min'] is None:
            console.print(f'[red bold][> Node-{node_name}][/] Please entering minimum of attribute filter ...')
            default = attr_range['min']
            default_str = "" if default is None else f"([bold green]{default}[/])"
            attr_range['min'] = prompt(f'\[minimum of AttrFilter  <{default_str}] ', default=default, reg='^((-|\+)?\d+(\.\d*)?)$', parser=lambda x:float(x))
        if  all_update or attr_range['max'] is None:
            console.print(f'[red bold][> Node-{node_name}][/] Please entering of attribute filter ...')
            default = attr_range['max']
            default_str = "" if default is None else f"([bold green]{default}[/])"
            attr_range['max'] = prompt(f'\[maximum of AttrFilter  <{default_str}] ', default=default, reg='^((-|\+)?\d+(\.\d*)?)$', parser=lambda x:float(x))
    return trigger

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
            if (
                obj['BEV']['mapping']['width'] is None or
                obj['BEV']['mapping']['height'] is None
            ):
                process_pipeline.append('mapWH')
            if (
                obj['BEV']['mapping']['biasX'] is None or 
                obj['BEV']['mapping']['biasY'] is None or
                obj['BEV']['mapping']['viewW'] is None or
                obj['BEV']['mapping']['viewH'] is None
            ):
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
                    console.print('[>] Please type width([bold red]P2-P3[/] & [bold red]P4-P1[/]) of this square mapping for...')
                    default = obj['BEV']['mapping']['width']
                    default_str = "" if default is None else f"([bold green]{default}[/])"
                    obj['BEV']['mapping']['width'] = prompt(f'\[mapping width  <{default_str}] ', default=default, reg="^\d*$", parser=lambda x:int(x))
                    console.print('[>] Please type height([bold red]P1-P2[/] & [bold red]P3-P4[/]) of this square mapping for...')
                    default = obj['BEV']['mapping']['height']
                    default_str = "" if default is None else f"([bold green]{default}[/])"
                    obj['BEV']['mapping']['height'] = prompt(f'\[mapping height  <{default_str}] ', default=default, reg="^\d*$", parser=lambda x:int(x))
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
                    obj['BEV']['mapping']['biasX'] = int(0 - xmin)
                    obj['BEV']['mapping']['biasY'] = int(0 - ymin)
                    obj['BEV']['mapping']['viewW'] = int(xmax - xmin)
                    obj['BEV']['mapping']['viewH'] = int(ymax - ymin)
                    console.print(f'[>] Finish setting [bold red]perspective view area[/].')
                elif process == 'result':
                    mapH, mapW = obj['BEV']['mapping']['height'], obj['BEV']['mapping']['width']
                    points = [(p['x'], p['y']) for p in obj['BEV']['position']]
                    points = np.asarray(points, dtype='float32')
                    biasX, biasY = obj['BEV']['mapping']['biasX'], obj['BEV']['mapping']['biasY']
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

            if (
                len(obj['trigger_description']['collector']) > 0 or
                len(obj['trigger_description']['base']) > 0
            ):
                console.print(f'[>] Read trigger description from file:{targetName}')
            else:
                process_pipeline.append('pre_check')
            if (
                obj['trigger_description']['threshold']['target'] is not None and
                obj['trigger_description']['threshold']['nonTarget'] is not None and
                obj['trigger_description']['threshold']['waiting'] is not None
            ):
                console.print(f'[>] Read event threshold from file:{targetName}')
            else:
                process_pipeline.append('threshold')
            process_pipeline.append('result')
            console.print(f'\n[>] [bold]Following operation will modify the event filter model.[/]')
            while len(process_pipeline) > 0:
                process = process_pipeline.popleft()
                if process == 'pre_check':
                    reg1 = '^\d*$'
                    name_reg = '\(\w+\)'
                    trigger_desc_reg = '(?:F|E|A)(?:{.*?})?'
                    trigger_reg = f'((?:{name_reg})(?:{trigger_desc_reg})+);?'
                    reg2 = f'({trigger_reg})+'
                    reg = f'({reg1})|({reg2})'
                    re.findall(reg2, '(name)F{vtx=1.0, vty=1.0}>>F{vtx=1.0, vty=1.0}')
                    for target in ["collector", "base"]:
                        console.print(f'[>] Please type amount of {target} node you want...')
                        value = prompt(f'\[{target} <] ', reg=reg, parser=lambda x:int(x) if re.match(reg1, x) is not None else x)
                        if isinstance(value, int):
                            for i in range(value):
                                console.print(f'[>] Please type name of [red bold]Node{i}[/]...')
                                name = prompt(f'\[name  <] ')
                                triggers = []
                                while True:
                                    enter = prompt(f'[>] Which type of trigger you want?', default=None, choice=['Fence', 'Edge', 'AttrFilter', 'exit'])
                                    if enter == 'Fence':
                                        trigger = {
                                            'type':'Fence',
                                            'parameter':{
                                                'vertex_type':{
                                                    'x':None,
                                                    'y':None,
                                                },
                                                'position':None
                                            }
                                        }
                                    elif enter == 'Edge':
                                        trigger = {
                                            'type':'Edge',
                                        }
                                    elif enter == 'AttrFilter':
                                        trigger = {
                                            'type':'AttrFilter',
                                            'parameter':{
                                                'range':{
                                                    'min':None,
                                                    'max':None,
                                                }
                                            }
                                        }
                                    elif enter == "exit":
                                        console.print('[red bold][> Node-{name}][/] Finish!')
                                        break
                                    triggers.append(IEdit(name, trigger, all_update=False))
                                obj['trigger_description'][target].append({
                                    'name':name,
                                    'triggers':triggers,
                                })
                        else:
                            console.print(value)
                            nodes_desc = re.findall(trigger_reg, value)
                            nodes = OEDparse(nodes_desc)                    
                            console.print(f'[yellow bold]Parse {len(nodes)} nodes for {target} from OED.')
                            for i, node in enumerate(nodes):
                                if node['name'] is None:
                                    console.print(f'[>] Please type name of [red bold]Node{i}[/]...')
                                    node['name'] = prompt(f'[name  <] ')
                                for trigger in node['triggers']:
                                    trigger = IEdit(node['name'], trigger, all_update=False)
                                console.print(f'[green bold][>] OED for Node-{node["name"]} accept.')
                            obj['trigger_description'][target] = nodes
                elif process == 'threshold':
                    threshold = obj['trigger_description']['threshold']
                    for kind in ['target', 'nonTarget', 'waiting']:
                        console.print(f'[>] Please enter limited [bold red]{kind}[/] time(s)...')
                        default = threshold[kind]
                        default_str = "" if default is None else f"([bold green]{default}[/])"
                        threshold[kind] = prompt(f'\[{kind} time(s)  <{default_str}] ', default=default, reg='^((-|\+)?\d+(\.\d*)?)$', parser=lambda x:float(x))
                elif process == "event":
                    endpoint_str = prompt(f'\[endpoint  <] ', default='collector', choice=['collector', 'base'])
                    endpoint = obj['trigger_description'][endpoint_str]
                    nodes_name = [node['name'] for node in endpoint]
                    node_str = prompt(f'\[node  <] ', default=nodes_name[0], choice=nodes_name)
                    node_idx = nodes_name.index(node_str)
                    node = endpoint[node_idx]
                    for trigger in node['triggers']:
                        trigger = IEdit(node_str, trigger, all_update=True)
                elif process == 'result':
                    then = prompt(f'[>] Edit again?', default='No', choice=['event', 'threshold', 'No'])
                    if then == 'No':
                        break
                    else:
                        process_pipeline.append(then)
                        process_pipeline.append('result')
            
            with open(confName, 'w') as outfile:
                yaml.dump(obj, outfile, default_flow_style=False, sort_keys=False)
