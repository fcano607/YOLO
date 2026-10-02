"""Inspect M1-02 architecture against the locked YAML and existing forward evidence.

No camera, training or new forward pass. Saves one structural record and overview.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'logs/ultralytics_settings')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'

from deploy.paths import resolve_path
import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def overview(path, layers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.is_file():
        plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(15, 8), layout='constrained')
    ax.set_xlim(-1.5, 16.5)
    ax.set_ylim(-0.8, 9.5)
    ax.axis('off')
    positions = {4:(0,6), 6:(0,4), 10:(0,2), 13:(4,4), 16:(4,6), 19:(8,4), 22:(8,2), 23:(12,4)}
    fills = {4:'#e7eef8',6:'#e7eef8',10:'#e7eef8',13:'#e7f3ed',16:'#e7f3ed',19:'#e7f3ed',22:'#e7f3ed',23:'#fff0dc'}
    edges = [(4,6),(6,10),(10,13),(6,13),(13,16),(4,16),
             (16,19),(13,19),(19,22),(10,22),(16,23),(19,23),(22,23)]
    for start, end in edges:
        a, b = positions[start], positions[end]
        color = '#d18430' if end == 23 else '#5c7b91'
        radius = -0.15 if (start,end) in [(16,23),(22,23)] else 0
        dx,dy=b[0]-a[0],b[1]-a[1]
        extent=min(1.22/abs(dx) if dx else float('inf'),0.64/abs(dy) if dy else float('inf'))
        start_point=(a[0]+extent*dx,a[1]+extent*dy)
        end_point=(b[0]-extent*dx,b[1]-extent*dy)
        ax.add_patch(FancyArrowPatch(start_point,end_point,arrowstyle='-|>',mutation_scale=15,
                                    shrinkA=2,shrinkB=2,color=color,lw=1.5,
                                    connectionstyle=f'arc3,rad={radius}',zorder=1))
    for index,(x,y) in positions.items():
        title = f'{index} {layers[index]["module"]}'
        if index == 23:
            label = title + '\n框 / 类别 / 系数 / 原型'
        else:
            shape = layers[index]['recorded_output_shape']
            label = title + '\n' + '×'.join(str(n) for n in shape[1:])
            if index in (16,19,22):
                label += f'\nP{ {16:3,19:4,22:5}[index] }'
        ax.add_patch(FancyBboxPatch((x-1.15,y-0.57),2.3,1.14,boxstyle='round,pad=0.06',
                                   facecolor=fills[index],edgecolor='#73869c',lw=1.1,zorder=2))
        ax.text(x,y,label,ha='center',va='center',fontsize=11,zorder=3)
    ax.text(0,7.45,'Backbone：0—10',ha='center',fontsize=15,fontweight='bold')
    ax.text(4,7.45,'融合：先上采样',ha='center',fontsize=15,fontweight='bold')
    ax.text(8,7.45,'融合：再下采样',ha='center',fontsize=15,fontweight='bold')
    ax.text(12,7.45,'Segment：23',ha='center',fontsize=15,fontweight='bold')
    ax.text(0,8.2,'输入 3×640×640\n逐级 Conv / C3k2',ha='center',fontsize=11)
    ax.text(0,0.85,'第 9 层 SPPF → 第 10 层 C2PSA\n保留 20×20 的空间尺寸',ha='center',fontsize=10,color='#42566f')
    ax.text(4,3,'10 ↑2 + 6 → 13\n13 ↑2 + 4 → 16',ha='center',fontsize=10,color='#2a6650')
    ax.text(8,6.25,'16 ↓2 + 13 → 19\n19 ↓2 + 10 → 22',ha='center',fontsize=10,color='#2a6650')
    ax.text(12,2.2,'P3 / P4 / P5\n步长 8 / 16 / 32',ha='center',fontsize=11)
    ax.add_patch(FancyArrowPatch((13.2,4),(14.4,4),arrowstyle='-|>',mutation_scale=16,color='#d18430'))
    ax.text(15.3,4,'候选\n116×8400\n\n共享原型\n32×160×160',ha='center',va='center',fontsize=11)
    ax.text(7.3,9,'M1-02：当前 YOLO11n-seg 的结构概览',ha='center',fontsize=19,fontweight='bold')
    ax.text(7.3,-0.3,'图中省略中间 Conv / Upsample / Concat；完整 24 模块与 30 条模块输入连接见 Markdown。尺寸取自已有 GPU 前向。',
            ha='center',fontsize=10,color='#526275')
    fig.savefig(path,dpi=160)
    plt.close(fig)


def run(args):
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.modules import C2f,C2PSA,C3k2,Concat,Conv,Proto,Segment,SPPF
    from ultralytics.nn.tasks import parse_model
    output = resolve_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new, empty --output to preserve evidence')
    lock = json.loads((ROOT/'configs/source-lock.json').read_text('utf-8'))
    e0 = yaml.safe_load((ROOT/'configs/e0.yaml').read_text('utf-8'))
    source = resolve_path(lock['path'])
    assert ultralytics.__version__ == lock['version']
    assert Path(ultralytics.__file__).resolve() == source/'ultralytics/__init__.py'
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() == lock['commit']
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
    weights = resolve_path(e0['weights']['path'])
    assert sha(weights) == e0['weights']['sha256']
    evidence_path = resolve_path(args.evidence)
    evidence = json.loads(evidence_path.read_text('utf-8'))
    assert evidence['status']=='engineering_checks_passed' and evidence['source_commit']==lock['commit']
    assert evidence['weights']['sha256']==sha(weights)
    for ref in evidence['source_refs'].values():
        assert sha(ROOT/ref['path'])==ref['sha256'], 'Source differs from earlier forward evidence'
    model = YOLO(str(weights)).model.cpu().float().eval()
    yaml_path = source/'ultralytics/cfg/models/11/yolo11-seg.yaml'
    config = yaml.safe_load(yaml_path.read_text('utf-8'))
    config_matches = {key:config[key]==model.yaml[key] for key in ['backbone','head','scales','nc']}
    assert all(config_matches.values())
    rows = config['backbone']+config['head']
    assert len(rows)==len(model.model)==len(evidence['layers'])==24
    depth,width,max_channels = config['scales']['n']
    layers=[]
    edges=[]
    for index, (module,entry,row) in enumerate(zip(model.model,evidence['layers'],rows)):
        f,repeats,name,arguments=row
        assert module.i==index and module.f==f==entry['from']
        assert type(module).__name__==entry['module']==name.removeprefix('nn.')
        actual_parameters=sum(p.numel() for p in module.parameters())
        assert actual_parameters==entry['parameters']
        sources=f if isinstance(f,list) else [f]
        sources=[index-1 if value==-1 else value for value in sources]
        edges.extend({'from':value,'to':index} for value in sources)
        record={'index':index,'from':f,'module':type(module).__name__,
                'yaml_repeats':repeats,'yaml_args':arguments,'parameters':actual_parameters,
                'recorded_output_shape':entry['output'].get('shape'),
                'resolved_inputs':sources,'section':'backbone' if index<11 else 'segment' if index==23 else 'fusion'}
        if isinstance(module,(C3k2,C2PSA)):
            scaled_repeats=max(round(repeats*depth),1) if repeats>1 else repeats
            assert len(module.m)==scaled_repeats
            record.update({'internal_repeats':len(module.m),'internal_types':[type(m).__name__ for m in module.m],
                           'hidden_channels':module.c})
        if isinstance(module,Conv):
            record['conv']={'in_channels':module.conv.in_channels,'out_channels':module.conv.out_channels,
                            'kernel':list(module.conv.kernel_size),'stride':list(module.conv.stride),
                            'padding':list(module.conv.padding),'activation':type(module.act).__name__}
        if isinstance(module,Concat):
            shapes=[evidence['layers'][i]['output']['shape'] for i in sources]
            out=entry['output']['shape']
            assert module.d==1 and sum(shape[1] for shape in shapes)==out[1]
            assert all(shape[2:]==out[2:] for shape in shapes)
            record.update({'concat_dimension':module.d,'input_shapes':shapes})
        if isinstance(module,torch.nn.Upsample):
            record['upsample']={'scale_factor':module.scale_factor,'mode':module.mode}
        if isinstance(module,SPPF):
            default=SPPF(module.cv1.conv.in_channels,module.cv2.conv.out_channels,5)
            record['sppf']={'pool_kernel':module.m.kernel_size,'pool_stride':module.m.stride,
                             'pool_padding':module.m.padding,'pool_repetitions':getattr(module,'n',3),
                             'loaded_cv1_activation':type(module.cv1.act).__name__,
                             'fresh_constructor_cv1_activation':type(default.cv1.act).__name__,
                             'loaded_shortcut':getattr(module,'add',False)}
        if isinstance(module,Segment):
            assert f==[16,19,22] and module.nc==80 and module.nm==32 and module.npr==64
            record['segment']={'nc':module.nc,'nm':module.nm,'npr':module.npr,'reg_max':module.reg_max,
                                'strides':module.stride.tolist(),'input_channels':[evidence['layers'][i]['output']['shape'][1] for i in f],
                                'branches':{key:[branch[-1].out_channels for branch in getattr(module,key)]
                                            for key in ['cv2','cv3','cv4']},
                                'proto_input_layer':f[0],'proto_upsample':type(module.proto.upsample).__name__}
        layers.append(record)
    assert sha(weights)==e0['weights']['sha256']
    symbols=[Conv,Concat,C2f,C3k2,SPPF,C2PSA,Proto,Segment,parse_model]
    source_refs={symbol.__name__:{'path':rel(inspect.getsourcefile(symbol)),
                                  'line':inspect.getsourcelines(symbol)[1],
                                  'sha256':sha(Path(inspect.getsourcefile(symbol)))} for symbol in symbols}
    report={'task':'M1-02','checked_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
             'status':'structure_checks_passed','source_commit':lock['commit'],'yaml_path':rel(yaml_path),
             'yaml_sha256':sha(yaml_path),'weight_sha256':sha(weights),'yaml_sections_match_checkpoint':config_matches,
             'scale':{'name':'n','depth':depth,'width':width,'max_channels':max_channels},
             'forward_evidence':{'path':rel(evidence_path),'sha256':sha(evidence_path),
                                 'original_checked_at':evidence['checked_at']},
             'layers':layers,'input_edges':edges,'source_refs':source_refs,'script_sha256':sha(Path(__file__)),
             'scope':{'new_forward_pass':False,'training':False,'model_modified':False,'human_learning_acceptance':'pending'}}
    output.mkdir(parents=True,exist_ok=True)
    overview(output/'structure_overview.png',layers)
    report['diagram_sha256']=sha(output/'structure_overview.png')
    (output/'structure_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'output':rel(output),'top_level_modules':len(layers),
                       'input_edges_including_image':len(edges),'concat_modules_checked':sum(item['module']=='Concat' for item in layers),
                       'c3k2_internals':{str(item['index']):item['internal_types'] for item in layers if item['module']=='C3k2'},
                       'sppf':layers[9]['sppf'],'segment':layers[23]['segment']},ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',default='reports/model/M1/summary.json')
    parser.add_argument('--output',default='reports/model/M1-02')
    run(parser.parse_args())


if __name__=='__main__':
    main()
