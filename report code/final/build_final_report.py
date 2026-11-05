"""2026-11-05 最终报告及三次验收 第1次修改版。只读已完成运行，不启动训练。"""
from pathlib import Path
import sys
import argparse
import json
import re
import shutil
import zipfile
from urllib.parse import unquote
import hashlib
import platform
import pandas as pd
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from config import save_json, digest, DATA_PATH


def check_links(report):
    report=Path(report)
    targets=re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8'))
    for target in targets:
        path=(report.parent/unquote(target)).resolve()
        if not path.is_relative_to(report.parent.resolve()) or not path.is_file():
            raise ValueError(f'非法或缺失附件：{target}')
    return len(targets)


def check_code_zip(path):
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            p=Path(name)
            if p.suffix in ('.csv','.npz','.pt','.joblib','.pyc') or '__pycache__' in name or '..' in p.parts:
                raise ValueError(f'代码包含非交付文件：{name}')


def mdtable(frame):
    cols=list(frame.columns)
    def fmt(v):
        return f'{v:.4f}' if isinstance(v,float) else str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(cols)+' |','|'+'---|'*len(cols)]+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def copy_attachment(source, folder, task, label):
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'任务{task}+{label}{source.suffix}'
    shutil.copy2(source,target)
    return target


def build_acceptance(run, number):
    folder=ROOT/'验收'/f'acceptance {number}'
    folder.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    lines=[f'# 任务1至{8 if number==1 else 7}+第{number}次阶段验收记录',
           '',f'生成日期：2026-11-05；第1次修改版。来源运行：`{run.name}`。',
           f'数据 SHA-256：`{manifest["data_sha256"]}`。',
           '原始 CSV 未改动。模型仅在 2000—2003 年训练；所有折内选择和缩放使用本折训练数据。',
           '正式回溯测试统一为 359 个预测原点（2004-01-01 至 2004-12-24），标签为 t+1…t+7。',
           '三份验收记录来自同一冻结运行；R²不是相关系数的平方。']
    def attachment(path,task,label,image=False):
        target=copy_attachment(path,folder/('figures' if image else 'data'),task,label)
        rel=target.relative_to(folder).as_posix()
        if image:
            lines.extend(['',f'![{label}]({rel})',f'[打开原图：{label}]({rel})'])
        else:
            lines.extend(['',f'[{label}]({rel})'])
    metrics=pd.read_csv(run/'metrics.csv')
    if number==1:
        for task,title,filename,desc in [
            (1,'英文字段中文含义','fields.csv','逐项说明日期、径流及七个气象字段。Dayl 是日长，不是实测日照小时数；流量单位未由教师资料明确，不擅自换算。'),
            (2,'全部非日期列时间序列','timeseries.png','八个非日期字段全部展示，包括恒零 Swe。红线为训练/测试年份分界。'),
            (3,'全部非日期列直方图','histogram.png','包含恒零 Swe，避免只画建模所选变量造成缺项。'),
            (4,'全部非日期列箱型图','boxplot.png','极端观测保留，不删除洪峰。箱型图反映分布和极端值，不把极端值自动判为错误。'),
            (5,'径流与因子及其滞后相关性','pearson.png','Pearson 热力图包括原径流、径流滞后1/3/7天及各因子的原序列和滞后序列。Swe 为常量，相关系数无定义，未放入热力图。'),
            (6,'季节月份滞后窗口特征','feature_catalog.csv','四季按12—2月冬、3—5月春、6—8月夏、9—11月秋编码；月份正余弦编码。滞后0/1/2/3/7/14/30天，滚动3/7/14天均值、标准差、最大值，均截止预测原点 t。'),
            (7,'Z-score标准化','standardization.png','常见方法包括 Min-Max、Z-score 和鲁棒缩放。本次以各训练折均值和标准差拟合 Z-score，验证/测试只 transform。RF 无需尺度缩放但沿用统一输入，排序不因线性缩放改变。'),
            (8,'训练期互信息特征选择',None,'常见方法分过滤法（相关系数、互信息）、包装法（逐步选择）和嵌入法（树重要性/L1）。本次用训练折目标逐h互信息平均排序，选择候选配置指定数量；不使用测试标签。')]:
            lines.extend(['',f'## 任务{task}+{title}','',desc])
            if filename:
                attachment(run/'analysis'/filename,task,title,filename.endswith('.png'))
                if filename=='fields.csv': lines.extend(['',mdtable(pd.read_csv(run/'analysis'/filename))])
            if task==6: attachment(run/'analysis'/'training_features.csv',6,'训练期特征明细')
            if task==7: attachment(run/'analysis'/'standardized.csv',7,'标准化结果')
            if task==8: attachment(run/'features'/'direct_RF_h1.csv',8,'互信息评分与选中特征')
        attachment(run/'analysis'/'statistics.csv',1,'原始数据描述统计')
        attachment(run/'analysis'/'pearson.csv',5,'Pearson完整系数表')
        attachment(run/'features'/'direct_RF_h1_scalers.json',7,'训练期缩放参数')
    else:
        if number==2:
            lines.extend(['','## 任务1+适用方法','',
                '线性回归用于可解释基线；支持向量回归适合非线性小样本；随机森林用树的平均降低方差；ANN 学习非线性特征组合；LSTM 通过循环状态处理历史序列。本实验实现教师指定 ANN/RF/LSTM，另两种仅作方法介绍。'])
            specs=[(2,'ANN',None),(3,'ANN','direct'),(4,'ANN','multi'),(5,'RF',None),(6,'RF','direct'),(7,'RF','multi')]
        else:
            specs=[(1,'LSTM',None),(2,'LSTM','direct'),(3,'LSTM','multi'),(4,'Mean','direct'),(5,'Stacking','direct'),(6,'Mean','multi'),(7,'Stacking','multi')]
        frozen=json.loads((run/'frozen.json').read_text(encoding='utf-8'))
        for task,model,strategy in specs:
            if strategy is None:
                lines.extend(['',f'## 任务{task}+{model}网格搜索','',
                    '使用 GridSearchCV，训练期 TimeSeriesSplit 两折、gap=7、原始尺度 R²评分。直接策略逐h选参，多输出按七步 R²均值选参。OOF 另做三折外层、每折内独立两折选参，保证参数选择无验证标签泄漏。'])
                selected=pd.DataFrame([dict(model=k,cv_R2=v['cv_R2'],parameters=json.dumps(v['params'],ensure_ascii=False)) for k,v in frozen['selected'].items() if f'_{model}_' in k])
                paramfile=run/'analysis'/f'{model}_selected.csv';selected.to_csv(paramfile,index=False,encoding='utf-8-sig')
                lines.extend(['',mdtable(selected)])
                attachment(paramfile,task,'最终参数与训练期验证分数')
                archive=run/'analysis'/f'{model}_grid.zip'
                with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                    for p in (run/'cv').glob(f'*_{model}_*.csv'):z.write(p,p.name)
                attachment(archive,task,'完整网格搜索结果')
            else:
                subset=metrics[(metrics.model==model)&(metrics.strategy==strategy)]
                lines.extend(['',f'## 任务{task}+{model} {strategy}策略预测精度','',mdtable(subset[['h','R2','RMSE','MAE','n']])])
                p=run/'analysis'/f'{strategy}_{model}.csv';subset.to_csv(p,index=False,encoding='utf-8-sig')
                attachment(p,task,f'{model}{strategy}指标')
                attachment(run/'analysis'/f'{strategy}_curves.png',task,f'{strategy}实测预测过程线',True)
                if model=='Mean':lines.extend(['','简单平均严格对 ANN、RF、LSTM 同日期、同h的原始尺度预测取算术平均，不替换任何基模型。'])
                if model=='Stacking':lines.extend(['','Ridge 元学习器只拟合训练期嵌套 OOF 预测；直接策略每h使用三列元特征，多输出策略一次拟合21列基预测到七个目标。测试标签不参与拟合；不存在以RF结果冒充Stacking的回退。'])
        attachment(run/'oof'/'splits.csv',1 if number==3 else 2,'OOF时间边界')
        attachment(run/'development_scores.csv',1 if number==3 else 2,'训练期逐折开发指标')
        attachment(run/'predictions.csv',2 if number==3 else 3,'全部正式逐日预测')
        attachment(run/'baseline.csv',2 if number==3 else 3,'同样本持续性基线')
    attachment(run/'protocol.json',1,'冻结搜索协议')
    attachment(run/'manifest.json',1,'运行清单')
    best=metrics[metrics.h==1].sort_values('R2',ascending=False).iloc[0]
    lines.extend(['','## 结果与边界','',f'同次运行全模型最高 h=1 R²为 {best.R2:.4f}（{best.strategy}/{best.model}）。'+
        ('未达到约0.8目标，保留真实指标，不继续用2004调参。' if best.R2<.8 else '达到约0.8目标。'),
        '2004 已在旧项目被查看，本次称回溯评估，不声称全新盲测。后续h允许自然下降，不对指标曲线强制排序或修饰。'])
    name=folder/f'任务1至{8 if number==1 else 7}+第{number}次阶段验收记录.md'
    name.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return name,check_links(name)


def build_word(run, destination):
    template=Path(__file__).with_name('template.docx')
    doc=Document(template)
    if len(doc.sections)!=3 or len(doc.tables)!=1 or not any(p.text=='一、任务要求' for p in doc.paragraphs):
        raise ValueError('模板副本的章节结构与教师模板不符')
    originals=doc.paragraphs[:]
    metrics=pd.read_csv(run/'metrics.csv')
    frozen=json.loads((run/'frozen.json').read_text(encoding='utf-8'))
    protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    stats=pd.read_csv(run/'analysis'/'statistics.csv',index_col=0)
    best=metrics[metrics.h==1].sort_values('R2',ascending=False).iloc[0]
    paragraphs=[]
    def p(anchor,text,bold=False):
        para=anchor.insert_paragraph_before(text)
        para.paragraph_format.space_after=Pt(6)
        para.paragraph_format.line_spacing=1.3
        para.paragraph_format.keep_with_next=bold
        for r in para.runs:
            r.font.name='宋体';r._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
            r.font.size=Pt(10.5);r.font.color.rgb=RGBColor(0,0,0);r.bold=bold
        paragraphs.append(text)
        return para
    def table(anchor,frame):
        t=doc.add_table(rows=1,cols=len(frame.columns));t.autofit=False
        available=Cm(14.6)
        for cell,col in zip(t.rows[0].cells,frame.columns):cell.text=str(col)
        for row in frame.itertuples(index=False,name=None):
            for c,v in zip(t.add_row().cells,row):c.text=f'{v:.4f}' if isinstance(v,float) else str(v)
        weights=[1]*len(frame.columns)
        if len(frame.columns)==5:weights=[2.3,1.1,1.3,.6,3.0]
        if len(frame.columns)==3 and '处理目标' in frame.columns:weights=[1,2,3]
        for col,weight in zip(t.columns,weights):col.width=int(available*weight/sum(weights))
        for rownum,row in enumerate(t.rows):
            for index,cell in enumerate(row.cells):
                cell.width=int(available*weights[index]/sum(weights))
                for para in cell.paragraphs:
                    para.paragraph_format.space_after=Pt(3)
                    para.paragraph_format.space_before=Pt(3)
                    for r in para.runs:
                        r.font.size=Pt(9);r.font.name='宋体';r._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
                        r.bold=rownum==0
                props=cell._tc.get_or_add_tcPr()
                borders=OxmlElement('w:tcBorders')
                for edge in ('top','left','bottom','right'):
                    e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
                props.append(borders)
                if rownum==0:
                    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E7EEF5');props.append(shade)
            prop=row._tr.get_or_add_trPr()
            cannot=OxmlElement('w:cantSplit');prop.append(cannot)
            if rownum==0:prop.append(OxmlElement('w:tblHeader'))
        anchor._p.addprevious(t._tbl)
    def pic(anchor,name,caption):
        para=p(anchor,'')
        para.paragraph_format.keep_with_next=True
        para.add_run().add_picture(str(run/'analysis'/name),width=Cm(14.5))
        p(anchor,caption)
    cover=doc.tables[0]
    for i,value in enumerate(('基于机器学习的径流预测','待填写','待填写（三名成员）','待填写（三名成员）','待填写','待填写')):
        cover.cell(i,1).text=value
    # 只更改日期文本，保留该段末尾的分节属性。
    originals[5].text='二〇二六 年 十一 月 五 日'
    a=originals[7]
    p(a,'本课题根据流域逐日径流和气象观测，预测未来1—7天的径流，属于监督学习中的时间序列多步回归任务。输入为预测原点t当天已观测值及其历史，输出为t+1至t+7的流量。')
    p(a,'实验依次完成数据分析、因果特征工程、人工神经网络/随机森林/LSTM建模、简单平均与Stacking集成。三种基模型各构建七个直接模型与一个七输出模型，共24个；两类集成共16个。全程只使用教师CSV。')
    p(a,f'正式回溯测试的最高h=1 R²为{best.R2:.4f}（{best.strategy}/{best.model}），RMSE为{best.RMSE:.2f}，MAE为{best.MAE:.2f}。'+('本次未达到约0.8目标。' if best.R2<.8 else '达到约0.8目标。')+'所有结果以原始流量尺度报告，不将相关系数平方替代R²。')
    a=originals[10]
    table(a,pd.DataFrame([['待填写','待填写','待填写真实承担任务'],['待填写','待填写','待填写真实承担任务'],['待填写','待填写','待填写真实承担任务']],columns=['成员姓名','学号','具体分工']))
    a=originals[13]
    table(a,pd.DataFrame([
        ['数据读取','pandas与日期断言','核验1827行、9列、连续日期和缺失'],
        ['数据分析','直方图/箱型图/Pearson','解释分布、极端值和滞后相关'],
        ['特征工程','日历编码/滞后/滚动/MI/Z-score','构建预测时点可得输入；折内拟合'],
        ['个体建模','ANN/RF/LSTM及GridSearchCV','训练期选参，原生多输出'],
        ['融合','简单平均/嵌套OOF/Ridge','生成真实集成预测'],
        ['冻结评价','359个原点/三指标','一次评价2004，归档报告与代码']
    ],columns=['步骤','工具或方法','处理目标']))
    p(a,'训练数据为2000—2003年。为保证七步标签全部留在训练期，训练预测原点最晚到2003-12-24。正式测试预测原点为2004-01-01至2004-12-24，共359行。每行标签为未来1—7天，同一原点同时用于所有策略。2004年内较早日期的已观测径流可作为随后原点的输入，这是滚动预报，不是用测试标签拟合模型。')
    p(a,'开发采用两折TimeSeriesSplit，gap=7。Stacking采用三折外层OOF，每个外层训练段再独立两折网格搜索。训练标签最晚日期必须早于验证起点。每次fit重新计算互信息、缩放和模型参数；验证与测试不参与这些统计。')
    a=originals[16]
    p(a,'开发环境与数据核验',True)
    devices=json.loads((run/'analysis'/'devices.json').read_text(encoding='utf-8'))
    p(a,'操作系统Windows；Python '+manifest['python']+'；'+ '，'.join(f'{k} {v}' for k,v in protocol['versions'].items())+
        '。保存的LSTM模型记录训练设备为'+','.join(devices['training_devices'])+'；报告生成时GPU核验为'+str(devices['current_gpu'])+'。所有随机种子固定为42。')
    p(a,'CSV包含2000-01-01至2004-12-31的1827条日记录，字段无缺失。Swe恒零，仅在建模时排除，原始观测保持不变。Discharge的物理单位未在教师资料中明确，本报告保留原值并将RMSE/MAE标为原始流量单位，不擅自换算。')
    table(a,pd.read_csv(run/'analysis'/'fields.csv')[['field','中文含义','单位']])
    table(a,stats[['mean','std','min','max']].reset_index().rename(columns={'index':'字段'}))
    for image,caption in [('timeseries.png','全部八个非日期字段时间序列，红线为2004年分界'),('histogram.png','全部八个非日期字段直方图'),('boxplot.png','全部八个非日期字段箱型图'),('pearson.png','径流、因子及1/3/7天滞后Pearson相关性')]:pic(a,image,caption)
    p(a,'特征构造与训练折内处理',True)
    p(a,'构建月份正余弦与四季one-hot，冬季为12—2月。非恒定字段构建滞后0/1/2/3/7/14/30天；径流、降水、最高/最低气温构建3/7/14天滚动均值、标准差、最大值，窗口截止t。总计91列候选输入。14天LSTM样本仅由历史序列组成；候选可使用最后7天或14天。')
    p(a,'常见缩放包括Min-Max、Z-score、鲁棒缩放；本次使用Z-score。常见选择法包括过滤、包装、嵌入；本次以逐目标互信息的平均值排序，选择16或24列。全部拟合限于本折训练行；正式模型以全部训练期拟合后只transform测试输入。')
    pic(a,'standardization.png','训练特征Z-score处理前后分布')
    selected=pd.read_csv(run/'features'/'direct_RF_h1.csv').query('selected == True').sort_values('MI',ascending=False)
    table(a,selected[['feature','MI']].head(12))
    p(a,'网格搜索和目标变换',True)
    p(a,'训练期先导搜索显示h=1验证R²约0.67—0.69，随后加入相对当天已知径流的残差候选。目标候选为原始值、平方根和log1p；残差方案在相应目标尺度预测相对Q(t)的变化，再逆变换恢复流量。最终模型均由冻结网格决定，测试结果不反向影响选择。')
    p(a,'ANN为单隐层MLP；RF使用120棵树并搜索叶节点最小样本数；LSTM为单层循环网络、线性输出头和Adam优化器。七输出网络不是七个独立包装器。LSTM以purged CV选择30或70轮训练，不采用随机验证集早停。')
    selected_rows=[]
    for key,v in frozen['selected'].items():
        cfg=v['params']
        detail=', '.join(f'{k}={value}' for k,value in cfg.items() if k not in ('top_k','transform'))
        selected_rows.append([key.replace('direct_','直/').replace('multi_','多/').replace('_h','/h'),round(v['cv_R2'],4),cfg['transform'],cfg['top_k'],detail])
    table(a,pd.DataFrame(selected_rows,columns=['模型','CV R²','目标变换','k','参数']))
    p(a,'正式预测与集成结果',True)
    p(a,'R²=1−Σ(y−ŷ)²/Σ(y−ȳ)²；RMSE为均方误差平方根，MAE为绝对误差均值。R²允许为负数，不截断也不将其改为相关系数平方。所有预测通过相同非负约束处理，指标保持原始尺度。')
    p(a,'简单平均对三个基模型原尺度预测等权平均。Stacking仅用训练期嵌套OOF预测拟合Ridge；直接策略每h三列元特征，多输出为21列基预测到七个目标。两种集成均为实际生成结果，未退回单模型代替。')
    for strategy in ('direct','multi'):
        p(a,'多步直接预测' if strategy=='direct' else '原生多输出预测',True)
        pic(a,f'{strategy}_metrics.png',f'{strategy}策略七个预见期的R²/RMSE/MAE')
        for model in ('ANN','RF','LSTM','Mean','Stacking'):
            p(a,f'{strategy} / {model}',True)
            table(a,metrics[(metrics.strategy==strategy)&(metrics.model==model)][['h','R2','RMSE','MAE']])
        pic(a,f'{strategy}_curves.png',f'{strategy}策略h=1及h=7实测与预测过程线')
    baseline=pd.read_csv(run/'baseline.csv')
    p(a,'持续性基线与误差分析',True)
    table(a,baseline[['h','R2','RMSE','MAE']])
    errors=pd.read_csv(run/'analysis'/'largest_errors.csv')
    predictions=pd.read_csv(run/'predictions.csv')
    chosen=predictions[(predictions.strategy==best.strategy)&(predictions.model==best.model)&(predictions.h==1)]
    share=errors.squared_error.sum()/((chosen.prediction-chosen.truth)**2).sum()
    p(a,f'正式h=1最高R²模型的20个最大误差日占其平方误差总和的{share:.1%}。该统计只用于冻结后的解释，未用来重选模型。极端值被保留；若模型对少数洪峰误差较大，R²与RMSE会明显下降，MAE则相对不敏感。')
    table(a,errors[['origin','target_date','truth','prediction']].head(8))
    a=originals[19]
    p(a,'日期与多步标签边界：起初的实现需求只写按年份切分，不足以证明七步标签未跨界。先写因果与日期失败测试，再实现训练标签截止日断言、gap=7和359个共同测试原点，边界测试通过。')
    p(a,'输入/输出一致性：为ANN、RF、LSTM分别编写原生七输出、单输出列形状、训练折缩放和序列化恢复测试，确认预测矩阵与实测日期对应。残差扩展先由失败测试暴露不支持参数，再验证在零变化目标下能重构当天流量。')
    p(a,'ANN收敛：快速单元测试故意只运行10轮，会出现ConvergenceWarning；正式训练的警告数量记录在每个搜索JSON中，不隐藏为已完全收敛。网格预算固定，未在正式测试后追加轮次。')
    p(a,'文档转换：打包渲染器缺少LibreOffice可执行文件，已记录错误。改用Microsoft Word转换教师.doc副本并导出PDF，通过逐页图像检查报告版面；没有改变教师原文件。')
    a=originals[22]
    p(a,'使用技术：pandas日期索引、numpy序列、matplotlib可视化、Pearson分析、月份正余弦/季节one-hot、滞后/滚动、互信息、Z-score、MLP、随机森林、PyTorch LSTM、GridSearchCV、purged TimeSeriesSplit、嵌套OOF、Ridge Stacking和Git版本记录。')
    p(a,f'本次全部模型最高h=1 R²为{best.R2:.4f}。'+('未实现约0.8的性能目标，应将这一点作为结果限制，而不是用训练分数替代测试分数。' if best.R2<.8 else '已达到约0.8的性能目标。')+'预见期越长，可用历史对未来的解释力可能下降，但不强迫指标单调。')
    p(a,'当前限制包括四年训练数据较少、气象只截至t、有限网格和单随机种子。后续如有新的合法独立数据，可预先设计验证期，再研究更多历史窗口、洪峰加权损失或多种子稳定性；本次不能用2004结果继续选参。由于2004已在旧实验查看，本次属于冻结配置后的回溯评价。')
    p(a,'本实验说明，时间序列任务的日期可用性和评价协议与网络结构同样重要。应用于防洪等实际场景时，应同时检查平均误差、洪峰漏报和预报时效，不能仅依据一个R²宣布可部署。个人心得及成员协作体会待三位成员按实际参与补充。')
    a=originals[31]
    originals[30].text='附录 代码提交说明'
    p(a,'本报告不粘贴源码。完整实现、自动测试、依赖清单、运行说明和配置/数据哈希通过同目录“机器学习课程实践代码.zip”提交。唯一运行入口：python code/run_all.py；验证入口：python -m pytest tests -q。')
    p(a,'复现时需将教师原CSV置于 Teacher Resources & Requirements/01047000.csv。数据不随代码包重复分发，SHA-256：'+manifest['data_sha256']+'。')
    p(a,'运行标识：'+run.name+'；训练源码提交：'+manifest['git_commit']+'。正式运行清单和冻结参数随阶段验收附件提供。整体提交压缩包需待真实组号、组长及两位组员姓名确认后按教师规则命名。')
    for i in (7,10,13,16,19,22,25,26,27,28,31):
        el=originals[i]._element;el.getparent().remove(el)
    doc.core_properties.title='机器学习课程实践 径流预测实验报告'
    doc.core_properties.author=''
    doc.save(destination)
    (Path(__file__).with_name('report_content.md')).write_text('\n\n'.join(paragraphs),encoding='utf-8')


def build_package(run, destination):
    save_json(ROOT/'config_manifest.json',dict(data_path='Teacher Resources & Requirements/01047000.csv',
        data_sha256=digest(DATA_PATH),run=run.name,protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8')),
        source_hashes={p.name:digest(p) for p in (run/'source').glob('*.py')}))
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('code','tests'):
            for p in (ROOT/folder).glob('*.py'):z.write(p,p.relative_to(ROOT))
        for name in ('README.md','requirements.txt','config_manifest.json'):z.write(ROOT/name,name)
        for pattern in ('*.py','*.ps1'):
            for p in (ROOT/'report code').rglob(pattern):z.write(p,p.relative_to(ROOT))
    check_code_zip(destination)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    run=args.run.resolve()
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='completed':raise ValueError('正式运行尚未完成，不生成报告')
    # 分析图由训练环境先生成，文档构建只读取现有结果。
    deliveries=[]
    for n in (1,2,3):deliveries.append(build_acceptance(run,n))
    out=ROOT/'最终提交';out.mkdir(exist_ok=True)
    build_word(run,out/'机器学习课程实践实验报告.docx')
    build_package(run,out/'机器学习课程实践代码.zip')
    save_json(out/'交付核验.json',dict(status='generated_pending_visual_review',reports=[str(p) for p,_ in deliveries],
        relative_links_checked=sum(n for _,n in deliveries),base_models=24,ensemble_models=16,
        personal_fields='待填写',h1_target_met=manifest['best_h1_R2']>=.8))
    print('GENERATED',out,deliveries)


if __name__=='__main__':main()
