"""数据概览和过程/结果图，全部由本次数据和正式结果生成。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from features import build_features

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'figure.dpi':120,'savefig.dpi':160})


def field_table():
    return pd.DataFrame([
        ('Date','日期','年-月-日','教师 CSV 日期列'),
        ('Discharge','径流流量','原始流量单位（教师资料未明确，未换算）','预测目标'),
        ('Dayl','日长','s/day','太阳位于地平线以上的时长，不等同实测日照时数'),
        ('Prcp','降水量','mm/day','各种形态降水的日总水当量'),
        ('Srad','短波辐射','W/m²','日间平均入射短波辐射通量密度'),
        ('Swe','雪水当量','kg/m²','积雪所含水量；本站五年全为零'),
        ('Tmax','日最高气温','℃','日最高近地面气温'),
        ('Tmin','日最低气温','℃','日最低近地面气温'),
        ('Vp','水汽压','Pa','日平均近地面水汽分压')],columns=['field','中文含义','单位','说明'])


def correlation_frame(raw):
    frame = raw.drop(columns=['Swe']).copy()
    for col in ('Discharge','Prcp','Tmax','Tmin','Vp','Srad','Dayl'):
        for lag in (1,3,7):
            frame[f'{col}_lag{lag}'] = raw[col].shift(lag)
    return frame


def savefig(fig,path):
    fig.tight_layout()
    fig.savefig(path,bbox_inches='tight')
    plt.close(fig)


def make_analysis(raw, ds, run):
    out = run/'analysis'
    out.mkdir(exist_ok=True)
    field_table().to_csv(out/'fields.csv',index=False,encoding='utf-8-sig')
    raw.describe().T.assign(missing=raw.isna().sum(),skew=raw.skew()).to_csv(out/'statistics.csv',encoding='utf-8-sig')
    for mode in ('timeseries','histogram','boxplot'):
        fig,axes = plt.subplots(4,2,figsize=(12,10))
        for col,ax in zip(raw.columns,axes.ravel()):
            if mode == 'timeseries':
                ax.plot(raw.index,raw[col],lw=.6)
                ax.axvline(pd.Timestamp('2004-01-01'),color='red',ls='--',lw=.8)
            elif mode == 'histogram':
                ax.hist(raw[col],bins=40,color='#527ba0')
            else:
                ax.boxplot(raw[col],vert=False)
            ax.set_title(col)
        savefig(fig,out/f'{mode}.png')
    corr = correlation_frame(raw).corr()
    corr.to_csv(out/'pearson.csv',encoding='utf-8-sig')
    fig,ax = plt.subplots(figsize=(14,12))
    im = ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(len(corr)),corr.columns,rotation=90,fontsize=7)
    ax.set_yticks(range(len(corr)),corr.index,fontsize=7)
    fig.colorbar(im,ax=ax,label='Pearson r')
    savefig(fig,out/'pearson.png')
    ds['catalog'].to_csv(out/'feature_catalog.csv',index=False,encoding='utf-8-sig')
    # 过程展示只来自全训练期已拟合的模型，绝不据此重新选择参数。
    import joblib
    model = joblib.load(run/'models'/'direct_RF_h1.joblib')
    train = ds['train']
    selected = ds['X'][train,-1][:,model.selected_]
    scaled = model.x_scaler_.transform(selected)
    preview = pd.DataFrame(ds['features'].loc[ds['dates'][train]])
    preview.to_csv(out/'training_features.csv',encoding='utf-8-sig')
    pd.DataFrame(scaled,index=ds['dates'][train],columns=[ds['names'][i] for i in model.selected_]).to_csv(out/'standardized.csv',encoding='utf-8-sig')
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].hist(selected[:,0],bins=30); axes[0].set_title('训练特征标准化前')
    axes[1].hist(scaled[:,0],bins=30); axes[1].set_title('训练期拟合 Z-score 后')
    savefig(fig,out/'standardization.png')
    metrics = pd.read_csv(run/'metrics.csv')
    for strategy in ('direct','multi'):
        fig,axes = plt.subplots(1,3,figsize=(13,4))
        for model_name,group in metrics[metrics.strategy==strategy].groupby('model'):
            for metric,ax in zip(('R2','RMSE','MAE'),axes):
                ax.plot(group.h,group[metric],marker='o',label=model_name)
                ax.set(xlabel='预见期 h（日）',ylabel=metric)
        axes[0].legend(fontsize=8)
        fig.suptitle('直接预测' if strategy=='direct' else '原生多输出预测')
        savefig(fig,out/f'{strategy}_metrics.png')
    pred = pd.read_csv(run/'predictions.csv')
    for strategy in ('direct','multi'):
        fig,axes = plt.subplots(2,1,figsize=(12,7))
        for h,ax in zip((1,7),axes):
            subset=pred[(pred.strategy==strategy)&(pred.h==h)]
            actual=subset[subset.model=='RF']
            ax.plot(pd.to_datetime(actual.target_date),actual.truth,color='black',lw=1,label='实测')
            for family,group in subset.groupby('model'):
                ax.plot(pd.to_datetime(group.target_date),group.prediction,lw=.7,label=family,alpha=.8)
            ax.set_title(f'h={h}');ax.set_ylabel('径流（原始单位）')
        axes[0].legend(ncol=6,fontsize=8)
        savefig(fig,out/f'{strategy}_curves.png')
    best = metrics[metrics.h==1].sort_values('R2',ascending=False).iloc[0]
    errors=pred[(pred.h==1)&(pred.model==best.model)&(pred.strategy==best.strategy)].copy()
    errors['squared_error']=(errors.prediction-errors.truth)**2
    errors.sort_values('squared_error',ascending=False).head(20).to_csv(out/'largest_errors.csv',index=False,encoding='utf-8-sig')
