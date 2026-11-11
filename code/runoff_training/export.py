"""将核验过的正式结果导出为轻量实验记录，保留旧验收Word不变。"""
import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from acceptance3_train import ROOT
from runoff_training.verify import verify_run, compare_to_delivery


def markdown_table(frame):
    def text(value):
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value).replace('|', '/')
    lines = ['| ' + ' | '.join(frame.columns) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    lines += ['| ' + ' | '.join(text(v) for v in row) + ' |'
              for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ['.', '..']:
        parser.error('run-id必须为单目录名')
    run = ROOT/'output/acceptance3_runs'/args.run_id
    result, frame, metrics = verify_run(run)
    compared, primary = compare_to_delivery(frame, metrics, ROOT/'验收/Acceptance 3')
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    for name in ['manifest.json', 'verification.json', 'config.json', 'lstm_grid_selection.json',
                 'development_gates.json', 'frozen_config.json', 'meta_selection.json',
                 'metrics.csv', 'metrics_all_seeds.csv', 'model_hashes_before_test.json']:
        shutil.copy2(run/name, out/name)
    (out/'independent_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    primary.to_csv(out/'comparison_primary_summary.csv', encoding='utf-8-sig')
    compared.to_csv(out/'comparison_by_horizon.csv', index=False, encoding='utf-8-sig')
    for name in ['cv_results', 'splits']:
        shutil.copytree(run/name, out/name)
    by_seed = compared.groupby(['strategy', 'model', 'seed'])[
        ['RMSE', 'RMSE_previous', 'MAE', 'MAE_previous']].mean().reset_index()
    by_seed['RMSE_change_pct'] = (by_seed.RMSE / by_seed.RMSE_previous - 1) * 100
    by_seed.to_csv(out/'comparison_by_seed.csv', index=False, encoding='utf-8-sig')
    gates = json.loads((run/'development_gates.json').read_text(encoding='utf-8'))
    selections = pd.DataFrame([dict(策略=g['strategy'], h=g['h'], units=str(g['params']['units']),
        epoch=g['params']['fixed_epochs'], 输入=g['params']['feature_mode'],
        输入变换=g['params']['input_transform'], 残差=g['params']['residual'],
        开发准入=g['passed'], 开发RMSE比值=g['RMSE_ratio'], 最差季度比值=g['worst_quarter_ratio']) for g in gates])
    selections.to_csv(out/'lstm_parameters_and_gates.csv', index=False, encoding='utf-8-sig')
    required = metrics[(metrics.seed == 42) & metrics.model.isin(['LSTM', 'Mean', 'Stacking'])]
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    avg = by_seed.groupby(['strategy', 'model'])[['RMSE', 'RMSE_previous', 'MAE', 'MAE_previous']].mean().reset_index()
    avg['RMSE_change_pct'] = (avg.RMSE / avg.RMSE_previous - 1) * 100
    lines = ['# 任务1至7+GPU改进训练结果', '', '2026-11-11｜独立实验记录，不替换现有验收Word。', '',
        '## 结论', '',
        '本轮训练完成，但只取得部分改善。直接LSTM与直接简单平均的七步平均RMSE在三seed均低于原报告；直接Stacking在三seed均退化。多输出Stacking只有主seed42略好，其他seed更差，不能宣称融合稳定提升。', '',
        '**未通过预设开发准入的配置不推荐直接替换原正式交付。** 八项LSTM配置只有direct h1通过；direct h2～7及multi均未通过。2004分数变好不能覆盖2003季度稳定性失败，也不能按2004逐h拼接最优模型。', '',
        f"运行 `{run.name}`，设备 `{manifest['gpu']}`，耗时 {manifest['seconds']/60:.2f} 分钟；872次基模型fit、三seed42/43/44、359个共同测试起报点。2004已在此前查看，本轮为回溯评估，不是盲测。", '',
        '## 主seed42：七步指标算术平均', '', markdown_table(primary.reset_index()), '',
        'RMSE_change_pct为相对原报告的变化，负值表示降低。R²/RMSE/MAE是逐h指标算术平均，不是把全部步长合并后重新计算的指标。', '',
        '## 三seed平均（先平均七步，再平均seed）', '', markdown_table(avg), '',
        '## 逐seed稳定性', '', markdown_table(by_seed[by_seed.model.isin(['LSTM', 'Mean', 'Stacking'])]), '',
        '## 网格胜出参数与2003开发准入', '', markdown_table(selections), '',
        '参数只由不晚于2002的GridSearchCV选择；2003门槛只决定是否推荐替换交付。本次实验没有用2003全年结果反选之前季度的OOF配置。', '',
        '## 任务2～7完整逐天主结果', '',
        '所有R²为1−SSE/SST，允许为负。RMSE/MAE为原始Discharge尺度，不擅自补未经确认的物理单位。Stacking的负预测如实保留。', '',
        markdown_table(required.drop(columns='seed')), '',
        '## 比较限制与解释', '',
        '- 直接ANN/RF采用固定历史参照参数，本轮没有重新网格搜索；它们与旧直接策略的参数可能不同。因此集成的变化不能全部归因于LSTM。',
        '- 本轮Stacking只用2003年前向OOF，旧多输出使用907条、旧直接使用723条OOF。不同训练协议限制严格归因；更少且季节分布不同的OOF可能影响元层稳定性，但本次实验没有证明它就是退化原因。',
        '- 本轮符号对数与残差组合是整体配置对照；batch也有区别，不构成纯单因素消融。',
        '- 多输出主seed42 LSTM/Mean/Stacking虽然RMSE下降，但MAE均变差；多输出LSTM h6/h7的R²为负，不应只报有利平均值。',
        '- 主seed42直接Mean的七步平均RMSE为746.7582，低于旧multi RF的757.0957；这是不同策略的回溯比较，不能据此证明可推广的部署提升。',
        '- 本轮未修改原验收Word、图片、截图或教师要求；此记录不能当作已更新的完整验收包。', '',
        '## 已核验与产物', '',
        '新增5项自动测试、历史25项测试均通过；独立核验从75390行预测重建210行指标，重建简单平均和Stacking；核对新旧目标日期和真值、872份训练记录、840个冻结文件，未发现不一致。', '',
        '- [主seed42完整指标](metrics.csv)',
        '- [三seed完整指标](metrics_all_seeds.csv)',
        '- [逐h新旧对照](comparison_by_horizon.csv)',
        '- [逐seed对照](comparison_by_seed.csv)',
        '- [参数与开发门槛](lstm_parameters_and_gates.csv)',
        '- [独立核验](independent_verification.json)',
        '- [运行来源](manifest.json)',
        '- [冻结配置](frozen_config.json)',
        '- [训练方案与任务要求](../../任务1至7+训练方案与要求_2026-11-11.md)', '',
        '完整权重、逐日预测、历史和日志保留在本地独立run中（约631MB），未复制到此轻量记录，也未上传远程。', '']
    (out/'任务1至7+GPU改进训练结果_2026-11-11.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
