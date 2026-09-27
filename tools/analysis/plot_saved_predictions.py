"""Plot descriptive paired recipe effects from the verified saved-prediction ledger."""
import os

from tools.experiment_b.common import ROOT, formats, unseal


def main():
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    analysis=unseal(ROOT/'results/summaries/saved-prediction-study-v1/analysis.json')
    rows=formats()
    models=['resnet18','mobilenet_v2','mobilenet_v3_large','yolov8n']
    effects={(r['model'],r['format']):r['difference_pp'] for r in analysis['paired_recipe_comparisons']}
    values=np.asarray([[effects[(model,row['name'])] for model in models] for row in rows])
    figure,axes=plt.subplots(figsize=(9,11))
    limit=float(np.abs(values).max())
    image=axes.imshow(values,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    axes.set_xticks(range(4),['ResNet18\nTop-1','MobileNetV2\nTop-1','MobileNetV3\nTop-1','YOLOv8n\nmAP50–95'])
    axes.xaxis.tick_top()
    axes.set_yticks(range(len(rows)),[r['name'] for r in rows])
    for y in range(len(rows)):
        for x in range(4):
            v=values[y,x]
            axes.text(x,y,f'{v:+.1f}',ha='center',va='center',fontsize=9,color='white' if abs(v)>limit*.55 else '#18202b')
        if y and rows[y]['family']!=rows[y-1]['family']:
            axes.axhline(y-.5,color='white',linewidth=2)
    axes.tick_params(length=0,labelsize=10,pad=8)
    axes.set_title('Recipe sensitivity on the saved development panel',fontsize=15,pad=47,loc='left')
    colorbar=figure.colorbar(image,ax=axes,fraction=.045,pad=.045)
    colorbar.set_label('Percentile 99.9 − maxabs (percentage points)',fontsize=10)
    figure.text(.03,.035,'128 shared images per configuration • FP32-QDQ exploration\n'
                'Positive values favor percentile; negative values favor maxabs.\n'
                'Descriptive development results; not exact-A quality or final benchmark rankings.',fontsize=9,color='#384455')
    figure.subplots_adjust(left=.24,right=.91,top=.87,bottom=.12)
    directory=ROOT/'results/figures';directory.mkdir(parents=True,exist_ok=True)
    for extension in ('png','svg'):
        path=directory/f'saved-predictions-recipe-effect-v1.{extension}'
        figure.savefig(path,dpi=180,facecolor='white')
        print(path)
    plt.close(figure)
    # A difference alone hides whether either recipe preserves useful quality.
    indexed={(r['model'],r['format'],r['recipe']):r for r in analysis['b_configurations']}
    absolute=np.asarray([[[indexed[(model,row['name'],recipe)]['candidate_percent'] for model in models]
                          for row in rows] for recipe in ('maxabs','percentile_99_9')])
    fig,axs=plt.subplots(1,2,figsize=(13,11),sharey=True)
    for recipe_index,(ax,title) in enumerate(zip(axs,['Maxabs','Percentile 99.9'])):
        im=ax.imshow(absolute[recipe_index],cmap='viridis',vmin=0,vmax=100,aspect='auto')
        ax.set_xticks(range(4),['ResNet18\nTop-1 %','MobileNetV2\nTop-1 %','MobileNetV3\nTop-1 %','YOLOv8n\nmAP points'])
        ax.xaxis.tick_top();ax.set_yticks(range(len(rows)),[r['name'] for r in rows])
        ax.set_title(title,fontsize=14,pad=44)
        ax.tick_params(length=0,labelsize=8,pad=7)
        for y in range(len(rows)):
            for x in range(4):
                value=absolute[recipe_index,y,x]
                ax.text(x,y,f'{value:.1f}',ha='center',va='center',fontsize=8,color='white' if value<60 else '#18202b')
            if y and rows[y]['family']!=rows[y-1]['family']:
                ax.axhline(y-.5,color='white',linewidth=1.5)
    fig.suptitle('Actual quality scores for both recipes',fontsize=16,y=.975)
    fp32=[indexed[(model,'int8','maxabs')]['baseline_percent'] for model in models]
    fig.text(.025,.04,'FP32 on the same panel: '+ ' | '.join(f'{model}: {value:.1f}' for model,value in zip(models,fp32))+
             '\n128 development images • FP32-QDQ exploration • Higher scores are better within each model/metric.'+
             '\nThe difference chart equals the right-hand score minus the left-hand score, in percentage points.',fontsize=9)
    fig.subplots_adjust(left=.15,right=.94,top=.85,bottom=.12,wspace=.10)
    fig.colorbar(im,ax=axs.tolist(),fraction=.02,pad=.02,label='Absolute quality score (0–100)')
    for extension in ('png','svg'):
        path=directory/f'saved-predictions-absolute-quality-v1.{extension}'
        fig.savefig(path,dpi=180,facecolor='white');print(path)
    plt.close(fig)


if __name__=='__main__':
    main()
