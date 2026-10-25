r"""Generate thesis figures from saved metrics only (no training or test inference).
Usage: E:\python\python.exe generate_thesis_figures.py
Dependencies: matplotlib, numpy
"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path(__file__).resolve().parent/'runs')
    parser.add_argument('--output',type=Path,default=Path(r'E:\study\论文\figures'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
        'axes.labelsize':10,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    sources=[]
    def load(relative):
        path=args.runs/relative;sources.append(str(path.resolve()));return read(path)
    def save(fig,name):
        for ext in ['pdf','png']:fig.savefig(args.output/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
        plt.close(fig)
    histories=[('CNN–LSTM','branch1_experiment/full'),('Original ViT','vit_recovery_extended/full'),('Original fusion','fusion_frozen')]
    fig,axes=plt.subplots(3,2,figsize=(7.2,8.1),layout='constrained')
    best_epochs={}
    for row,(name,folder) in enumerate(histories):
        history=load(folder+'/history.json');epochs=[e['epoch'] for e in history]
        assert epochs==list(range(1,len(history)+1)),f'Non-contiguous history: {folder}'
        best=max(history,key=lambda e:e['val']['macro_f1']);best_epochs[name]=best['epoch']
        for col,(metric,label) in enumerate([('accuracy','Accuracy (%)'),('macro_f1','Macro F1 (%)')]):
            ax=axes[row,col]
            for split,color,style in [('train','#0072B2','-'),('val','#D55E00','--')]:
                values=[100*e[split][metric] for e in history]
                ax.plot(epochs,values,color=color,linestyle=style,lw=1.6,label='Training' if split=='train' else 'Validation')
            ax.axvline(best['epoch'],color='#555555',ls=':',lw=1.1,label=f"Selected epoch {best['epoch']}")
            ax.scatter([best['epoch']],[best['val'][metric]*100],color='#D55E00',s=28,zorder=4)
            ax.set(title=name,xlabel='Epoch',ylabel=label,ylim=(0,100),xlim=(1,len(history)))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=6));ax.grid(alpha=.2);ax.legend(loc='lower right',framealpha=.9)
    save(fig,'training_curves')
    models=[('CNN–LSTM','branch1_experiment/test'),('Original ViT','vit_low_lr/test'),('Original fusion','fusion_frozen/test')]
    metrics=[load(folder+'/test_metrics.json') for _,folder in models]
    for m in metrics:
        cm=np.asarray(m['confusion_matrix']);assert cm.shape==(5,5) and cm.sum()==700 and np.all(cm.sum(1)==140)
        assert m['split_hash']==metrics[0]['split_hash'] and m['classes']==metrics[0]['classes']
    short=['Real','DF','F2F','FS','NT']
    fig,axes=plt.subplots(3,1,figsize=(6.9,8.9),layout='constrained')
    for ax,(name,_),m in zip(axes,models,metrics):
        cm=np.asarray(m['confusion_matrix']);percent=cm/cm.sum(1,keepdims=True)*100
        im=ax.imshow(percent,cmap='Blues',vmin=0,vmax=100,aspect='auto')
        for i in range(5):
            for j in range(5):ax.text(j,i,f'{cm[i,j]}\n({percent[i,j]:.1f}%)',ha='center',va='center',fontsize=8,color='white' if percent[i,j]>55 else '#222222')
        ax.set(xticks=range(5),yticks=range(5),xticklabels=short,yticklabels=short,xlabel='Predicted class',ylabel='True class',title=name)
        ax.tick_params(length=0)
    fig.colorbar(im,ax=axes,shrink=.75,pad=.025,label='Within-class percentage (%)')
    save(fig,'test_confusion_matrices')
    refined=load('vit_partial_refine/test/test_metrics.json');newfusion=load('fusion_refined_vit/test/test_metrics.json')
    comparison=[metrics[1],refined,metrics[2],newfusion]
    for m in comparison:assert m['samples']==700 and m['split_hash']==metrics[0]['split_hash']
    fig,ax=plt.subplots(figsize=(7.2,3.6),layout='constrained');x=np.arange(4);width=.35
    for offset,metric,label,color in [(-width/2,'accuracy','Accuracy','#0072B2'),(width/2,'macro_f1','Macro F1','#D55E00')]:
        bars=ax.bar(x+offset,[100*m[metric] for m in comparison],width,label=label,color=color)
        ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
    ax.set(xticks=x,xticklabels=['Original\nViT','Refined\nViT','Original fusion\n(retained)','Fusion with\nrefined ViT'],ylabel='Test performance (%)',ylim=(0,100))
    ax.legend(loc='upper left',ncol=2);ax.yaxis.grid(alpha=.2);ax.set_axisbelow(True)
    save(fig,'refinement_comparison')
    manifest={'sources':sources,'selected_epochs':best_epochs,'notes':'Original models used for training curves and confusion matrices. Refinements only in comparison chart. No per-epoch test evaluation.'}
    (args.output/'figure_sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'selected_epochs':best_epochs,'figures':3},ensure_ascii=True))
if __name__=='__main__':main()
