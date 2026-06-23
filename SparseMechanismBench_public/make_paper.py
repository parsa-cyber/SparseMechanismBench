from pathlib import Path
import pandas as pd, numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

root=Path(__file__).resolve().parent; results=root/'results'; figs=root/'figures'; out=root/'sparsity_is_not_enough_mechanism_study.pdf'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Small', parent=styles['BodyText'], fontSize=8, leading=10))
styles.add(ParagraphStyle(name='Tiny', parent=styles['BodyText'], fontSize=7, leading=8))
styles.add(ParagraphStyle(name='Caption', parent=styles['BodyText'], fontSize=8, leading=10, alignment=1, italic=True))
styles.add(ParagraphStyle(name='Eq', parent=styles['BodyText'], fontSize=12, leading=16, alignment=1, spaceBefore=6, spaceAfter=6))
styles['Title'].fontSize=18; styles['Title'].leading=22

def pct(x, sd=None):
    if pd.isna(x): return 'NA'
    if sd is None: return f'{100*x:.1f}%'
    return f'{100*x:.1f}% ± {100*sd:.1f}%'

def tbl(data, col_widths=None, font=7):
    t=Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#d9eaf7')),
        ('GRID',(0,0),(-1,-1),0.25,colors.grey),
        ('FONT',(0,0),(-1,0),'Helvetica-Bold',font),
        ('FONT',(0,1),(-1,-1),'Helvetica',font),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ALIGN',(1,1),(-1,-1),'CENTER'),
        ('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2),
    ]))
    return t

def add_fig(story, filename, caption, width=6.2):
    p=figs/filename
    if p.exists():
        
        from PIL import Image as PILImage
        im=PILImage.open(p); aspect=im.height/im.width; h=width*inch*aspect
        maxh=5.2*inch
        if h>maxh:
            h=maxh; w=maxh/aspect
        else:
            w=width*inch
        story.append(Image(str(p), width=w, height=h)); story.append(Paragraph(caption, styles['Caption'])); story.append(Spacer(1,8))

def heading(text, level=1):
    return Paragraph(text, styles[f'Heading{level}'])

story=[]
story.append(Paragraph('Sparsity Is Not Enough: Disentangling Local Plasticity, Task Discrimination, and Forgetting in Biologically Inspired Learning Rules', styles['Title']))
story.append(Paragraph('Parsa Fatehi / Langley High School', styles['Small'])); story.append(Spacer(1,8))

abstract='''This empirical study separates representational sparsity from task-discriminative learning in biologically inspired local plasticity. Prior results showed that backpropagation-trained and supervised models outperform simple Hebbian/Oja feature learners on digits, MNIST, and Fashion-MNIST, while Sparse Oja produces high activation sparsity. I extended the project with a mechanism-isolation experiment: a LeNet-style standard CNN baseline, a Competitive/Homeostatic Sparse Oja variant with adaptive firing-rate thresholds, a sparsity sweep over active fractions from 5% to 100%, and a replay sweep over 0%, 1%, 5%, 10%, and 20% replay on fixed mechanism-test subsets. The Standard CNN reached 97.4% ± 0.2% on MNIST and 83.7% ± 0.5% on Fashion-MNIST, removing the weak-CNN concern from the earlier paper. Homeostatic Sparse Oja preserved high sparsity but did not close the accuracy gap: 16.8% ± 2.2% on MNIST and 24.7% ± 12.4% on Fashion-MNIST. The sparsity sweep showed a clear accuracy-sparsity trade-off: extreme sparsity produced weak task discrimination, while less sparse Oja features improved accuracy but sacrificed biological-style sparsity. Replay reduced forgetting in some subset conditions but did not fully stabilize local-learning models. These results support the central claim that sparsity alone is not enough; biologically plausible efficiency likely requires interaction among local plasticity, competition, homeostasis, replay, feedback, and task-relevant signals.'''
story.append(heading('Abstract',2)); story.append(Paragraph(abstract, styles['BodyText']))

story.append(heading('Scientific Contribution',2))
contrib='''This study makes three contributions. First, it separates representational sparsity from task-discriminative accuracy by measuring both across local and gradient-based learning systems. Second, it evaluates whether sparse local plasticity improves sample efficiency and continual-learning stability across digits, MNIST, and Fashion-MNIST. Third, it tests whether biologically inspired stabilizing mechanisms such as competition, adaptive homeostasis, and replay improve the accuracy-sparsity-forgetting trade-off. The main result is a strengthened negative result: sparse local plasticity creates sparse representations, but by itself it does not create robust task-discriminative or stable representations.'''
story.append(Paragraph(contrib, styles['BodyText']))

story.append(heading('Research Question and Hypotheses',2))
story.append(Paragraph('Central question: which biologically inspired mechanisms - local plasticity, sparse coding, competition, homeostasis, replay, and task-relevant feedback - are sufficient or insufficient for closing the gap between local learning and gradient-based optimization?', styles['BodyText']))
for h in ['H1: Supervised/backpropagation-based models will outperform simple Hebbian/Oja feature learners in accuracy and sample efficiency.', 'H2: Sparse Oja-style models will produce substantially higher activation sparsity than dense MLPs.', 'H3: Competition and homeostasis may improve the accuracy-sparsity trade-off relative to fixed Sparse Oja.', 'H4: Replay may reduce raw forgetting, but stability is meaningful only when Task 1 is first learned well.']:
    story.append(Paragraph('• '+h, styles['BodyText']))

story.append(heading('Mathematical Setup',2))
story.append(Paragraph('Hebbian local plasticity updates a synapse using pre- and postsynaptic activity:', styles['BodyText']))
story.append(Paragraph('Δw<sub>ij</sub> = η x<sub>j</sub> y<sub>i</sub>', styles['Eq']))
story.append(Paragraph('Oja normalization stabilizes weight growth:', styles['BodyText']))
story.append(Paragraph('Δw<sub>i</sub> = η (y<sub>i</sub>x - y<sub>i</sub><sup>2</sup>w<sub>i</sub>)', styles['Eq']))
story.append(Paragraph('Backpropagation updates weights by descending the loss gradient:', styles['BodyText']))
story.append(Paragraph('w ← w - η ∂L/∂w', styles['Eq']))
story.append(Paragraph('Forgetting and normalized forgetting are:', styles['BodyText']))
story.append(Paragraph('F = A<sub>before</sub> - A<sub>after</sub>,     F<sub>norm</sub> = (A<sub>before</sub> - A<sub>after</sub>) / A<sub>before</sub>', styles['Eq']))

story.append(heading('Methods',2))
story.append(Paragraph('Datasets: scikit-learn digits, MNIST IDX, and Fashion-MNIST IDX. Pixels were normalized to [0,1]. Seeds 0-4 were used for completed full benchmark runs. No test-set tuning was performed. Mechanism sweeps use fixed 10,000-train/2,000-test or fixed 10,000-train/full-test subsets and are explicitly labeled as mechanism-isolation analyses, not state-of-the-art benchmarks.', styles['BodyText']))

# validation table
val=pd.read_csv(results/'dataset_validation.csv')
data=[['Dataset','Train','Test','Shape','Train min-max','Test min-max']]
for _,r in val.iterrows(): data.append([r.dataset, str(r.train_size), str(r.test_size), r.image_shape, f"{r.train_min:.1f}-{r.train_max:.1f}", f"{r.test_min:.1f}-{r.test_max:.1f}"])
story.append(tbl(data, [1.2*inch,.8*inch,.8*inch,1*inch,1.2*inch,1.2*inch])); story.append(Paragraph('Table 1. Dataset validation. Source: dataset_validation.csv.', styles['Caption']))
add_fig(story,'example_images_grid.png','Figure 1. Example images from digits, MNIST, and Fashion-MNIST. Source: validate_datasets.py.',width=6.2)

# model mechanism table
mech_data=[['Model','Local?','Sparse?','Homeostatic?','Replay?','Task-directed signal'],['Oja','Yes','No','No','Optional','No'],['Sparse Oja','Yes','Yes','No','Optional','No'],['Homeostatic Sparse Oja','Yes','Yes','Yes','Optional','No'],['MLP','No','No','No','Optional','Yes'],['Standard CNN','No','No','No','Optional','Yes']]
story.append(tbl(mech_data, [1.5*inch,.7*inch,.7*inch,.9*inch,.7*inch,1.4*inch])); story.append(Paragraph('Table 2. Mechanism ladder tested in the revised experiment.', styles['Caption']))

story.append(PageBreak())
story.append(heading('Results',2))
cls=pd.read_csv(results/'updated_classification_summary.csv')
models=['logistic_regression','mlp','mlp_dropout','compact_cnn','standard_cnn','hebbian_oja','sparse_hebbian_oja','homeostatic_sparse_oja']
labels={'logistic_regression':'LogReg','mlp':'MLP','mlp_dropout':'MLP+dropout','compact_cnn':'compact CNN','standard_cnn':'Standard CNN','hebbian_oja':'Oja','sparse_hebbian_oja':'Sparse Oja','homeostatic_sparse_oja':'Homeostatic Sparse Oja'}
data=[['Model','digits','MNIST','Fashion-MNIST']]
for m in models:
    row=[labels[m]]
    for d in ['digits','mnist','fashion_mnist']:
        r=cls[(cls.dataset==d)&(cls.model==m)]
        row.append(pct(r.accuracy_mean.iloc[0], r.accuracy_std.iloc[0]) if len(r) else 'NA')
    data.append(row)
story.append(tbl(data, [1.8*inch,1.3*inch,1.3*inch,1.5*inch], font=7)); story.append(Paragraph('Table 3. Updated classification accuracy. Source: updated_classification_summary.csv.', styles['Caption']))
add_fig(story,'standard_vs_compact_cnn.png','Figure 2. Standard CNN removes the weak-CNN concern: it strongly outperforms the compact CNN on MNIST/Fashion-MNIST. Source: standard_cnn_by_seed.csv and updated_classification_summary.csv.',width=5.8)
add_fig(story,'mechanism_ladder_accuracy.png','Figure 3. Mechanism ladder classification results. Homeostatic Sparse Oja improves over fixed Sparse Oja on digits and Fashion-MNIST but remains far below supervised baselines. Source: updated_classification_summary.csv.',width=6.2)

story.append(heading('Sparsity Sweep: Accuracy vs. Sparsity',3))
story.append(Paragraph('To test whether Sparse Oja failed simply because 90% sparsity was too extreme, I swept active fractions of 5%, 10%, 20%, 40%, 60%, and 100% on fixed 10,000-train/2,000-test subsets. This is a mechanism-isolation experiment, not the main full-data benchmark.', styles['BodyText']))
sw=pd.read_csv(results/'sparsity_sweep_subset_summary.csv')
data=[['Dataset','Active %','Sparsity','Accuracy']]
for _,r in sw.sort_values(['dataset','active_frac']).iterrows(): data.append([r.dataset.replace('_','-'),f"{int(r.active_frac*100)}%",pct(r.sparsity_mean,r.sparsity_std),pct(r.accuracy_mean,r.accuracy_std)])
story.append(tbl(data, [1.5*inch,.8*inch,1.5*inch,1.5*inch], font=7)); story.append(Paragraph('Table 4. Sparsity sweep on fixed subset. Source: sparsity_sweep_subset_summary.csv.', styles['Caption']))
add_fig(story,'ablation_accuracy_vs_sparsity.png','Figure 4. Sparsity sweep: extreme sparsity weakens task discrimination; accuracy improves as more units are active. Source: sparsity_sweep_subset_summary.csv.',width=6.2)

story.append(heading('Replay Sweep',3))
story.append(Paragraph('Replay was tested on fixed 10k-train/full-test subsets for Oja, Sparse Oja, and Homeostatic Sparse Oja. The goal was not to claim state-of-the-art continual learning, but to test whether replay changes the stability trade-off.', styles['BodyText']))
rp=pd.read_csv(results/'replay_sweep_subset_summary.csv')
# condensed replay table for MNIST/Fashion: replay 0 and 20 only to save space
data=[['Dataset','Model','Replay','A_before','A_after','A_T2','Norm F']]
for _,r in rp[rp.replay_frac.isin([0.0,0.2])].sort_values(['dataset','model','replay_frac']).iterrows(): data.append([r.dataset.replace('_','-'),r.model, f"{int(r.replay_frac*100)}%", pct(r.A_before_mean,r.A_before_std), pct(r.A_after_mean,r.A_after_std), pct(r.A_T2_mean,r.A_T2_std), f"{r.normalized_forgetting_mean:.2f} ± {r.normalized_forgetting_std:.2f}"])
story.append(tbl(data, [1.0*inch,1.5*inch,.7*inch,1.0*inch,1.0*inch,1.0*inch,1.0*inch], font=6.5)); story.append(Paragraph('Table 5. Replay sweep subset results, condensed to 0% and 20% replay. Full 0/1/5/10/20% results are in replay_sweep_subset_summary.csv.', styles['Caption']))
add_fig(story,'replay_sweep_partial_mnist.png','Figure 5. Replay sweep on MNIST subset. Source: replay_sweep_subset_summary.csv.',width=5.8)

story.append(heading('Statistical Analysis',3))
stats_df=pd.read_csv(results/'statistical_tests.csv')
data=[['Dataset','Comparison','Mean diff','Wilcoxon p','Effect size dz']]
for _,r in stats_df.iterrows():
    comp=f"{r.model_a} vs {r.model_b}"; data.append([r.dataset.replace('_','-'),comp,f"{r.mean_diff_a_minus_b:.3f}",f"{r.wilcoxon_p:.4f}",f"{r.cohens_dz:.2f}"])
story.append(tbl(data, [1.1*inch,2.2*inch,.8*inch,.8*inch,.9*inch], font=6.2)); story.append(Paragraph('Table 6. Paired statistical tests across seeds. With five seeds, p-values are supportive only; effect sizes and cross-dataset consistency are emphasized. Source: statistical_tests.csv.', styles['Caption']))
add_fig(story,'mechanism_map.png','Figure 6. Mechanism map: the project tests how local plasticity, sparse coding, homeostasis, replay, and task-directed optimization affect the trade-off. Source: conceptual synthesis.',width=6.2)

story.append(heading('Discussion',2))
story.append(Paragraph('The revised results strengthen the paper substantially. The Standard CNN baseline reaches realistic accuracy on MNIST and Fashion-MNIST, so the earlier weak compact-CNN limitation is no longer a major threat to the supervised baseline comparison. Supervised models still dominate task-discriminative accuracy. Homeostatic Sparse Oja improves over fixed Sparse Oja in some cases, especially digits and Fashion-MNIST, but it does not close the gap to MLPs or the Standard CNN. This supports the mechanism-level interpretation: sparse local plasticity creates sparse activity, but task discrimination requires additional structure or task-relevant signals.', styles['BodyText']))
story.append(Paragraph('The sparsity sweep is the most important new mechanism-isolation result. When active units are extremely limited, representations are sparse but weakly discriminative. As active fraction increases, accuracy improves while sparsity declines. This separates two properties that are often conflated: representational sparsity and useful class separation. The replay sweep also suggests that memory stabilization can reduce forgetting in some cases, but replay alone did not make local-feature models competitive. Overall, the data support a hybrid interpretation rather than a claim that Hebbian/Oja learning beats backpropagation.', styles['BodyText']))

story.append(heading('Limitations',2))
for l in ['The Standard CNN was trained for two epochs on MNIST/Fashion-MNIST because a two-epoch LeNet-style CNN already produced realistic baseline accuracy; it is not presented as state of the art.', 'Homeostatic Sparse Oja was tested with a simple adaptive threshold mechanism; more biologically detailed neuromodulation, recurrent circuitry, and reward modulation remain future work.', 'Sparsity and replay sweeps were run on fixed subsets for mechanism isolation, not as full-data benchmark claims.', 'Only five seeds were completed for the full benchmark results; the requested ten-seed extension remains a recommended next step.', 'Approximate MACs remain proxies, not direct energy measurements.', 'No test-set tuning was performed, but hyperparameters were not exhaustively optimized.']:
    story.append(Paragraph('• '+l, styles['BodyText']))

story.append(heading('Conclusion',2))
story.append(Paragraph('Across digits, MNIST, and Fashion-MNIST, supervised and backpropagation-based models remain stronger for classification accuracy and sample efficiency. Sparse Oja-style learning reliably creates sparse representations, but sparse local plasticity alone is insufficient for task-discriminative learning or stable continual learning. Adding homeostatic competition improves some trade-offs but does not close the gap to supervised optimization. The strongest scientific conclusion is therefore not that Hebbian/Oja learning is better than backpropagation, but that biological learning efficiency likely depends on interacting mechanisms: local plasticity, sparse coding, competition, homeostasis, replay, feedback, and task-relevant signals.', styles['BodyText']))

story.append(heading('Code and Data Availability',2))
story.append(Paragraph('The accompanying reproducibility package contains README.md, requirements.txt, data_loading.py, models.py, train_classification.py, train_hebbian_oja.py, train_sparse_oja.py, train_homeostatic_oja.py, run_sample_efficiency.py, run_continual_learning.py, run_all_experiments.py, plot_results.py, raw CSV files, summary CSV files, generated figures, and scripts. A public GitHub or OSF link should be added at submission time. All reported numerical results must be traceable to raw CSV files and plotting scripts.', styles['BodyText']))

story.append(heading('One-Page Project Summary',2))
story.append(Paragraph('Problem: Biological brains learn efficiently with local plasticity, while artificial networks rely on global gradients. Hypothesis: sparsity, homeostasis, and replay may improve local-learning trade-offs. Methods: compare supervised models, Oja, Sparse Oja, Homeostatic Sparse Oja, sparsity sweeps, replay sweeps, and statistical tests across digits, MNIST, and Fashion-MNIST. Results: supervised models dominate accuracy; sparse local learning produces sparsity but weak discrimination; homeostasis helps only partially; replay reduces forgetting in some subset conditions. Significance: the work isolates why sparsity alone is not enough and motivates hybrid mechanisms.', styles['BodyText']))

story.append(heading('Preprint / STS Checklist',2))
for item in ['Remove application-specific language from public version.', 'Add public GitHub/OSF/Zenodo link.', 'Ensure all figures regenerate from CSVs.', 'Keep negative results.', 'Report no test-set tuning.', 'Use complete APA-style references.', 'Add mentor/teacher review before submission if available.']:
    story.append(Paragraph('• '+item, styles['BodyText']))

story.append(heading('150-Character Description',2)); story.append(Paragraph('Tested why sparse brain-like learning fails: local Oja rules create sparsity, but need homeostasis/replay/feedback for useful learning.', styles['BodyText']))
story.append(heading('350-Character Description',2)); story.append(Paragraph('Built a multi-dataset neural-learning study comparing backpropagation, Oja-style local plasticity, Sparse Oja, Homeostatic Sparse Oja, CNNs, and replay. Results show sparsity alone is not enough: local rules create sparse representations but weak task discrimination, motivating hybrid biologically inspired learning.', styles['BodyText']))

story.append(heading('Originality Statement',2)); story.append(Paragraph('This project is original because it does not simply compare Hebbian learning against backpropagation. It isolates mechanisms: sparsity, competition, homeostasis, and replay. The negative result is scientifically useful because it shows that representational sparsity and task-discriminative learning are separable properties.', styles['BodyText']))
story.append(heading('Reviewer-Response Memo',2)); story.append(Paragraph('The project was improved by adding a full reproducibility repository, replacing the weak compact CNN with a LeNet-style Standard CNN baseline, adding Homeostatic Sparse Oja, running a sparsity sweep to test whether extreme sparsity caused failure, adding a replay sweep, computing confidence intervals and paired tests, and rewriting the paper around the mechanism-level claim that sparsity alone is insufficient.', styles['BodyText']))

story.append(heading('References',2))
refs=['Hebb, D. O. (1949). The organization of behavior: A neuropsychological theory. Wiley.','Oja, E. (1982). A simplified neuron model as a principal component analyzer. Journal of Mathematical Biology, 15(3), 267-273. https://doi.org/10.1007/BF00275687','Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323, 533-536. https://doi.org/10.1038/323533a0','Bi, G.-Q., & Poo, M.-M. (1998). Synaptic modifications in cultured hippocampal neurons. Journal of Neuroscience, 18(24), 10464-10472. https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998','LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324. https://doi.org/10.1109/5.726791','Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J. (2016). Random synaptic feedback weights support error backpropagation for deep learning. Nature Communications, 7, 13276. https://doi.org/10.1038/ncomms13276','Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS, 114(13), 3521-3526. https://doi.org/10.1073/pnas.1611835114','Whittington, J. C. R., & Bogacz, R. (2019). Theories of error back-propagation in the brain. Trends in Cognitive Sciences, 23(3), 235-250. https://doi.org/10.1016/j.tics.2018.12.005','Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. arXiv:1708.07747.']
for ref in refs: story.append(Paragraph(ref, styles['Small']))

SimpleDocTemplate(str(out), pagesize=letter, rightMargin=.55*inch,leftMargin=.55*inch,topMargin=.55*inch,bottomMargin=.55*inch).build(story)
print('wrote',out)
