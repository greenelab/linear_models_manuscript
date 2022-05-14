---
title: The Effects of Nonlinear Signal on Expression-Based Prediction Performance
keywords:
- machine learning
- transcriptomics
- tissue prediction
- nonlinear models
lang: en-US
date-meta: '2022-05-14'
author-meta:
- Benjamin J. Heil
- Jake Crawford
- Casey S. Greene
header-includes: |-
  <!--
  Manubot generated metadata rendered from header-includes-template.html.
  Suggest improvements at https://github.com/manubot/manubot/blob/main/manubot/process/header-includes-template.html
  -->
  <meta name="dc.format" content="text/html" />
  <meta name="dc.title" content="The Effects of Nonlinear Signal on Expression-Based Prediction Performance" />
  <meta name="citation_title" content="The Effects of Nonlinear Signal on Expression-Based Prediction Performance" />
  <meta property="og:title" content="The Effects of Nonlinear Signal on Expression-Based Prediction Performance" />
  <meta property="twitter:title" content="The Effects of Nonlinear Signal on Expression-Based Prediction Performance" />
  <meta name="dc.date" content="2022-05-14" />
  <meta name="citation_publication_date" content="2022-05-14" />
  <meta name="dc.language" content="en-US" />
  <meta name="citation_language" content="en-US" />
  <meta name="dc.relation.ispartof" content="Manubot" />
  <meta name="dc.publisher" content="Manubot" />
  <meta name="citation_journal_title" content="Manubot" />
  <meta name="citation_technical_report_institution" content="Manubot" />
  <meta name="citation_author" content="Benjamin J. Heil" />
  <meta name="citation_author_institution" content="Genomics and Computational Biology Graduate Group, Perelman School of Medicine, University of Pennsylvania" />
  <meta name="citation_author_orcid" content="0000-0002-2811-1031" />
  <meta name="twitter:creator" content="@autobencoder" />
  <meta name="citation_author" content="Jake Crawford" />
  <meta name="citation_author_institution" content="Genomics and Computational Biology Graduate Group, Perelman School of Medicine, University of Pennsylvania" />
  <meta name="citation_author_orcid" content="0000-0001-6207-0782" />
  <meta name="twitter:creator" content="@jjc2718" />
  <meta name="citation_author" content="Casey S. Greene" />
  <meta name="citation_author_institution" content="Department of Pharmacology, University of Colorado School of Medicine" />
  <meta name="citation_author_institution" content="Department of Biochemistry and Molecular Genetics, University of Colorado School of Medicine" />
  <meta name="citation_author_orcid" content="0000-0001-8713-9213" />
  <meta name="twitter:creator" content="@greenescientist" />
  <link rel="canonical" href="https://greenelab.github.io/linear_models_manuscript/" />
  <meta property="og:url" content="https://greenelab.github.io/linear_models_manuscript/" />
  <meta property="twitter:url" content="https://greenelab.github.io/linear_models_manuscript/" />
  <meta name="citation_fulltext_html_url" content="https://greenelab.github.io/linear_models_manuscript/" />
  <meta name="citation_pdf_url" content="https://greenelab.github.io/linear_models_manuscript/manuscript.pdf" />
  <link rel="alternate" type="application/pdf" href="https://greenelab.github.io/linear_models_manuscript/manuscript.pdf" />
  <link rel="alternate" type="text/html" href="https://greenelab.github.io/linear_models_manuscript/v/60524839e6d285e6d50c03d0b1923e5d81585df6/" />
  <meta name="manubot_html_url_versioned" content="https://greenelab.github.io/linear_models_manuscript/v/60524839e6d285e6d50c03d0b1923e5d81585df6/" />
  <meta name="manubot_pdf_url_versioned" content="https://greenelab.github.io/linear_models_manuscript/v/60524839e6d285e6d50c03d0b1923e5d81585df6/manuscript.pdf" />
  <meta property="og:type" content="article" />
  <meta property="twitter:card" content="summary_large_image" />
  <link rel="icon" type="image/png" sizes="192x192" href="https://manubot.org/favicon-192x192.png" />
  <link rel="mask-icon" href="https://manubot.org/safari-pinned-tab.svg" color="#ad1457" />
  <meta name="theme-color" content="#ad1457" />
  <!-- end Manubot generated metadata -->
bibliography:
- content/manual-references.json
manubot-output-bibliography: output/references.json
manubot-output-citekeys: output/citations.tsv
manubot-requests-cache-path: ci/cache/requests-cache
manubot-clear-requests-cache: false
...






<small><em>
This manuscript
([permalink](https://greenelab.github.io/linear_models_manuscript/v/60524839e6d285e6d50c03d0b1923e5d81585df6/))
was automatically generated
from [greenelab/linear_models_manuscript@6052483](https://github.com/greenelab/linear_models_manuscript/tree/60524839e6d285e6d50c03d0b1923e5d81585df6)
on May 14, 2022.
</em></small>

## Authors



+ **Benjamin J. Heil**<br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0000-0002-2811-1031](https://orcid.org/0000-0002-2811-1031)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [ben-heil](https://github.com/ben-heil)
    · ![Twitter icon](images/twitter.svg){.inline_icon width=16 height=16}
    [autobencoder](https://twitter.com/autobencoder)<br>
  <small>
     Genomics and Computational Biology Graduate Group, Perelman School of Medicine, University of Pennsylvania
     · Funded by Grant XXXXXXXX
  </small>

+ **Jake Crawford**<br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0000-0001-6207-0782](https://orcid.org/0000-0001-6207-0782)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [jjc2718](https://github.com/jjc2718)
    · ![Twitter icon](images/twitter.svg){.inline_icon width=16 height=16}
    [jjc2718](https://twitter.com/jjc2718)<br>
  <small>
     Genomics and Computational Biology Graduate Group, Perelman School of Medicine, University of Pennsylvania
     · Funded by Grant XXXXXXXX
  </small>

+ **Casey S. Greene**<br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0000-0001-8713-9213](https://orcid.org/0000-0001-8713-9213)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [cgreene](https://github.com/cgreene)
    · ![Twitter icon](images/twitter.svg){.inline_icon width=16 height=16}
    [greenescientist](https://twitter.com/greenescientist)<br>
  <small>
     Department of Pharmacology, University of Colorado School of Medicine; Department of Biochemistry and Molecular Genetics, University of Colorado School of Medicine
     · Funded by Grant XXXXXXXX
  </small>



## Abstract {.page_break_before}

Within the field of transcriptomic prediction there are two conflicting theories.
The first argues that the complexity of predicting phenotypes makes the task well-suited for complex nonlinear models such as neural networks.
The second believes that simpler models are better, as they are easier to interpret and have similar performance for some tasks.
By comparing neural networks and logistic regression across multiple prediction tasks on GTEx and Recount3 datasets, we were able to show that both theories are valid.
We demonstrated the presence of nonlinear signal in transcriptomic prediction problems by removing the predictive linear signal with Limma.
However, we also found that the presence of nonlinear signal was not necessarily sufficient for neural networks to outperform logistic regression.
These results show that while neural networks may be useful for making predictions from gene expression data, including a linear baseline model is critical.


## Introduction

Transcriptomic data contains a wealth of information about biology.
Gene expression-based models are already being used for subtyping cancer [@doi:10.1200/JCO.2008.18.1370], predicting transplant rejections [@doi:10.1161/CIRCULATIONAHA.116.022907], and uncovering biases in public data [@pmc:PMC8011224].
In fact, both the capability of machine learning models [@arxiv:2202.05924] and the amount of transcriptomic data available [@doi:10.1038/s41467-018-03751-6; @doi:10.1093/database/baaa073] are increasing rapidly.
It makes sense, then, that neural networks are frequently being used in the transcriptomic prediction space [@doi:10.1038/s41598-019-52937-5; @doi:10.1093/gigascience/giab064; @doi:10.1371/journal.pcbi.1009433].

However, there are two conflicting ideas in the literature regarding the utility of nonlinear models.
One theory draws on prior biological understanding: the paths linking gene expression to phenotypes are complex [@doi:10.1016/j.semcdb.2011.12.004; @doi:10.1371/journal.pone.0153295], and nonlinear models like neural networks should be more capable of learning that complexity.
Unlike purely linear models such as logistic regression, nonlinear models should learn more sophisticated representations of the relationships between expression and phenotype.
Accordingly, many have used nonlinear models to learn representations useful for making predictions of phenotypes from gene expression [@doi:10.1128/mSystems.00025-15; @doi:10.1016/j.cmpb.2018.10.004; @doi:10.1186/s12859-017-1984-2].

The other theory disagrees with the first hypothesis.
When using expression to make predictions about phenotypes, linear models seem to do as well as or better than nonlinear ones in many cases [@doi:10.1186/s12859-020-3427-8].
While papers of this sort are harder to come by — scientists do not tend to write papers about how their deep learning model was worse than logistic regression — other complex biological problems have also seen linear models prove equivalent to nonlinear ones [@doi:10.1016/j.jclinepi.2019.02.004].

We demonstrate that both theories have merit.
There is nonlinear signal relating the phenotypes to genotypes, but the signal does not always lead nonlinear models to provide better predictive accuracy.
We construct a system of binary and multi-class classification problems on the GTEx and Recount3 compendia [@doi:10.1038/ng.2653;@doi:10.1186/s13059-021-02533-6] that shows linear and nonlinear models have similar accuracy on several prediction tasks.
We then remove the linear signals relating the phenotype to gene expression and show that there is nonlinear signal in the data even when the linear models outperform the nonlinear ones.
Finally, we validate the results by testing our models on a sex-metadata prediction problem, running controls on simulated data, and examining different problem formulations such as samplewise splitting and pretraining.

In reconciling these two ostensibly conflicting theories, we assist future scientists by showing the importance of trying a linear baseline model before developing a complex nonlinear approach.
While nonlinear models may outperform simpler models at the limit of infinite data, they do not necessarily do so even when trained on the largest datasets publicly available today.


## Results 

### Approach 
We compared the performance of linear and nonlinear models across multiple datasets and tasks (fig. @fig:workflow top).
Our datasets consisted of gene expression and tissue labels from GTEx [@doi:10.1038/ng.2653], expression from Recount3 [@doi:10.1186/s13059-021-02533-6] with tissue labels from the Recount3 metadata and sex labels from Flynn et al. [@doi:10.1186/s12859-021-04070-2], and simulated data.
Before use in model training, we removed scRNA samples, TPM normalized, and zero-one standardized the data.
To avoid leakage between cross-validation folds, we place entire studies into single folds instead of splitting them across folds (fig. @fig:workflow bottom).
We then ran the models on increasingly large training sets to determine how model performance is affected by the amount of training data.

![
Schematic of the model analysis workflow. We evaluate three models on multiple classification problems in three datasets (top). We use studywise splitting by default and evaluate the effects of samplewise splitting and pretraining.
](./images/workflow.svg "Workflow diagram"){#fig:workflow}


### Linear and nonlinear models have similar performance in many tasks
We selected expression data from GTEx [@doi:10.1038/ng.2653] to determine whether linear and nonlinear models performed similarly, as it is a relatively well understood dataset with consistent experimental design across samples.
We first trained our models to differentiate between tissue types on pairs of the five most common tissues in the dataset.
Likely due to the clean nature of the data, all models were able to perform perfectly on these binary classification tasks (fig. @fig:prediction_combined top right).

Because the binary classification task was too easy to determine any difference between models, evaluated the models on a more challenging task.
Namely, we tested the models on their ability to perform multiclass classification on all 31 tissues present in the dataset.
The multitask setting showed logistic regression slightly outperforming the five-layer neural network, which in turn slightly outperformed the three-layer net (fig. @fig:prediction_combined top left).

We then validated our findings in a separate dataset: Sequence Read Archive [@doi:10.1093/nar/gkq1019] samples from the Recount3 [@doi:10.1186/s13059-021-02533-6] dataset.
Again we compared the models' ability to differentiate between pairs of tissues (supp. fig. @fig:recount-binary) and found their performance was roughly equivalent.
We also evaluated the models' performance on a multiclass classification problem differentiating between the 21 most common tissues in the dataset.
As in the GTEx setting, the logistic regression model outperformed the five-layer network, which outperformed the three-layer network (fig. @fig:prediction_combined bottom left).

To examine whether these results held in a problem domain other than tissue type prediction, we used our models to predict metadata-derived sex labels (fig. @fig:prediction_combined bottom right), a task previously studied by Flynn et al. [@doi:10.1186/s12859-021-04070-2].
We used the same experimental setup as in our other binary prediction tasks to train the models, but rather than using tissue labels we used metadata-derived sex labels.
In this setting we found that while the models all performed similarly, the nonlinear models tended to have a slight edge over the linear one.

![
Performance of models across four classification tasks. In each panel the loess curve and its 95% confidence interval are plotted based on points from three seeds, ten data subsets, and five folds of studywise cross-validation (for a total of 150 points per model per panel).
](./images/full_signal_combined.svg ){#fig:prediction_combined}

### There is predictive nonlinear signal in biological problems
One can imagine a world where all the signal relevant to tissue prediction is linear.
If that were the case, nonlinear models like neural networks would fail to give any advantage in a prediction problem.
To determine whether there is nonlinear signal in our tissue prediction tasks learnable by our neural nets, we used Limma [@doi:10.1093/nar/gkv007] to remove the linear signal associated with each tissue.

We began by simulating three datasets to better understand model performance for a variety of linear or nonlinear data generating processes.
Our initial dataset simulated both linear and nonlinear signal by generating two types of features: half of the features with a linear dividing line between the simulated classes and half with a nonlinear dividing line (see [Methods](#methods) for more detail).
After training to classify the simulated dataset, all models effectively predicted the simulated classes.
After removing the linear signal from the dataset, nonlinear models remained able to predict the correct classes, but logistic regression was no better than random (fig @fig:simulation middle).

To measure the models' performance in data with only linear signal, we generated another simulated dataset consisting solely of features with a linear dividing line between the classes.
As before, all models were able to predict the different classes well.
However, once the linear signal was removed, all models had accuracy no better than random guessing (fig @fig:simulation left).
That the nonlinear models only achieved baseline accuracy also indicated that the signal removal method was not injecting nonlinear signal into data where nonlinear signal did not exist.

We also trained the models on a dataset where all features were Gaussian noise as a negative control.
As expected, the models all performed at baseline accuracy both before and after the signal removal process (fig. @fig:simulation right).
This finding supported our decision to perform signal removal on the training and validation sets separately, as removing the signal in the full dataset may introduce predictive signal into this setting (supp. fig. @fig:split-signal-correction).

![
Performance of models in binary classification of simulated data before and after signal removal. Dotted lines indicate expected performance for a naive baseline classifier that predicts the most frequent class.
](./images/simulated_data_combined.svg ){#fig:simulation}

When we ran our models on the signal-removed data from GTEx and Recount3, we found that the neural nets performed better than the baseline while logistic regression did not (fig. @fig:signal_removed top right, supp. fig. @fig:recount-binary-combined).
Similarly, the multiclass problems had the logistic regression model performing poorly, while the nonlinear models had performance that increased with an increase in data while remaining worse than before the linear signal was removed (fig. @fig:signal_removed left).
Likewise, the sex label prediction task showed a marked difference between the neural networks and logistic regression: only the neural networks could learn from the data (fig. @fig:signal_removed bottom right).
In each of the settings, the models performed less well than when run on data without signal, indicating an increase in the problem's difficulty, and logistic regression, in particular, performed no better than random.

![
Performance of models across four classification tasks before and after signal removal
](./images/signal_removed_combined.svg ){#fig:signal_removed}

To verify that our results were not an artifact of how we assigned samples to cross-validation folds, we compared the method we used to assign folds with an alternate method called samplewise splitting.
Samplewise splitting (see [Methods](#methods)) is common in machine learning but leaks information between the train and validation sets when used in transcriptomic tasks.
To avoid this data leakage, we split the dataset at the study level when that information was available.
We found that there is, in fact, a significant degree of performance inflation evident when comparing the samplewise split results to the studywise split results in the Recount3 multiclass setting (supp. fig. @fig:splitting).
While this supports our decision to use studywise splitting, the relative performance of each model stays the same regardless of the data splitting technique.

Another common usage pattern in machine learning is training models on a general dataset and fine-tuning them on a dataset of interest.
To ensure that our results were not made irrelevant by different behavior in the pretraining context, we examined the performance of the models with and without pretraining (supp. fig @fig:pretrain).
To do so, we split the Recount3 data into three sets: pretraining, training, and validation (fig. @fig:workflow bottom), then trained two identically initialized copies of each model.
One copy was trained solely on the training data, while the other was trained on the pretraining data and fine-tuned on the training data.
The pretrained models showed high performance even when trained with small amounts of data from the training set.
However, the nonlinear models did not have a greater performance gain from pretraining than logistic regression, and the balanced accuracy was similar across models.


## Methods

### Data
**GTEx**  
We downloaded 17,382 TPM-normalized samples of bulk RNA-seq expression data from version 8 of GTEx to validate our results.
We then zero-one standardized the data and kept the 5000 most variable genes.
The tissue labels we used for the GTEx dataset were derived from the 'SMTS' column of the sample metadata file.

**Recount3**  
Our Recount3 dataset consisted of bulk RNA-seq data downloaded from the Recount3 compendium [@pmc:PMC86284] during the week of March 14, 2022.
Before filtering, the dataset contained 317,258 samples, each containing 63,856 genes.

To filter out single-cell data, we removed all samples with greater than 75 percent sparsity.
We also removed all samples marked 'scrna-seq' by Recount3's pattern matching method (stored in the metadata as 'recount_pred.pattern.predict.type').

We then converted the data to transcripts per kilobase million using gene lengths from BioMart [@pmc:PMC2649164] and performed standardization to scale each gene's range from zero to one.
We kept the 5,000 most variable genes within the dataset.

Samples were labeled with their corresponding tissues using the 'recount_pred.curated.tissue' field in the Recount3 metadata.
These labels were based on manual curation by the Recount3 authors.
A total of 20324 samples in the dataset had corresponding tissue labels.

Samples were also labeled with their corresponding sex using labels from Flynn et al. [@pmc:PMC8011224].
These labels were derived using pattern matching on metadata from the European Nucleotide Archive [@pmc:PMC3013801].
A total of 23,525 samples in our dataset had sex labels.

**Simulated data**  
We generated three simulated datasets.
The first dataset contained 1000 samples of 5000 features corresponding to two classes. 
Of those features, 2500 contained linear signal.
That is to say that the feature values corresponding to one class were drawn from a standard normal distribution, while the feature values corresponding to the other were drawn from a Gaussian with a mean of 6 and unit variance.

The nonlinear features were generated similarly.
The values for the nonlinear features were drawn from a standard normal distribution for one class, while the second class had values drawn from either a mean six or negative six Gaussian with equal probability.
These features are referred to as "nonlinear" because two dividing lines are necessary to perfectly classify such data, while a linear classifier can only draw one such line per feature.

The second dataset was similar to the first dataset, but it consisted solely of 2500 linear features.
The final dataset contained only values drawn from a standard normal distribution regardless of class label.

### Model architectures
We used three representative models to demonstrate the performance profiles of different model classes.
Each model was implemented in Pytorch [@arxiv:1912.01703], used the same optimizer, and was trained for at most 50 epochs.

The nonlinear models were fully connected neural networks.
The first was a three-layer network with hidden layers of sizes 2500 and 1250.
Our second was a five-layer network, with hidden layers of sizes 2500, 2500, 2500, and 1250.
Both models used ReLU nonlinearities [@https://dl.acm.org/doi/10.5555/3104322.3104425].

The final model was an implementation of logistic regression, a linear model.
As there are known differences in performance between implementations of logistic regression [@pytorch_lr], we implemented ours in PyTorch as similarly to the neural nets as possible to allow for a fair comparison.

### Model training
**Optimization**  
Our models minimized the cross-entropy loss using an Adam [@arxiv:1412.6980] optimizer on mini-batches of data.
They also used inverse frequency weighting to avoid giving more weight to more common classes.

**Regularization**  
The models used early stopping and gradient clipping to regularize their training.
Both neural nets used dropout [@https://jmlr.org/papers/v15/srivastava14a.html] with a probability of 0.5.
The deeper network used batch normalization [@https://proceedings.mlr.press/v37/ioffe15.html] to mitigate the vanishing gradient problem.

**Signal removal**  
We used Limma[@doi:10.1093/nar/gkv007] to remove linear signal associated with tissues in the data.
More precisely, we ran the 'removeBatchEffect' function from Limma on the training and validation sets separately, using the tissue labels as batch labels.

**Hyperparameters**  
The learning rate and weight decay hyperparameters for each model were selected via nested cross-validation over the training folds at runtime.

**Determinism**  
Model training was made deterministic by setting the Python, NumPy, and PyTorch random seeds for each run, as well as setting the PyTorch backends to deterministic and disabling the benchmark mode.

**Logging**  
Model training progress was tracked and recorded using Neptune [@neptune].

### Model Evaluation
In our analyses we use five-fold cross-validation with studywise data splitting.
In a studywise split, the studies are randomly assigned to cross-validation folds such that all samples in a given study end up in a single fold (fig. @fig:workflow bottom).

**Hardware**  
Our analyses were performed on an Ubuntu 18.04 machine and the Colorado Summit compute cluster.
The desktop CPU used was an AMD Ryzen 7 3800xt processor with 16 cores and access to 64 GB of RAM, and the desktop GPU used was an Nvidia RTX 3090.
The Summit cluster used Intel Xeon E5-2680 CPUs and NVidia Tesla K80 GPUs.
From initiating data download to finishing all analyses and generating all figures, the full Snakemake [@doi:10.1093/bioinformatics/bts480] pipeline took around one month to run.

**Recount3 tissue prediction**  
In the Recount3 setting, the multi-tissue classification analyses were trained on the 21 tissues (see Supp. Methods) that had at least ten studies in the dataset.
Each model was trained to determine which of the 21 tissues a given expression sample corresponded to.

To address class imbalance, our models' performance was then measured based on the balanced accuracy across all classes.
Unlike raw accuracy, balanced accuracy (the mean across all classes of the per-class recall) isn't predominantly determined by performance on the largest class in an imbalanced class setting.
For example, in a binary classification setting with 9 instances of class A and 1 instance of class B, successfully predicting 8 of the 9 instances of class A and none of class B yields an accuracy of 0.8 but a balanced accuracy of 0.44.

The binary classification setting was similar to the multi-class one.
The five tissues with the most studies (brain, blood, breast, stem cell, and cervix) were compared against each other pairwise.
The expression used in this setting was the set of samples labeled as one of the two tissues being compared.

The data for both settings were split in a stratified manner based on their study.

**GTEx classification**  
The multi-tissue classification analysis for GTEx used all 31 tissues.
The multiclass and binary settings were formulated and evaluated in the same way as in the Recount3 data.
However, rather than being split studywise, the cross-validation splits were stratified according to the samples' donors.

**Simulated data classification/sex prediction**  
The sex prediction and simulated data classification tasks were solely binary.
Both settings used balanced accuracy, as in the Recount3 and GTEx problems. 

**Pretraining**  
When testing the effects of pretraining on the different model types, we split the data into three sets.
Approximately forty percent of the data went into the pretraining set, forty percent went into the training set, and twenty percent went into the validation set.
The data was split such that each study's samples were in only one of the three sets to simulate the real-world scenario where a model is trained on publicly available data and then fine-tuned on a dataset of interest.

To ensure the results were comparable, we made two copies of each model with the same weight initialization.
The first copy was trained solely on the training data, while the second was trained on the pretraining data, then the training data.
Both models were then evaluated on the validation set.
This process was repeated four more times with different studies assigned to the pretraining, training, and validation sets.


## Conclusion

We performed a series of analyses determining the relative performance of linear and nonlinear models in multiple domains.
Consistent with previous papers [@doi:10.1186/s12859-020-3427-8; @doi:10.1016/j.jclinepi.2019.02.004], linear and nonlinear models performed roughly equivalently in a number of tasks.
That is to say that there are some tasks where linear models perform better, some tasks where nonlinear models have better performance, and some tasks where both model types are equivalent.

When we removed all linear signal in the data, we found that residual nonlinear signal remained.
Not only was this true in simulated data, it also held in GTEx and Recount3 data in several problems.
These results also held in slightly altered problem settings, such as using a pretraining dataset before the training dataset and using samplewise data splitting instead of studywise splitting.
This consistent presence of nonlinear signal demonstrated that the similarity in performance across model types was not due to our problem domains having solely linear signals.

Given that nonlinear signal is present in our problem domains, why doesn't that signal allow nonlinear models to make better predictions?
We believe that the nonlinear signal is either redundant with the linear signal or unreliable enough that nonlinear models choose to learn the linear signal instead.
Determining which of these hypotheses (if either) is true is an interesting avenue for future research.

One limitation of our study is that the results likely do not hold in an infinite data setting.
Deep learning models have been shown to solve complex problems in biology and tend to significantly outperform linear models when given enough data.
However, we do not yet live in a world with vast amounts of gene expression data and accompanying uniform metadata.
Our results are generated on some of the largest labeled expression datasets in existence (Recount3 and GTEx), but our tens of thousands of samples are far from the millions or billions used in deep learning research.

We are also unable to make claims about all problem domains.
There are many potential transcriptomic prediction tasks and many datasets to perform them on.
While we show that nonlinear signal is not always helpful in tissue or sex prediction, and others have shown the same for various disease prediction tasks, there may be problems where nonlinear signal is more important.

Ultimately, our results show that task-relevant nonlinear signal in the data does not necessarily lead nonlinear models to outperform linear ones.
Additionally, we demonstrate that while there are problems where complicated models are helpful, scientists making predictions from expression data should always include simple linear baseline models to determine whether more complex models are warranted.



### Acknowledgements
We would like to thank Alexandra Lee and Jake Crawford for reviewing code that went into this project.
We would also like to thank the past and present members of GreeneLab who gave feedback on this project during lab meetings.
This work utilized resources from the University of Colorado Boulder Research Computing Group, which is supported by the National Science Foundation (awards ACI-1532235 and ACI-1532236), the University of Colorado Boulder, and Colorado State University.


# Supplementary Materials
## Results 

### Recount binary classification 

![                                                                                                                                                                                                          
Comparison of models' binary classification performance on Recount3 data
](./images/recount_binary.svg "Recount binary classification"){#fig:recount-binary}

### Signal removal
While it's possible to remove signal in the full dataset or the train and validation sets independently, we decided to do the latter.
We made this decision because we observed potential data leakage when removing signal from the entire dataset in one go (supp. fig. @fig:split-signal-correction).

![                                                                                                                                                                                                          
Full dataset signal removal in a dataset without signal
](./images/no_signal_sim_signal_removed.svg "Signal removal from data with no signal to begin with"){#fig:split-signal-correction} 

![                                                                                                                                                                                                          
Comparison of models' binary classification performance before and after removing linear signal                                                                                                             
](./images/recount_binary_combined.svg "Recount binary classification before and after signal removal"){#fig:recount-binary-combined}


### Samplewise splitting
![                                                                                                                                                                                                          
Performance of Recount3 multiclass prediction with samplewise train/val splitting
](./images/recount_multiclass_sample_split.svg ){#fig:splitting}

### Recount3 Pretraining 
![                                                                                                                                                                                                          
Performance of Recount3 multiclass prediction with pretraining
](./images/recount_pretraining.svg "Pretraining"){#fig:pretrain} 

## Methods
### Recount3 tissues used
The tissues used from Recount3 were blood, breast, stem cell, cervix, brain, kidney, umbilical cord, lung, epithelium, prostate, liver, heart, skin, colon, bone marrow, muscle, tonsil, blood vessel, spinal cord, testis, and placenta.



## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>
