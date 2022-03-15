---
title: The Effects of Nonlinear Signal on Expression-Based Prediction Performance
keywords:
- markdown
- publishing
- manubot
lang: en-US
date-meta: '2022-03-15'
author-meta:
- Benjamin Heil
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
  <meta name="dc.date" content="2022-03-15" />
  <meta name="citation_publication_date" content="2022-03-15" />
  <meta name="dc.language" content="en-US" />
  <meta name="citation_language" content="en-US" />
  <meta name="dc.relation.ispartof" content="Manubot" />
  <meta name="dc.publisher" content="Manubot" />
  <meta name="citation_journal_title" content="Manubot" />
  <meta name="citation_technical_report_institution" content="Manubot" />
  <meta name="citation_author" content="Benjamin Heil" />
  <meta name="citation_author_institution" content="Genomics and Computational Biology Graduate Group, Perelman School of Medicine, University of Pennsylvania" />
  <meta name="citation_author_orcid" content="0000-0002-2811-1031" />
  <meta name="twitter:creator" content="@autobencoder" />
  <link rel="canonical" href="https://greenelab.github.io/linear_models_manuscript/" />
  <meta property="og:url" content="https://greenelab.github.io/linear_models_manuscript/" />
  <meta property="twitter:url" content="https://greenelab.github.io/linear_models_manuscript/" />
  <meta name="citation_fulltext_html_url" content="https://greenelab.github.io/linear_models_manuscript/" />
  <meta name="citation_pdf_url" content="https://greenelab.github.io/linear_models_manuscript/manuscript.pdf" />
  <link rel="alternate" type="application/pdf" href="https://greenelab.github.io/linear_models_manuscript/manuscript.pdf" />
  <link rel="alternate" type="text/html" href="https://greenelab.github.io/linear_models_manuscript/v/8b5bb49e7e82b15cdedd63fd912dab95e6dd92d8/" />
  <meta name="manubot_html_url_versioned" content="https://greenelab.github.io/linear_models_manuscript/v/8b5bb49e7e82b15cdedd63fd912dab95e6dd92d8/" />
  <meta name="manubot_pdf_url_versioned" content="https://greenelab.github.io/linear_models_manuscript/v/8b5bb49e7e82b15cdedd63fd912dab95e6dd92d8/manuscript.pdf" />
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
([permalink](https://greenelab.github.io/linear_models_manuscript/v/8b5bb49e7e82b15cdedd63fd912dab95e6dd92d8/))
was automatically generated
from [greenelab/linear_models_manuscript@8b5bb49](https://github.com/greenelab/linear_models_manuscript/tree/8b5bb49e7e82b15cdedd63fd912dab95e6dd92d8)
on March 15, 2022.
</em></small>

## Authors



+ **Benjamin Heil**<br>
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



## Abstract {.page_break_before}




## Introduction

Transcriptomic data contains a wealth of information about a person's biology, so predicting phenotypes from RNA-seq data is a promising field of research.
Gene expression-based models are already being used to subtype cancer [@doi:10.1200/JCO.2008.18.1370], predict transplant rejections [@doi:10.1161/CIRCULATIONAHA.116.022907], and uncover biases in public data [@pmc:PMC8011224].
In fact, both the capability of machine learning models [@arxiv:2202.05924] and the amount of transcriptomic data available [@doi:10.1038/s41467-018-03751-6; @doi:10.1093/database/baaa073] are increasing rapidly.
It makes sense, then, that neural networks are frequently being used in the transcriptomic prediction space [@doi:10.1038/s41598-019-52937-5; @doi:10.1093/gigascience/giab064; @doi:10.1371/journal.pcbi.1009433].

However, there are two conflicting ideas in the literature regarding the utility of nonlinear models.
One theory is based on the prior biological understanding: the paths linking gene expression to phenotypes are complex [@doi:10.1016/j.semcdb.2011.12.004; @doi:10.1371/journal.pone.0153295], and nonlinear models like neural networks should be more capable of learning that complexity.
Unlike purely linear models such as logistic regression, nonlinear models should be learn more sophisticated representations of the relationships between expression and phenotype.
Accordingly, many have used nonlinear models to learn representations useful for making predictions of phenotypes from gene expression [@doi:10.1128/mSystems.00025-15; @doi:10.1016/j.cmpb.2018.10.004; @doi:10.1186/s12859-017-1984-2].

The other theory disagrees with the first hypothesis: when using expression to make predictions about phenotypes, linear models seem to do as well as or better than nonlinear ones in many cases[@doi:10.1186/s12859-020-3427-8].
While papers of this sort are harder to come by — scientists don't tend to write papers about how their deep learning model was worse than logistic regression — other complex biological problems have also seen linear models prove equivalent to nonlinear ones [@doi:10.1016/j.jclinepi.2019.02.004].

In this paper we demonstrate that both theories have merit.
There is nonlinear signal relating the phenotypes to genotypes, but it doesn't always lead nonlinear models to provide better prediction accuracy.
We construct a system of binary and multi-class classification problems on the Recount3 compendium[@doi:10.1186/s13059-021-02533-6] that allows us to show that linear and nonlinear models have similar accuracy on several (but not all) prediction tasks.
We then remove the linear signals relating the phenotype to gene expression and show that there is in fact nonlinear signal in the data even when the linear models outperform the nonlinear ones.
Finally, we validate the results by testing the same problems on a dataset from GTEx[@doi:10.1038/ng.2653], running controls on simulated data, and examining different problem formulations such as samplewise splitting and pretraining.

In reconciling these two obstensibly conflicting theories, we assist future scientists by showing the importance of trying a linear baseline model before developing a complex nonlinear approach.
While nonlinear models may outperform simpler models at the limit of infinite data, they don't necessarily do so even when trained on the largest datasets publicly available today.


## Results 

### Approach 
We compared the performance of linear and nonlinear models across a number of datasets and on multiple tasks (fig. @fig:workflow top).
Our datasets consisted of gene expression from Recount3 [@doi:10.1186/s13059-021-02533-6] with tissue labels from the recount3 metadata and sex labels from Flynn et al. [@doi:10.1186/s12859-021-04070-2], simulated data, and expression and tissue labels from GTEx[@doi:10.1038/ng.2653].
Before use in model training, we removed scRNA samples, RPM normalized, and zero-one standardized the data (see Methods for more details).

We split our dataset via fivefold cross-validation to evaluate each model on multiple training and validation sets.
In order to avoid leakage between folds, the studies were placed in their entirety into a fold instead of being split across folds (fig. @fig:workflow bottom).
We then ran the models on increasingly large subsets of the training data to determine how model performance is affected by the amount of training data.

To ensure that artifacts specific to a single data split or model initialization don't drive the signal, we run each of our experiments with three different random seeds.
As a result of these different dimensions of variation, each evaluation we perform reflects 150 trained instances of each model.
The three models we selected were logistic regression, a three layer neural network, and a five layer neural network.
Our three layer and five layer networks were chosen to be representative of a fairly shallow and moderately deep network respectively.
Our logistic regression implementation was implemented and optimized as similarly to the neural nets as possible to allow comparisons unbiased by implementation details.
For a comparison against a scikit-learn [@url:https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html] implementation of logistic regression, see Supplementary Results section TODO.

![
Schematic of the model analysis workflow. We evaluate three models on multiple classification problems in three datasets (top). To do so, we use studywise splitting as a default, but also evaluate the effects of samplewise splitting and using a pretraining dataset.
](./images/workflow.svg "Workflow diagram"){#fig:workflow}


### Linear and nonlinear models have similar performance in some tasks
To determine whether linear and nonlinear models performed similarly, we used them for a number of tissue prediction tasks.
First we compared their abilities to differentiate between pairs of tissues (Fig. @fig:recount-binary), and found that their performance was roughly equivalent.
More specifically, given tissue pairs seemed to have maximum accuracy thresholds that models would achieve and then plateau.

In case the accuracy threshold effects were due to the relative easiness of the binary classification task, we selected a harder problem.
Namely, we evaluated the models on their ability to predict which of the 21 most common tissues in the Recount3 dataset a sample belonged to.
In this setting, we found that our five layer network and logistic regression performed roughly the same, while the three layer network had lower accuracy (fig. @fig:recount-multiclass).

![
Comparison of models' binary classification performance before and after removing linear signal
](./images/recount_binary_combined.svg "Recount binary classification before and after signal removal"){#fig:recount-binary}

![
Graph of the Recount3 multiclass classification results. Each point represents the validation set balanced accuracy of a separate trained model. The color of the points and the trend lines shows their corresponding model class, while the dashed line represents the baseline accuracy of random predictions.
](./images/recount_multiclass.svg "Recount multiclass classification"){#fig:recount-multiclass}

### In spite of the similar performance of the models, there is nonlinear signal relevant to the problem
One can imagine a world where all the signal relevant to tissue prediction is linear.
If that were the case, nonlinear models like neural networks would fail to give any advantage in a prediction problem.
To determine whether there is nonlinear signal in our tissue prediction tasks learnable by our neural nets, we used Limma[@doi:10.1093/nar/gkv007] to remove the linear signal associated with each tissue.

When we ran our models on the signal-removed data, we found that while the neural networks manage to perform better than the random classification baseline, the logistic regression models do worse than the baseline (Fig. @fig:recount-binary).
The anticorrelation between the amount of data used and the linear model performance is due to running the signal removal on the full dataset at once.
Because there can be no predictive linear signal in the dataset, any linear model trained to greater than random performance on the training set will necessarily perform worse than random on the rest of the data.
This artifact was selected as the lesser of two evils, as removing signal from the training and validation set poses its own problems (Supp results TODO).

### This nonlinear signal is real, and not an artifact of the signal removal method
We then simulated simple binary classification tasks to ensure that the results weren't due to an unknown signal in the data.
Our initial simulated dataset consisted of two types of features: half of the features had a linear dividing line between the simulated classes while the other half had a nonlinear dividing line.
After training to classify the simulated dataset, all models were able to effectively predict the simulated classes.
After removing the linear signal from the dataset, nonlinear models were still able to easily predict the correct classes, but logistic regression was no better than random (fig @fig:simulation middle).

To ensure that the high performance of the nonlinear models wasn't due to nonlinear signal induced by the correction method, we generated another simulated dataset consiting solely of features with a linear dividing line between the classes.
As before, all models were able to predict the different classes well.
However, once the linear signal was removed all models had accuracy no better than random guessing, indicating that the signal removal method was not generating nonlinear signal (fig @fig:simulation left).

We also trained the models on a dataset where all features were gaussian noise as a negative control.
As expected, the models all performed at baseline accuracy both before and after the signal removal process (fig. @fig:simulation right).

![
Performance of models in binary classification of simulated data before and after signal removal
](./images/simulated_data_combined.svg ){#fig:simulation}

### Our results hold in a cleaner validation dataset
To validate our findings on a separate real dataset, we selected the expression data from GTEx [@doi:10.1038/ng.2653].
Because it was generated by fewer labs with more consistent experimental design across samples, it is a less heterogeneous dataset than the Recount3 compendium.
We trained our models to do binary classification on pairs of the five most common tissues in the dataset, then performed multiclass classification on all 31 tissues present in the dataset.
Likely due to the cleaner nature of the GTEx data, all models were able to perform perfectly on the binary classification tasks (Fig. @fig:gtex bottom)
The harder multitask classification problem showed logistic regression outperforming the five layer neural network, which in turn outperformed the three layer net (fig. @fig:gtex top).

The linear signal removal results on the binary classification problems were consistent with those from the Recount3 compendium.
The neural networks performed less well in the low-data regime, indicating an increase in the difficulty of the problem, and the logistic regression implementation performed no better than random (Fig. @fig:gtex bottom).
Similarly, the multiclass problem had the logistic regression model performing poorly, while the nonlinear models had performance that increased with an increase in data while remaining worse than before the linear signal was removed (Fig. @fig:gtex top).

![
Performance of model on GTEx classification problems. The top figures show the difference in training models in the multiclass setting with and without signal removal, while the bottom figures show binary classification with and without signal removal.
](./images/gtex_combined.svg ){#fig:gtex}

### While linear models seem to be equivalent to nonlinear ones in most problems, that isn't true for all problems
To rule out the possibility that our findings were specific to tissue prediction tasks, we examined models' ability to predict metadata-derived sex (Fig. @fig:sex-prediction).
We used the same experimental setup as in our other Recount3 binary prediction tasks to train the models, but rather than using tissue labels we used metadata-derived sex labels.
In this setting we found that, at least in the 5000-15000 sample range, the nonlinear models outperformed logistic regression.
This result demonstrates that despite the compelling accuracy of linear models, there are still problem settings where nonlinear models perform better.

![
Metadata sex prediction
](./images/sex_prediction.svg ){#fig:sex-prediction}

### The results so far aren't invalidated by a jump in accuracy caused by pretraining
A common usage pattern in machine learning is to train models on a general dataset then fine-tune them on a dataset of interest.
To ensure that our results weren't made irrelevant by different behavior in the pretraining context, we examined the performance of the models with and without pretraining (Supp. fig TODO).
We split our data into three sets: pretraining, training, and validation (Fig. @fig:workflow bottom), then trained two identically initialized copies of each model.
One copy was trained solely on the training data, while the other was trained on the pretraining data then fine-tuned on the training data.

The pretrained models showed high performance even when trained with small amounts of data from the training set.
However, the nonlinear models did not have a greater performance gain from pretraining than logistic regression, and the balanced accuracy was similar across models.
In fact, all models showed lower performance than when using the full training data, as models forget information from previous runs during fine-tuning [@doi:10/ckpftf].

### Our results aren't caused by using a more sophisticated form of data splitting
We considered it possible that our results were an artifact of our method of dataset splitting, and set out to test it.
There is a common method of data splitting we refer to as samplewise splitting (see Methods) that leaks information between the train and validation sets when used in transcriptomic tasks.
To avoid this data leakage, we split the dataset at the study level in our Recount3 analyses.
We found that there is in fact a large degree of performance inflation evident when comparing the sample-split results to the study-split results in the Recount3 multiclass setting (Supp Fig. @fig:splitting).
While this supports our decision to use study-level splitting, the relative performance of each model stays the same regardless of data splitting technique.


## Methods

### Data
#### Recount3
Our first dataset consisted of bulk RNA-seq data downloaded from the recount3 compendium [@pmc:PMC86284] on TODO date.
Before filtering, the dataset contained 317,258 samples, each containing 63,856 genes.

To filter out single-cell data, we removed all samples with a sparsity greater than 75 percent.
We also removed all samples marked 'scrna-seq' by Recount3's pattern matching method (stored in the metadata as 'recount_pred.pattern.predict.type').

To ensure the samples were comparable, we converted the data to transcripts per kilobase million using gene lengths from BioMart [@pmc:PMC2649164].
To ensure the genes' magnitudes were comparable, we performed standardization to scale each gene's range from zero to one.
We kept the 5,000 most variable genes within the dataset.

Samples were labeled with their corresponding tissues using the 'recount_pred.curated.tissue' field in the Recount3 metadata.
These labels were based on manual curation by the Recount3 authors.
A total of 20324 samples in the dataset had corresponding tissue labels.

Samples were also labeled with their corresponding sex using labels from Flynn et al. [@pmc:PMC8011224].
These labels were derived using pattern matching on metadata from the European Nucleotide Archive [@pmc:PMC3013801].
A total of 23,525 samples in our dataset had sex labels.

#### GTEx 
We downloaded 17,382 TPM-normalized samples of bulk RNA-seq expression data from version 8 of GTEx to validate our results.
We then zero-one standardized the data and kept the 5000 most variable genes.
The tissue labels we used for the GTEx dataset were derived from the 'SMTS' column of the sample metadata file.

#### Simulated data
We generated three simulated datasets to ensure the signal removal process was working as expected.
The first dataset contained 1000 samples of 5000 features corresponding to two classes. 
2500 of those features contained linear signal.
That is to say that the feature values corresponding to one class were drawn from a standard normal distribution, while the feature values corresponding to the other were drawn from a Gaussian with a mean of 6 and unit variance.

The nonlinear features were generated similarly.
The values for the nonlinear features were drawn from a standard normal distribution for one class, while the second class had values drawn from either a mean 6 or mean -6 Gaussian with equal probability.
These features are referred to as "nonlinear" because a linear classifier is unable to draw the two dividing lines necessary to correctly classify such data.

The second dataset was similar to the first dataset, but it consisted solely of 2500 linear features.
The final dataset consisted solely of values drawn from a standard normal distribution regardless of class label.

### Model architectures
We use three representative models to demonstrate the performance profiles of different model classes.
Each model was a implemented in Pytorch [@arxiv:1912.01703], used the same optimizer, and was trained for at most 50 epochs.

The nonlinear models were fully connected neural networks.
The first was a three layer network with hidden layers of size 2500 and 1250.
Our second was a five layer network, with hidden layers of size 2500, 2500, 2500, and 1250.
Both models used ReLU nonlinearities [@https://dl.acm.org/doi/10.5555/3104322.3104425].

The final model was an implementation of logistic regression, a linear model.
It was designed to be trained as similarly to the neural nets as possible to allow for a fair comparison.

### Model training
#### Optimization
Our models minimized the cross-entropy loss using an Adam [@arxiv:1412.6980] optimizer on minibatches of data.
They also used inverse frequency weighting to avoid giving more weight to more common classes.

#### Regularization
The models used early stopping and gradient clipping to regularize their training.
Both neural nets used dropout [@https://jmlr.org/papers/v15/srivastava14a.html] with a probability of 0.5.
The deeper network used batch normalization [@https://proceedings.mlr.press/v37/ioffe15.html] to mitigate the vanishing gradient problem.

#### Hyperparameters
The hyperparameters for each model can be found in their corresponding config file at https://github.com/greenelab/saged/tree/master/model_configs/supervised.

#### Determinism
Model trainining was made deterministic by setting the Python, NumPy, and PyTorch random seeds for each run, as well as setting the PyTorch backends to deterministic and disabling the benchmark mode.

#### Logging
Model training progress was tracked and recorded using Neptune [@neptune].

#### Signal removal
We used Limma[@doi:10.1093/nar/gkv007] to remove linear signal associated with tissues in the data.
More precisely, we ran the 'removeBatchEffect' funcion from Limma on the full dataset, using the tissue labels as batch labels.

#### Model Evaluation
In our analyses we use five-fold cross-validation with two types of data splitting.
The first type is samplewise splitting.
In the samplewise paradigm, gene expression samples are split into cross-validation folds at random without respect to which studies they belong to.
In the stratified paradigm, samples are added to folds in chunks.
For example, in a studywise split, the studies are randomly assigned to folds such that all samples in a given study end up in a single fold.

While samplewise splitting is common in the machine learning and computational biology literature, it is ill-suited to gene expression data.
There are study-specific signals in the data, and having samples from the same study in the training and validation sets causes information leakage.
As a result, samplewise splitting inflates the estimated performance of the models.
Studywise splitting avoids leakage by ensuring all the study-specific signals stay within either the training or the validation sets.

#### Hardware
All analyses were performed on an Ubuntu 18.04 machine with 64 GB of RAM.
The CPU used was an AMD Ryzen 7 3800xt processor with 16 cores, and the GPU used was an Nvidia RTX 3090.
The pipeline can be run on a computer with lower specs, but would have to run fewer elements in parallel.
From initiating data download to finishing all analyses and generating all figures, the full Snakemake [@doi:10.1093/bioinformatics/bts480] pipeline takes about TODO days to run.

#### Recount3 tissue prediction
In the Recount3 setting the multitissue classification analyses were trained on the 21 tissues (see Supp. Methods) that had at least 10 studies in the dataset.
Each model was trained to determine which of the 21 tissues a given expression sample corresponded to.
The models' performance was then measured based on the balanced accuracy across all classes.

The binary classification setting was similar.
The five tissues with the most studies (brain, blood, breast, stem cell, and cervix) were compared against each other pairwise.
The expression used in this setting was the set of samples labled as one of the two tissues being compared.

The data for both settings were split in a stratified manner based on study.

#### GTEx classification
The multitissue classification analysis for GTEx used all 31 tissues.
Both the multiclass and binary settings were formulated and evaluated in the same way as in the recount data.
However, rather than being split studywise, the data were stratified according to donor.

#### Simulated data classification/sex prediction
The sex prediction and simulated data classification tasks were solely binary.
Both settings used balanced accuracy, as in the Recount3 and GTEx problems. 

#### Pretraining
In order to test the effects of pretraining on the different model types, we split the data into three sets.
Approximately forty percent of the data went into the pretraining set, forty percent went into the training set, and twenty percent went into the validation set.
The data was split such that each study's samples were in only one of the three sets, to simulate the real-world scenario where a model is trained on a publicly available data then fine-tuned on a dataset of interest.

To evaluate the models, we made two copies of each model with the same weight initialization.
The first copy was trained solely on the training data, while the second was trained on the pretraining data, then the training data.
Both models were then evaluated on the validation set.
This process was then repeated four more times with different studies assigned to the pretraining, training, and validation sets.



## Conclusion

In this paper, we performed a series of analyses determining the relative performance of linear and nonlinear models in multiple domains.
We found that, consistent with previous papers [@doi:10.1186/s12859-020-3427-8; @doi:10.1016/j.jclinepi.2019.02.004], linear and nonlinear models performed roughly equivalently in a number of tasks.
That is to say that there are some tasks where linear models perform better, some tasks where nonlinear models have better performance, and some tasks where both model types are equivalent.

To determine what led to the performance of the two model classes, we removed all linear signal in the data and found that even in situations where both model types had the same performance there was residual signal that only our nonlinear models were capable of learning.
This implies that the results that we observed were not driven by a lack of nonlinear signal.
We then simulated data to ensure that the signal removal method was not inducing nonlinear signal.
We continued by showing that these results held in slightly altered problem settings, such as using a pretraining dataset before the training dataset and using samplewise data splitting instead of studywise splitting.
Finally, we validated our results on different datasets and domains by running the same analyses on GTEx data and predicting sex labels from expression.

We were able to show that there is both linear and nonlinear signal in our datasets, but that the existence of nonlinear signal does not necessarily lead nonlinear models to make higher-accuracy predictions.
Given that there is nonlinear signal that relates expression to tissue types, why is it that such signal doesn't allow models to make better predictions?
We believe that it is because the nonlinear signal is either redundant with the linear signal, or unreliable enough that nonlinear models choose to learn the linear signal instead.

One limitation of our study is that the results likely do not hold in an infinite data setting.
Deep learning models have been shown to solve hard problems in biology and tend to greatly outperform linear models when given enough data.
However, we do not yet live in a world with huge amounts of gene expression data and accompanying uniform metadata.
Our results are generated on some of the largest labeled expression datasets in existence (Recount3 and GTEx), but our tens of thousands of samples are far from the millions or billions used in deep learning research.

We are also unable to make claims about all problem domains.
There are many potential transcriptomic prediction tasks, and many datasets to perform them on.
While we show that nonlinear signal is not always helpful in tissue prediction, and others have shown the same for various disease prediction tasks, there are also problems such as sex metadata prediction where the nonlinear signal seems to be important.

Ultimately, our results show that the existence of task-relevant nonlinear signal in the data does not necessarily lead nonlinear models to outperform linear ones.
Determining what causes this disconnect is an exciting avenue of future research.
Additionally we demonstrate that while there are problems where complicated models are useful, scientists making predictions from expression data should always include simple linear baseline models to determine whether more complex models are warranted.



# Supplementary Materials
## Methods
### Recount3 tissues used
The tissues used from Recount3 were blood, breast, stem cell, cervix, brain, kidney, umbilical cord, lung, epithelium, prostate, liver, heart, skin, colon, bone marrow, muscle, tonsil, blood vessel, spinal cord, testis, and placenta.

### Data exploration
To determine whether our results were driven by an artifact in the data, we performed exploratory data analysis.
First, we looked for whether anything stood out when comparing per-sample performance between models.
Upon doing so, we found that TODO describe after running on all-tissue
When looking at the results, we noticed that some samples were consistently misclassified across models. 
We suspected it might be due to label imbalance, but a confusion matrix showed that not to be the case.
We examined the metadata for attributes that might be correlated with sample prediction hardness, and found that these samples tend to have a lower read quality than other samples.

## Results 

### Signal removal
As mentioned in the main results section, training a more accurate linear model on signal-removed training data leads to a less accurate model on the validation data due to removing signal on the entire dataset at once.
However, the alternative can lead to even worse artifacts, as seen in fig. sup. @fig:split-signal-correction, where the the linear and nonlinear models have random average performance via different methods.
We suspect the swings in model performance in the nonlinear data are due to colinearity in the features.
That is to say that given the option of a number of possible corrections that can be made to remove the linear signal in the data, there is no guarantee that the same one is selected in the train and validation sets.
For that reason, it's possible to end up with results where the model performance varies wildly in different runs of the signal removal method based on the input data.

![                                                                                                                                                                                                          
Sex prediction results when removing signal from training fold and validation fold separately.
](./images/sex_prediction_split_signal.svg "Sex prediction split signal"){#fig:split-signal-correction} 

### Scikit-learn logistic regression
The Pytorch logistic regression implementation we used was designed to be as close to the neural network implementations as possible to ensure the models were comparable.
Accordingly, we were optimizing for similarity of implementation instead of maximal performance.
Scikit-learn, on the other hand, optimizes their models to have the best out-of-box performance they can manage.
To understand the magnitude of the difference, we compared the sklearn logistic regression model to our pytorch models.
We found that it generally outperformed the other models (supp. fig. @fig:sklearn).

![                                                                                                                                                                                                          
TODO description, build figure
](./images/sklearn.svg "Sklearn comparison"){#fig:sklearn} 

### Recount3 Pretraining Figure
![                                                                                                                                                                                                          
Performance of Recount3 multiclass prediction with pretraining
](./images/recount_pretraining.svg "Pretraining"){#fig:pretrain} 

### Samplewise splitting
![                                                                                                                                                                                                          
Performance of models with different data splitting                                                                                                                                                         
](./images/recount_multiclass_sample_split.svg ){#fig:splitting}


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>
