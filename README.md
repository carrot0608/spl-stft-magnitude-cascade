# Microwave Brain-State Classification Using Soft Probability Fusion

Core implementation for healthy, ischemic and simulated-hemorrhage classification from complex S21 measurements.

- `features.py`: IFFT-envelope STFT maps and absolute-magnitude features.
- `model.py`: the 20,114-parameter STFT-CNN and training augmentation.
- `soft_fusion.py`: training-side epoch selection, magnitude ensemble and three-class soft probability fusion.

`SoftFusionClassifier.fit(spectra, labels, animal_ids, seed, outer_fold, device)` accepts training animals only. `spectra` has shape `(N, 9, 201)` with complex values. Path order is `11,12,13,21,22,23,31,32,33`; the 201 frequencies are uniformly spaced from 10 MHz to 6.5 GHz. Labels are `0=healthy`, `1=ischemia`, `2=simulated hemorrhage`.

Training uses inner leave-one-animal-out folds, five CNNs per inner fold, an 80-epoch ceiling, batch size four and AdamW. The common epoch minimizes pooled inner-validation cross-entropy. The magnitude branch averages logistic regression, shrinkage LDA and PCA/linear-SVM conditional subtype probabilities. Preprocessing is fitted on training data.

`predict_proba(spectra)` returns scores in H/I/B order: `[pH, (1-pH)*pI, (1-pH)*pB]`.

`predict(spectra)` selects the maximum score, resolving ties in H/I/B order. No separate healthy gate threshold is used. For outer leave-one-animal-out evaluation, call `fit` separately for each training-animal set and evaluate the held-out animal with `predict`.

Dependencies are listed in `requirements.txt`. The default device is CUDA; CPU execution uses float32. CUDA training and inference use automatic mixed precision. No dataset or trained weights are distributed here.
